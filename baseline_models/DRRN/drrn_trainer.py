# Built-in Imports
import time
import heapq as pq
import statistics as stats
import random
import copy
from typing import Callable, List, Dict, Union
import json
import numpy as np
import os

# Libraries
import torch
from jericho.util import clean

# Custom imports
from trainers import Trainer

from utils.util import process_action, check_exists, load_object, save_object
from utils.vec_env import VecEnv
from utils.memory import Transition
import utils.logger as logger
import utils.ngram as Ngram

from data_generation.text_interface.jericho_env import HMTJerichoEnv


class DrrnTrainer(Trainer):
    def __init__(
            self,
            tb: logger.Logger,
            log: Callable[..., None],
            agent,
            envs: VecEnv,
            eval_env: HMTJerichoEnv,
            args: Dict[str, Union[str, int, float]]
    ):
        super().__init__(tb, log, agent, envs, eval_env, args)

        # Action model settings
        self.use_action_model = args.use_action_model
        self.rotating_temp = args.rotating_temp
        if self.use_action_model:
            Ngram.init_trainer(self, args)

        self.collected_trajs = []
        self.full_traj_folder = args.output_dir.split('/')[-1][:-3]
        self.dump_traj_freq = args.dump_traj_freq

        self.args = args

    def setup_env(self, envs):
        """
        Setup the environment.
        """
        obs, infos = envs.reset()
        for idx, info in enumerate(infos):
            infos[idx]['look'] = ' '.join(info['look'].split())
            infos[idx]['inv'] = ' '.join(info['inv'].split())
        if self.use_action_model:
            states = self.agent.build_states(
                obs, infos, ['reset'] * 8, [[]] * 8)
        else:
            states = self.agent.build_states(obs, infos)
        valid_ids = [self.agent.encode(info['valid']) for info in infos]
        transitions = [[] for info in infos]

        return obs, infos, states, valid_ids, transitions

    def update_envs(self, action_strs, action_ids, states, max_score: int,
                    transitions, obs, infos):
        """
        TODO
        """
        next_obs, next_rewards, next_dones, next_infos = self.envs.step(
            action_strs)

        if self.use_action_model:
            next_states = self.agent.build_states(
                next_obs, next_infos, action_strs, [state.acts for state in states])
        else:
            next_states = self.agent.build_states(next_obs, next_infos)

        next_valids = [self.agent.encode(next_info['valid'])
                       for next_info in next_infos]

        self.envs.add_full_traj(
            [
                (ob, info['look'], info['inv'], act, r) for ob, info, act, r in
                zip(obs, infos, action_strs, next_rewards)
            ]
        )

        if self.use_action_model:
            # Add to environment trajectory
            trajs = self.envs.add_traj(
                list(map(lambda x: process_action(x), action_strs)))

            for next_reward, next_done, next_info, traj in zip(next_rewards, next_dones, next_infos, trajs):
                # Push to trajectory memory if reward was positive and the episode didn't end yet
                if next_reward > 0:
                    Ngram.push_to_traj_mem(self, next_info, traj)

        for i, (next_ob, next_reward, next_done, next_info, state, next_state, next_action_str) in enumerate(
                zip(next_obs, next_rewards, next_dones, next_infos, states, next_states, action_strs)):
            # Log
            self.log('Action_{}: {}'.format(
                self.steps, next_action_str), condition=(i == 0))
            self.log("Reward{}: {}, Score {}, Done {}".format(
                self.steps, next_reward, next_info['score'], next_done), condition=(i == 0))
            # self.log('Obs{}: {} Inv: {} Desc: {}'.format(
            #     self.steps, clean(next_ob), clean(next_info['inv']),
            #     clean(next_info['look'])), condition=(i == 0))

            transition = Transition(
                state, action_ids[i], next_reward, next_state, next_valids[i], next_done)
            transitions[i].append(transition)
            self.agent.observe(transition)

            if next_done:
                self.tb.logkv_mean('EpisodeScore', next_info['score'])
                if next_info['score'] >= max_score:  # put in alpha queue
                    if next_info['score'] > max_score:
                        self.agent.memory.clear_alpha()
                        max_score = next_info['score']
                    for transition in transitions[i]:
                        self.agent.observe(transition, is_prior=True)
                transitions[i] = []

                if self.use_action_model:
                    Ngram.log_recovery_metrics(self, i)

                    if self.envs.get_ngram_needs_update(i):
                        Ngram.update_ngram(self, i)

                if self.rotating_temp:
                    self.agent.network.T[i] = random.choice([1.0, 2.0, 3.0])

                next_infos = list(next_infos)
                # add finished to trajectory to collection
                traj = self.envs.add_full_traj_i(
                    i, (next_obs[i], next_infos[i]['look'], next_infos[i]['inv']))
                self.collected_trajs.append(traj)

                next_obs[i], next_infos[i] = self.envs.reset_one(i)

                if self.use_action_model:
                    next_states[i] = self.agent.build_skip_state(
                        next_obs[i], next_infos[i], 'reset', [])
                else:
                    next_states[i] = self.agent.build_state(
                        next_obs[i], next_infos[i])

                next_valids[i] = self.agent.encode(next_infos[i]['valid'])

        return next_infos, next_states, next_valids, max_score, next_obs

    def _wrap_up_episode(self, info, env, max_score, transitions, i):
        """
        Perform final logging, updating, and building for next episode.
        """
        # Logging & update
        self.tb.logkv_mean('EpisodeScore', info['score'])
        if env.max_score >= max_score:
            for t in transitions[i]:
                self.agent.observe(t, is_prior=True)
        transitions[i] = []
        self.env_steps += info["moves"]

        # Build ingredients for next step
        next_ob, next_info = env.reset()
        if self.use_action_model:
            next_state = self.agent.build_skip_state(
                next_ob, next_info, [], 'reset')
        else:
            next_state = self.agent.build_state(next_ob, next_info)
        next_valid = self.agent.encode(next_info['valid'])

        return next_state, next_valid, next_info

    def train(self):
        """
        Train the agent.
        """
        start = time.time()
        max_score, max_eval, self.env_steps = 0, 0, 0

        obs, infos, states, valid_ids, transitions = self.setup_env(self.envs)

        for step in range(1, self.max_steps + 1):
            self.steps = step
            self.log("Step {}".format(step))
            action_ids, action_idxs, action_qvals = self.agent.act(states,
                                                                   valid_ids,
                                                                   [info['valid'] for info in infos],
                                                                   sample=True)

            action_strs = [
                info['valid'][idx] for info, idx in zip(infos, action_idxs)
            ]

            # Log envs[0]
            s = ''
            for idx, (act, val) in enumerate(
                    sorted(zip(infos[0]['valid'], action_qvals[0]),
                           key=lambda x: x[1],
                           reverse=True), 1):
                s += "{}){:.2f} {} ".format(idx, val.item(), act)
            self.log('Q-Values: {}'.format(s))
            # Update all envs
            infos, next_states, next_valids, max_score, obs = self.update_envs(
                action_strs, action_ids, states, max_score, transitions, obs, infos)
            states, valid_ids = next_states, next_valids
            self.end_step(step, start, max_score, action_qvals, max_eval)

    def end_step(self, step: int, start, max_score: int, action_qvals,
                 max_eval: int):
        """
        TODO
        """
        if step % self.q_update_freq == 0:
            self.update_agent()

        if step % self.target_update_freq == 0:
            self.agent.transfer_weights()

        if step % self.log_freq == 0:
            # rank_metrics = self.evaluate_optimal()
            rank_metrics = dict()
            self.write_to_logs(step, start, self.envs, max_score, action_qvals,
                               rank_metrics)

        # Save model weights etc.
        MODEL_DIR = os.path.join(self.args.save_path, self.args.model, self.args.observability)

        if step % self.checkpoint_freq == 0:
            self.agent.save(int(step / self.checkpoint_freq), MODEL_DIR, self.args.exp_name)

        if self.use_action_model:
            Ngram.end_step(self, step)

    def write_to_logs(self, step, start, envs, max_score, qvals, rank_metrics,
                      *args):
        """
        Log any relevant metrics.
        """
        self.tb.logkv('Step', step)
        self.tb.logkv('Env Steps', self.env_steps)
        # self.tb.logkv('Beta', self.agent.network.beta)
        for key, val in rank_metrics.items():
            self.tb.logkv(key, val)
        self.tb.logkv("FPS", int((step * len(envs)) / (time.time() - start)))
        self.tb.logkv("EpisodeScores100", self.envs.get_end_scores().mean())
        self.tb.logkv('MaxScore', max_score)
        self.tb.logkv('#UniqueActs', self.envs.get_unique_acts())
        self.tb.logkv('#CacheEntries', self.envs.get_cache_size())

        if self.use_action_model:
            Ngram.log_metrics(self)

        self.tb.dumpkvs()

    def eval(self, fully, level=None):
        eval_results = list()
        step_limit = 8      # todo
        data_path = self.args.data_path
        data_dir_name = self.args.data_dir_name
        data_info = data_path + '/' + 'HandMeThat_data_info.json'
        with open(data_info, 'r') as f:
            json_str = json.load(f)
        eval_files = json_str[-1][self.args.eval_split]
        if level:
            eval_files = [file for file in eval_files if level in file]
        eval_files = np.random.permutation(eval_files)

        eval_files = eval_files[:50]

        for filename in eval_files:
            path = data_path + '/' + data_dir_name + '/' + filename
            eval_env = HMTJerichoEnv(path, None, fully, step_limit=step_limit, get_valid=True)
            print(eval_env.json_path)

            eval_envs = VecEnv(1, self.eval_env)
            obs, infos, states, valid_ids, transitions = self.setup_env(eval_envs)
            for step in range(0, step_limit):
                with torch.no_grad():
                    action_ids, action_idxs, action_qvals = self.agent.act(states,
                                                                           valid_ids,
                                                                           [info['valid'] for info in infos],
                                                                           sample=True)
                    action_strs = [
                        info['valid'][idx] for info, idx in zip(infos, action_idxs)
                    ]
                    next_obs, next_rewards, next_dones, next_infos = eval_envs.step(action_strs)
                    print(step, action_strs[0], next_rewards)
                    next_valids = [self.agent.encode(next_info['valid']) for next_info in next_infos]
                    next_state = self.agent.build_state(next_obs[0], next_infos[0])
                    states, valid_ids = [next_state], next_valids
                    infos = next_infos
                    if next_dones[0] or next_infos[0]['score'] > 0:
                        break
            eval_results.append((infos[0]['moves'], infos[0]['score'], infos[0]['question_cost']))
            for ps in eval_envs.ps:
                ps.terminate()
        print(eval_results)
        scores = [result[1] for result in eval_results]
        scores = np.array(scores)
        average = np.mean(scores)
        success = np.where(scores > 50)
        success_rate = float(len(success[0]) / len(scores))
        if success_rate != 0:
            average_score_when_success = np.mean(scores[success])
        else:
            average_score_when_success = 0.0
        print('{}, {}, {}'.format(self.args.model, self.args.level, self.args.observability))
        print('Average Score:', average)
        print('Success Rate:', success_rate)
        print('Average Score On Success Cases:', average_score_when_success)
        return eval_results
