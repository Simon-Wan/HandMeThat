import datetime
import os
import json
import numpy as np
from alfworld.agents.agent import TextDAggerAgent
from alfworld.agents.modules.generic import HistoryScoreCache, EpisodicCountingMemory, ObjCentricEpisodicMemory

from baseline_models.Seq2Seq.seq2seq_layers import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(config, args):
    # import ipdb; ipdb.set_trace()

    time_1 = datetime.datetime.now()
    agent = TextDAggerAgent(config)
    # todo: add layers
    add_layers(agent.online_net)

    MODEL_DIR = os.path.join(args.save_path, args.model, args.observability)
    MAX_TRAIN_STEP = args.max_train_step
    REPORT_FREQUENCY = args.report_frequency

    episode_no = 0
    running_avg_dagger_loss = HistoryScoreCache(capacity=500)

    agent.load_from_tag = args.load_from_tag
    agent.load_pretrained = bool(args.load_pretrained)
    if agent.load_pretrained:
        if os.path.exists(MODEL_DIR + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(MODEL_DIR + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()
            episode_no = int(agent.load_from_tag.split('_')[-1])
    # load dataset
    # push experience into replay buffer (dagger)
    data_info = args.data_path + '/HandMeThat_data_info.json'
    if args.observability == 'fully':
        fully = True
    elif args.observability == 'partially':
        fully = False
    else:
        raise Exception('Unknown observability!')

    with open(data_info, 'r') as f:
        json_str = json.load(f)
    train_files = json_str[-1]['train']
    train_files = np.random.permutation(train_files)    # random permutation for input
    for file in train_files:   # todo
        with open(args.data_path + '/' + args.data_dir_name + '/' + file, 'r') as f:
            json_str = json.load(f)
            actions = json_str['demo_actions']
            full_observations = json_str['demo_observations_fully']
            partial_observations = json_str['demo_observations_partially']
            task = json_str['task_description']
            if args.given_goal:
                task += ' goal: ' + json_str['_goal']
            if args.given_subgoal:
                task += ' subgoal: ' + get_subgoal(json_str['_goal'], json_str['_objects_in_meaning'])
            trajectory = []
            for i in range(len(actions)):
                full_obs = full_observations[i]
                full_obs_list = full_obs.split()
                if len(full_obs_list) > 1000:
                    full_obs_list = full_obs_list[:500] + full_obs_list[-500:]     # avoid too long obs
                full_obs = ' '.join(full_obs_list)
                partial_obs = partial_observations[i]
                partial_obs_list = partial_obs.split()
                if len(partial_obs_list) > 1000:
                    partial_obs_list = partial_obs_list[:500] + partial_obs_list[-500:]  # avoid too long obs
                partial_obs = ' '.join(partial_obs_list)

                if fully:
                    obs = [partial_obs, full_obs]
                else:
                    obs = [partial_obs, partial_obs]
                action = actions[i]
                trajectory.append([obs, task, None, action, None])
            agent.dagger_memory.push(trajectory)

    while True:
        print(episode_no, datetime.datetime.now())
        if episode_no > MAX_TRAIN_STEP:
            break

        agent.train()
        # import ipdb; ipdb.set_trace()
        for i in range(4):
            dagger_loss = agent_update_dagger(agent)
            print('dagger loss:', dagger_loss)
            if dagger_loss is not None:
                running_avg_dagger_loss.push(dagger_loss)

        report = (episode_no % REPORT_FREQUENCY == 0 and episode_no > 0)
        episode_no += 10
        if not report:
            continue
        time_2 = datetime.datetime.now()
        print("Episode: {:3d} | time spent: {:s} | loss: {:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], running_avg_dagger_loss.get_avg()))

        model_name = '{}_weights_{}.pt'.format(args.observability, episode_no - 10)
        agent.save_model_to_path(MODEL_DIR + '/' + model_name)


def get_subgoal(goal, objects):
    subgoals = list()
    start = list()
    end = list()
    left = 0
    right = 0
    for idx, letter in enumerate(goal):
        if letter == '(':
            if (left - right) == 1:
                start.append(idx)
            left += 1
        if letter == ')':
            if (left - right) == 2:
                end.append(idx)
            right += 1
    for i in range(len(start)):
        subgoals.append(goal[start[i]:end[i]+1])
    obj_type = objects[0][0].split('#')[0]
    related_subgoals = list()
    for subgoal in subgoals:
        if obj_type in subgoal:
            related_subgoals.append(subgoal)
    return '; '.join(related_subgoals)
