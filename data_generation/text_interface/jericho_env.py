import json
import os

import numpy as np

from data_generation.text_interface.env import HMTEnv


class HMTJerichoEnv:
    """
    edited from JerichoEnv
    """

    def __init__(self,
                 json_path,
                 split,
                 fully,
                 step_limit=None,
                 get_valid=True,
                 cache=None,
                 seed=None,
                 start_from_reward=0,
                 start_from_wt=0,
                 log=None,
                 args=None):
        self.json_path = json_path
        self.seed = seed
        self.file = None
        self.env = None
        self.bindings = None
        self.steps = 0
        self.step_limit = step_limit
        self.get_valid = get_valid
        self.max_score = 0
        self.end_scores = []
        self.cache = cache
        self.traj = []
        self.full_traj = []
        self.on_trajectory = True
        self.start_from_reward = start_from_reward
        self.start_from_wt = start_from_wt

        self.log = log
        self.cache_hits = 0
        self.ngram_hits = 0
        self.ngram_needs_update = False
        if args:
            self.filter_drop_acts = args.filter_drop_acts
        else:
            self.filter_drop_acts = None
        self.args = args
        self.split = split
        self.fully = fully
        np.random.seed(self.seed)

    def step(self, action):
        '''

        :param action: natural language command (string)
        :return:
            ob: observation (string)
            reward: reward (int)
            done: check goal (bool)
            info
        '''
        ob, reward, done, info = self.env.step(action)
        ob = ob.replace('#', ' ')
        ob = ob.replace(',', '')
        ob = ' '.join(ob.split())

        # Initialize with default values
        info['look'] = 'unknown'
        info['inv'] = 'unknown'
        info['valid'] = ['wait', 'yes', 'no']
        if not done:
            look = self.env.get_look(fully=self.fully)
            look = look.replace('#', ' ')
            look = look.replace(',', '')
            info['look'] = look
            inv = self.env.get_inventory()
            inv = inv.replace('#', ' ')
            inv = inv.replace(',', '')
            info['inv'] = inv
            if self.get_valid:
                valid = self.env.get_valid_actions()
                valid = [action.replace('#', ' ') for action in valid]
                valid = [action.replace(',', '') for action in valid]
                if len(valid) == 0:
                    valid = ['wait', 'yes', 'no']
                info['valid'] = valid
        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        self.max_score = max(self.max_score, info['score'])
        if done:
            self.end_scores.append(info['score'])
        return ob, reward, done, info

    def reset(self):
        self.file = self.sample_file()
        del self.env
        self.env = HMTEnv(self.file, self.fully)
        initial_ob, info = self.env.reset()
        ob = initial_ob.replace('#', ' ')
        ob = ob.replace(',', '')
        ob = ' '.join(ob.split())

        look = self.env.get_look(fully=self.fully)
        look = look.replace('#', ' ')
        look = look.replace(',', '')
        info['look'] = look
        inv = self.env.get_inventory()
        inv = inv.replace('#', ' ')
        inv = inv.replace(',', '')
        info['inv'] = inv
        valid = self.env.get_valid_actions()
        valid = [action.replace('#', ' ') for action in valid]
        valid = [action.replace(',', '') for action in valid]
        info['valid'] = valid
        self.steps = 0
        self.max_score = 0
        return ob, info

    def get_end_scores(self, last=1):
        last = min(last, len(self.end_scores))
        return sum(self.end_scores[-last:]) / last if last else 0

    def close(self):
        return

    def get_score(self):
        return self.env.info['score']

    def sample_file(self):
        # if self.json_path is not folder,
        if self.json_path[-5:] == '.json':      # if it is already a specified file, no need to sample
            return self.json_path
        with open(self.json_path + '/../HandMeThat_data_info.json', 'r') as f:
            files = json.load(f)[-1][self.split]

        file = np.random.choice(files)
        return os.path.join(self.json_path, file)
