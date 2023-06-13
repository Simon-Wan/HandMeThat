import numpy as np
from baseline_models.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self):
        super(RandomAgent, self).__init__()

    def act(self, ob, reward, done, info):
        self.score = info['score']
        self.moves = info['moves']
        self.question_cost = info['question_cost']
        action = np.random.choice(info['valid'])
        return action

    def reset(self, env=None):
        self.score = 0
        self.moves = 0
        self.question_cost = 0

