# base agent for random, heuristic, and Seq2Seq models

class BaseAgent:
    def __init__(self):
        self.score = 0
        self.moves = 0
        self.question_cost = 0

    def act(self, ob, reward, done, info):
        raise NotImplementedError

    def reset(self, env):
        raise NotImplementedError
