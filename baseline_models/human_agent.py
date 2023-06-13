from baseline_models.base_agent import BaseAgent


class HumanAgent(BaseAgent):
    def __init__(self):
        super(HumanAgent, self).__init__()

    def act(self, ob, reward, done, info):
        self.score = info['score']
        self.moves = info['moves']
        self.question_cost = info['question_cost']
        # print(ob)
        # print(info)
        action = input()
        return action

    def reset(self, env=None):
        self.score = 0
        self.moves = 0
        self.question_cost = 0

