from baseline_models.base_agent import BaseAgent
from data_generation.text_interface.env import HMTEnv
from baseline_models.Seq2Seq.seq2seq_layers import agent_command_generation_greedy_generation
STEP_LIMIT = 40


class Seq2SeqAgent(BaseAgent):
    def __init__(self, agent):
        self.score = 0
        self.moves = 0
        self.question_cost = 0
        self.solution = list()
        self.env = None
        self.game = None
        self.object_dict = None
        self.agent = agent

    def act(self, ob, reward, done, info):
        inv = info['inv']
        idx = inv.find('Recall your task:')
        task_description = inv[idx+17:]
        # import ipdb; ipdb.set_trace()
        actions, _ = agent_command_generation_greedy_generation(self.agent, [ob], [task_description], None)
        return actions[0]

    def reset(self, env: HMTEnv):
        return
