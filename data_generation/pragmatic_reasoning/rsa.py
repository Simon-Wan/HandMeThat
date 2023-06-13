from data_generation.utils import *
from data_generation.pragmatic_reasoning.rsa_utils import *
import numpy as np


def get_utterance(meaning_idx, meanings, utterances, hierarchy, probs, depth=10, alpha=BETA_query2):
    costs = get_utterances_costs(utterances)
    visualize = {'speaker': [], 'listener': []}
    s = get_pragmatic_speaker(visualize, depth, meanings, utterances, hierarchy, probs, costs, alpha)
    utterance_idx = np.random.choice(range(len(utterances)), p=s[meaning_idx])
    return utterance_idx, utterances[utterance_idx]


def get_utterances_costs(utterances):
    costs = np.zeros((len(utterances)))
    for idx, u in enumerate(utterances):
        arg = True
        if u[0] == 'bring_me' or not u[2]:
            arg = False
        costs[idx] = utterance_cost(u[0], u[1], arg)
    return costs


def estimate_meaning(utterance_idx, utterances, meanings, hierarchy, probs, depth=10, alpha=BETA_query2):
    costs = get_utterances_costs(utterances)
    l = get_pragmatic_listener(None, depth, meanings, utterances, hierarchy, probs, costs, alpha)
    meaning_idx = np.random.choice(range(len(meanings)), p=l.T[utterance_idx])
    return meanings[meaning_idx]


def sample_meaning(meaning_list, probs):
    meaning_idx = np.random.choice(range(len(meaning_list)), p=probs)
    return meaning_idx, meaning_list[meaning_idx]


def get_pragmatic_speaker(visualize, depth, meanings, utterances, hierarchy, probs, costs, alpha=BETA_query2):
    if depth == 0:
        s0 = np.zeros((len(meanings), len(utterances)))
        for m_idx, m in enumerate(meanings):
            for u_idx, u in enumerate(utterances):
                if dict_dominate_dict(m, u, hierarchy):
                    s0[m_idx, u_idx] = 1
        return s0   # M * U
    else:
        l = get_pragmatic_listener(visualize, depth - 1, meanings, utterances, hierarchy, probs, costs, alpha)
        s = np.exp(alpha * (np.log(l) - costs).T)
        s /= sum(s)
        if visualize:
            visualize['speaker'].append(s.T)
        return s.T  # M * U


def get_pragmatic_listener(visualize, depth, meanings, utterances, hierarchy, probs, costs, alpha=BETA_query2):
    s = get_pragmatic_speaker(visualize, depth, meanings, utterances, hierarchy, probs, costs, alpha)
    l = np.matmul(np.diag(probs), s)
    l /= sum(l)
    if visualize:
        visualize['listener'].append(l)
    return l        # M * U

