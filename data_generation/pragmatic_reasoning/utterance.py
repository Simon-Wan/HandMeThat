from data_generation.pragmatic_reasoning.rsa_utils import *
import numpy as np


def get_all_utterances(meanings):
    utterances = []
    for meaning in meanings:
        if meaning[0] == 'move_to':
            if ('move_to', meaning[1], 'none') not in utterances:
                utterances.append(('move_to', meaning[1], 'none'))
    return meanings + utterances


def remove_low_probs(meanings, probs, utterances, hierarchy):
    short_meanings = list()
    short_probs = list()
    short_utterances = list()
    for idx, p in enumerate(probs):
        if p > 1e-3:
            short_probs.append(p)
            short_meanings.append(meanings[idx])
    '''
    for idx, u in enumerate(utterances):    # todo: rewrite
        append = False
        for m in short_meanings:
            if dict_dominate_dict(m, u, hierarchy):
                append = True
                break
        if append:
            short_utterances.append(u)
    '''
    repeated_utterances = list()
    for m in short_meanings:
        repeated_utterances += get_utterances_under_meaning(m, hierarchy)

    for u in repeated_utterances:
        if u not in short_utterances:
            short_utterances.append(u)

    short_p_array = np.array(short_probs)
    short_p_array /= sum(short_p_array)
    return short_meanings, short_p_array, short_utterances


def get_utterances_under_meaning(m, hierarchy):
    utterances = list()
    desc = m[1].copy()
    if 'class' in desc.keys():
        cls = desc.pop('class')
        tmp_dict_list = [{}, {'class': cls}]
        for key in desc.keys():
            tmp_dict_list = append_dict(tmp_dict_list, key, desc[key])
    elif 'subclass' in desc.keys():
        subcls = desc.pop('subclass')
        cls = find_class(subcls, hierarchy)
        tmp_dict_list = [{}, {'class': cls}, {'subclass': subcls}]
        for key in desc.keys():
            tmp_dict_list = append_dict(tmp_dict_list, key, desc[key])
    elif 'type' in desc.keys():
        t = desc.pop('type')
        subcls, cls = find_subclass_and_class(t, hierarchy)
        tmp_dict_list = [{}, {'class': cls}, {'subclass': subcls}, {'type': t}]
        for key in desc.keys():
            tmp_dict_list = append_dict(tmp_dict_list, key, desc[key])
    else:
        tmp_dict_list = [{}]
        for key in desc.keys():
            tmp_dict_list = append_dict(tmp_dict_list, key, desc[key])
    if m[0] == 'bring_me':
        return [('bring_me', d, None) for d in tmp_dict_list]
    elif m[0] == 'move_to':
        return [('move_to', d, None) for d in tmp_dict_list] + [('move_to', d, m[2]) for d in tmp_dict_list]
    else:
        return [('change_state', d, m[2]) for d in tmp_dict_list]

