import pdsketch as pds
import numpy as np
from data_generation.utils import obj_in_dict, dict_dominate_dict, power_set
import math
from data_generation.pragmatic_reasoning.rsa_utils import *

INF = 1000
ability_verb_mapping = {'toggleable': 'toggle', 'cookable': 'heat', 'freezable': 'cool', 'sliceable': 'slice',
                        'soakable': 'soak', 'stainable': 'clean', 'dustyable': 'clean', 'openable': 'open'}


def generate_reasonable_meaning(object_dict, positions, quest_type=None):
    locations = positions.copy()
    ability_verb_mapping = {'toggleable': 'toggle', 'cookable': 'heat', 'freezable': 'cool', 'sliceable': 'slice',
                            'soakable': 'soak', 'stainable': 'clean', 'dustyable': 'clean', 'openable': 'open'}
    meaning_list = {'bring_me': [], 'move_to': {}, 'change_state': {}}
    for loc in locations:
        meaning_list['move_to'][loc] = []
    for verb in ability_verb_mapping.values():
        meaning_list['change_state'][verb] = []
    for name in object_dict.keys():
        if object_dict[name]['class'] == 'LOCATION':
            continue
        abilities = [a for a in object_dict[name]['ability'] if
                     a not in ['has-ontop', 'has-inside', 'has-size', 'has-color']]
        states = list(object_dict[name]['states'].keys())
        all_attributes = ['type', 'position'] + states
        for t in power_set(all_attributes):
            new_meaning = dict()
            for attr in t:
                if attr == 'type':
                    continue
                if attr == 'position':
                    if 'inside' in object_dict[name].keys():
                        new_meaning['inside'] = object_dict[name]['inside']
                    if 'ontop' in object_dict[name].keys():
                        new_meaning['ontop'] = object_dict[name]['ontop']
                    continue
                new_meaning[attr] = object_dict[name]['states'][attr]
            if 'type' in t:
                meaning_add_class = new_meaning.copy()
                meaning_add_class['class'] = object_dict[name]['class']
                if meaning_add_class not in meaning_list['bring_me']:
                    meaning_list['bring_me'].append(meaning_add_class)
                for loc in locations:
                    if meaning_add_class not in meaning_list['move_to'][loc]:
                        meaning_list['move_to'][loc].append(meaning_add_class)
                for ability in abilities:
                    if 'cleanable' in ability:
                        continue
                    if meaning_add_class not in meaning_list['change_state'][ability_verb_mapping[ability]]:
                        meaning_list['change_state'][ability_verb_mapping[ability]].append(meaning_add_class)
                meaning_add_subclass = new_meaning.copy()
                meaning_add_subclass['subclass'] = object_dict[name]['subclass']
                if meaning_add_subclass not in meaning_list['bring_me']:
                    meaning_list['bring_me'].append(meaning_add_subclass)
                for loc in locations:
                    if meaning_add_class not in meaning_list['move_to'][loc]:
                        meaning_list['move_to'][loc].append(meaning_add_subclass)
                for ability in abilities:
                    if 'cleanable' in ability:
                        continue
                    if meaning_add_class not in meaning_list['change_state'][ability_verb_mapping[ability]]:
                        meaning_list['change_state'][ability_verb_mapping[ability]].append(meaning_add_subclass)
                new_meaning['type'] = object_dict[name]['type']

            if new_meaning not in meaning_list['bring_me']:
                meaning_list['bring_me'].append(new_meaning)
            for loc in locations:
                if new_meaning not in meaning_list['move_to'][loc]:
                    meaning_list['move_to'][loc].append(new_meaning)
            for ability in abilities:
                if 'cleanable' in ability:
                    continue
                if new_meaning not in meaning_list['change_state'][ability_verb_mapping[ability]]:
                    meaning_list['change_state'][ability_verb_mapping[ability]].append(new_meaning)

    result = dict()
    if quest_type:
        for key in meaning_list.keys():
            if quest_type == key:
                result[key] = meaning_list[key].copy()
            elif key == 'bring_me':
                result[key] = []
            else:
                result[key] = {}
    # meaning_list['change_state'].pop('open')
    return result


def get_robot_operators_upon_goal(cur_extended_state, translator, strips_operators, strips_goal, remaining_steps,
                                  rel_targets, hff, quest_type=None):
    robot_operators_dict = {'bring_me': {}, 'move_to': {}, 'change_state': {}}
    reward_dict = {'bring_me': {}, 'move_to': {}, 'change_state': {}}
    all_operators = [op for op in strips_operators if op.applicable(cur_extended_state)]
    useful_objects = {'bring_me': [], 'move_to': [], 'change_state': []}
    for op in all_operators:
        obj_name = op.raw_operator.arguments[1]
        if op.raw_operator.name == 'robot-move-obj-to-human':
            arg = None
            op_type = 'bring_me'
            if op_type != quest_type:
                continue
        elif op.raw_operator.name in ['robot-move-obj-from-rec-into-rec', 'robot-move-obj-from-rec-onto-rec']:
            arg = op.raw_operator.arguments[3]
            op_type = 'move_to'
            if op_type != quest_type:
                continue
        else:
            arg = op.raw_operator.name.split('-')[1]
            op_type = 'change_state'
            if op_type != quest_type:
                continue
        # bypass many useless operators
        cont = True

        if '#' in obj_name:
            for t in rel_targets['MOVABLE']:
                if obj_name.find(t) != -1:
                    cont = False
                    break
        else:
            for t in rel_targets['POSITION']:
                if obj_name.find(t) != -1:
                    cont = False
                    break
        if cont:
            reward_dict[op_type][(obj_name, arg)] = - remaining_steps
            continue

        if len(op.raw_operator.arguments) > 3:
            target_position = op.raw_operator.arguments[3]
            found = False
            for t in rel_targets['POSITION']:
                if target_position.find(t) != -1:
                    found = True
                    break
            if not found:
                reward_dict[op_type][(obj_name, arg)] = - remaining_steps
                continue

        robot_operators_dict[op_type][(obj_name, arg)] = op

        cur_task = pds.strips.GStripsTask(op.apply(cur_extended_state), strips_goal, strips_operators, is_relaxed=False)
        cur_task = translator.relevance_analysis(cur_task)
        cur_task = cur_task.compile()
        heuristic = pds.strips.StripsHFFHeuristic(cur_task, translator)

        # plan = pds.strips.strips_heuristic_search(cur_task, heuristic, verbose=True)
        # reward = - len(plan)
        reward = - heuristic.compute(cur_task.state)
        if reward:
            # increase bring me
            if op_type == 'bring_me':
                reward += 1
            reward_dict[op_type][(obj_name, arg)] = reward
        else:
            goal_func = strips_goal.compile()
            if goal_func(op.apply(cur_extended_state)):
                reward_dict[op_type][(obj_name, arg)] = 0
            else:
                reward_dict[op_type][(obj_name, arg)] = - INF
    for op_type in reward_dict.keys():
        for key in reward_dict[op_type].keys():
            if reward_dict[op_type][key] > - hff:
                useful_objects[op_type].append(key)  # useful objects store both obj and arg
    return robot_operators_dict, reward_dict, useful_objects


def prob_m_given_g(meaning_list, reward_dict, object_dict, useful_objects, remaining_steps, positions, beta=BETA_query1, c=C1):
    total_boltzmann_sum = 0.0
    locations = positions.copy()
    prob_list = {'bring_me': [], 'move_to': {}, 'change_state': {}}
    for loc in locations:
        prob_list['move_to'][loc] = []
    for verb in ability_verb_mapping.values():
        prob_list['change_state'][verb] = []

    for obj_description in meaning_list['bring_me']:
        if not is_useful_meaning((obj_description, None), useful_objects, object_dict, 'bring_me'):
            prob_list['bring_me'].append(0.0)
            continue
        tmp_list = list()
        for name in object_dict.keys():
            if obj_in_dict(object_dict[name], obj_description):
                if (name, None) in reward_dict['bring_me'].keys():
                    tmp_list.append(reward_dict['bring_me'][(name, None)])
                else:
                    tmp_list.append(-remaining_steps)
        if not tmp_list:
            prob_list['bring_me'].append(0.0)
            continue
        else:
            avg_reward = sum(tmp_list) / len(tmp_list)
        prob = math.exp(beta * (avg_reward - c * meaning_cost('bring_me', obj_description)))
        prob_list['bring_me'].append(prob)
        total_boltzmann_sum += prob

    for loc in meaning_list['move_to'].keys():
        for obj_description in meaning_list['move_to'][loc]:
            if not is_useful_meaning((obj_description, loc), useful_objects, object_dict, 'move_to'):
                prob_list['move_to'][loc].append(0.0)
                continue
            tmp_list = list()
            for name in object_dict.keys():
                if obj_in_dict(object_dict[name], obj_description):
                    if (name, loc) in reward_dict['move_to'].keys():
                        tmp_list.append(reward_dict['move_to'][(name, loc)])
                    else:
                        tmp_list.append(-remaining_steps)
            if not tmp_list:
                prob_list['move_to'][loc].append(0.0)
                continue
            else:
                avg_reward = sum(tmp_list) / len(tmp_list)
            prob = math.exp(beta * (avg_reward - c * meaning_cost('move_to', obj_description)))
            prob_list['move_to'][loc].append(prob)
            total_boltzmann_sum += prob

    for verb in meaning_list['change_state'].keys():
        for obj_description in meaning_list['change_state'][verb]:
            if not is_useful_meaning((obj_description, verb), useful_objects, object_dict, 'change_state'):
                prob_list['change_state'][verb].append(0.0)
                continue
            tmp_list = list()
            for name in object_dict.keys():
                if obj_in_dict(object_dict[name], obj_description):
                    if (name, verb) in reward_dict['change_state'].keys():
                        tmp_list.append(reward_dict['change_state'][(name, verb)])
                    else:
                        tmp_list.append(-remaining_steps)
            if not tmp_list:
                prob_list['change_state'][verb].append(0.0)
                continue
            else:
                avg_reward = sum(tmp_list) / len(tmp_list)
            prob = math.exp(beta * (avg_reward - c * meaning_cost('change_state', obj_description)))
            prob_list['change_state'][verb].append(prob)
            total_boltzmann_sum += prob

    if total_boltzmann_sum != 0.0:
        prob_list['bring_me'] = np.array(prob_list['bring_me'])
        prob_list['bring_me'] /= total_boltzmann_sum
        for loc in prob_list['move_to']:
            prob_list['move_to'][loc] = np.array(prob_list['move_to'][loc])
            prob_list['move_to'][loc] /= total_boltzmann_sum
        for verb in prob_list['change_state']:
            prob_list['change_state'][verb] = np.array(prob_list['change_state'][verb])
            prob_list['change_state'][verb] /= total_boltzmann_sum
    return prob_list


def is_useful_meaning(meaning, useful_objects, object_dict, meaning_type):
    obj_description = meaning[0]
    argument = meaning[1]
    if meaning_type == 'bring_me':
        useful = True
        useful_objs = [k[0] for k in useful_objects['bring_me']]
        for obj_name in object_dict.keys():
            if obj_in_dict(object_dict[obj_name], obj_description):
                if obj_name not in useful_objs:
                    useful = False
                    break
    else:
        useful = True
        useful_objs = [k[0] for k in useful_objects[meaning_type] if k[1] == argument]
        for obj_name in object_dict.keys():
            if obj_in_dict(object_dict[obj_name], obj_description):
                if obj_name not in useful_objs:
                    useful = False
                    break
    return useful


def reformat_meanings(meaning_list, prob_list):
    meanings = list()
    probs = list()
    for idx, obj_description in enumerate(meaning_list['bring_me']):
        meanings.append(('bring_me', obj_description, None))
        probs.append(prob_list['bring_me'][idx])
    for loc in meaning_list['move_to'].keys():
        for idx, obj_description in enumerate(meaning_list['move_to'][loc]):
            meanings.append(('move_to', obj_description, loc))
            probs.append(prob_list['move_to'][loc][idx])
    for verb in meaning_list['change_state']:
        for idx, obj_description in enumerate(meaning_list['change_state'][verb]):
            meanings.append(('change_state', obj_description, verb))
            probs.append(prob_list['change_state'][verb][idx])
    p_array = np.array(probs)
    p_array /= sum(p_array)
    for idx, p in enumerate(p_array):
        if p < 1e-3:
            p_array[idx] = 0
    p_array /= sum(p_array)
    return meanings, p_array
