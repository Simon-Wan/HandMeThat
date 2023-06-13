import numpy as np


def get_each_hierarchy_level(object_names):
    subclasses = dict()     # 'subclasses' is a dictionary for all subclasses in all branches
    types = dict()          # 'types' is a dictionary for the number of sampled objects in each particular type
    for cls in object_names.keys():
        if cls == 'LOCATION':
            for subclass in object_names[cls]['inside'] + object_names[cls]['ontop']:
                subclasses[subclass] = [subclass]
                types[subclass] = 1
            continue
        for subclass in object_names[cls].keys():
            subclasses[subclass] = list()
            for obj_type in object_names[cls][subclass].keys():
                num = len(object_names[cls][subclass][obj_type])
                types[obj_type] = num
                if num:
                    subclasses[subclass].append(obj_type)
    return subclasses, types


def update_rel_types(rel_types, object_dict, obj_type):
    if obj_type + '#0' in object_dict.keys():
        subclass = object_dict[obj_type + '#0']['subclass']
    else:
        subclass = None
    for key in object_dict.keys():
        if key.split('#')[0] == obj_type or key == obj_type or object_dict[key]['subclass'] == subclass:
            # object_type can be a particular obj in some case
            rel_types[object_dict[key]['class']].append(key)        # type
            for prep in ['inside', 'ontop']:
                if prep in object_dict[key].keys():
                    rec = object_dict[key][prep]
                    rel_types[object_dict[rec]['class']].append(rec)  # particular obj
                    if 'inside' in object_dict[rec].keys():
                        rec_rec = object_dict[rec]['inside']
                        rel_types[object_dict[rec_rec]['class']].append(rec_rec)  # particular obj
                    elif 'ontop' in object_dict[rec].keys():
                        rec_rec = object_dict[rec]['ontop']
                        rel_types[object_dict[rec_rec]['class']].append(rec_rec)  # particular obj


def update_rel_targets(rel_targets, x, y, object_dict):
    if x + '#0' in object_dict.keys():
        subclass = object_dict[x + '#0']['subclass']
    else:
        subclass = None
    for key in object_dict.keys():
        if key.split('#')[0] == x or key == x:          # object_type can be a particular obj in some case
            if object_dict[key]['class'] == 'LOCATION':
                rel_targets['POSITION'].append(key)
            else:
                rel_targets['MOVABLE'].append(key)
        elif key.split('#')[0] == y or key == y:          # object_type can be a particular obj in some case
            if object_dict[key]['class'] != 'LOCATION':
                rel_targets['MOVABLE'].append(key)
            rel_targets['POSITION'].append(key)
        elif object_dict[key]['subclass'] == subclass:
            rel_targets['MOVABLE'].append(key)


def grounding_objects_in_state(object_dict, state):
    objects = list()
    state = state.split()
    if state[0] == 'inside':
        for key in object_dict.keys():
            if 'inside' in object_dict[key].keys():
                if object_dict[key]['inside'] == state[1]:
                    objects.append(key)
        return objects
    if state[0] == 'ontop':
        for key in object_dict.keys():
            if 'ontop' in object_dict[key].keys():
                if object_dict[key]['ontop'] == state[1]:
                    objects.append(key)
        return objects
    if_state = True
    if len(state) == 2:
        if_state = False
    state = state[-1]
    for key in object_dict.keys():
        if state in object_dict[key]['states'].keys():
            if object_dict[key]['states'][state] == if_state:
                objects.append(key)
        if state.split('-')[0] in object_dict[key]['states'].keys():
            if object_dict[key]['states'][state.split('-')[0]] == state.split('-')[1]:
                objects.append(key)
    return objects


def replace_subclass_by_same_type(goal, subclasses):
    replaced_goal = list()
    tmp_dict = dict()
    for sub_goal in goal:
        words = sub_goal.split()
        for word in words:
            if word[0] == '[':
                name = word[1:-2]
                if name in subclasses.keys() and name not in tmp_dict.keys():
                    tmp_dict[name] = np.random.choice(subclasses[name])
    for sub_goal in goal:
        replaced_sub_goal = sub_goal
        for name in tmp_dict.keys():
            replaced_sub_goal = replaced_sub_goal.replace(name, tmp_dict[name])
        replaced_goal.append(replaced_sub_goal)
    return replaced_goal
