
BETA = 0.5
BETA_query1 = 3.0
BETA_query2 = 2.0
C1 = 0.5
C2 = 10.0


def append_dict(dict_list, key, value):
    new_list = dict_list.copy()
    for old_dict in dict_list:
        new_dict = old_dict.copy()
        new_dict[key] = value
        new_list.append(new_dict)
    return new_list


def find_class(subclass, hierarchy):
    for key in hierarchy.keys():
        if subclass in hierarchy[key]:
            return key
    return None


def find_subclass_and_class(t, hierarchy):
    for cls in hierarchy.keys():
        for subcls in hierarchy[cls].keys():
            if t in hierarchy[cls][subcls]:
                return subcls, cls
    return None, None


def meaning_cost(action_type, obj_description):
    type_cost = 0
    if 'class' in obj_description.keys():
        type_cost = 0
    if 'subclass' in obj_description.keys():
        type_cost = 0
    if 'type' in obj_description.keys():
        type_cost = 0
    total_cost = (len(obj_description.keys()) - 1) * 1 + type_cost
    if action_type == 'move_to':
        total_cost += 2                 # todo: tune the params here
    elif action_type == 'change_state':
        total_cost += 4                 # todo: tune the params here
    return total_cost


def utterance_cost(action_type, obj_description, arg=False):
    type_cost = 0
    if 'class' in obj_description.keys():
        type_cost = 0
    if 'subclass' in obj_description.keys():
        type_cost = 1
    if 'type' in obj_description.keys():
        type_cost = 2
    total_cost = (len(obj_description.keys()) - 1) * 1 + type_cost
    if arg:
        total_cost += 1
    return total_cost

