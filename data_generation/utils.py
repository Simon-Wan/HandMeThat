import itertools
from itertools import chain, combinations
from typing import Optional, Union, Callable, Tuple, Sequence, List, Mapping, Any, Dict
from pdsketch.value import ObjectType, NamedValueType, NamedValueTypeSlot
from pdsketch.state import State
from pdsketch.domain import Domain
from pdsketch.operator import OperatorApplier
from pdsketch.session import Session
from pdsketch.strips.grounding import filter_static_grounding


def generate_predicates(ctx, object_dict):
    predicates = list()

    for name in object_dict.keys():
        # movable and receptacle
        if object_dict[name]['class'] != 'LOCATION':
            predicates.append(ctx.get_pred('movable')(name))
            if object_dict[name]['class'] == 'RECEPTACLE':
                predicates.append(ctx.get_pred('receptacle')(name))
        else:
            predicates.append(ctx.get_pred('receptacle')(name))
        # type
        predicates.append(ctx.get_pred('type-' + object_dict[name]['type'])(name))
        # ability
        for ability in object_dict[name]['ability']:
            if 'cleanable' in ability:
                tool = ability.split('-')[0]
                for tool_name in object_dict.keys():
                    if tool in tool_name:
                        predicates.append(ctx.valid_clean_pair(name, tool_name))
            else:
                predicates.append(ctx.get_pred(ability)(name))
        # location
        if 'inside' in object_dict[name].keys():
            predicates.append(ctx.inside(name, object_dict[name]['inside']))
        if 'ontop' in object_dict[name].keys():
            predicates.append(ctx.ontop(name, object_dict[name]['ontop']))
        # states
        for key in object_dict[name]['states'].keys():
            if key == 'color':
                predicates.append(ctx.get_pred('color-' + object_dict[name]['states'][key])(name))
            elif key == 'size':
                predicates.append(ctx.get_pred('size-' + object_dict[name]['states'][key])(name))
            else:
                if object_dict[name]['states'][key]:
                    predicates.append(ctx.get_pred(key)(name))

    return predicates


OTHER_ATTRIBUTES = ['cooked', 'dusty', 'frozen', 'stained', 'sliced', 'soaked', 'toggled', 'open']
OTHER_ABILITIES = ['cookable', 'dustyable', 'freezable', 'stainable', 'sliceable', 'soakable', 'toggleable', 'openable']
ACTION_ARGS = {
    'human-move': ['human', 'location', 'location'],
    'human-pick-up-at-location': ['human', 'movable', 'location'],
    'human-pick-up-from-receptacle-at-location': ['human', 'movable', 'receptacle', 'location'],
    'human-put-inside-location': ['human', 'movable', 'location'],
    'human-put-ontop-location': ['human', 'movable', 'location'],
    'human-put-inside-receptacle-at-location': ['human', 'movable', 'receptacle', 'location'],
    'human-put-ontop-receptacle-at-location': ['human', 'movable', 'receptacle', 'location'],
    'human-open-location': ['human', 'location'],
    'human-close-location': ['human', 'location'],
    'human-open-receptacle-at-location': ['human', 'receptacle', 'location'],
    'human-close-receptacle-at-location': ['human', 'receptacle', 'location'],
    'human-toggle-on-location': ['human', 'location'],
    'human-toggle-off-location': ['human', 'location'],
    'human-toggle-on-movable-at-location': ['human', 'tool', 'location'],
    'human-toggle-off-movable-at-location': ['human', 'tool', 'location'],
    'human-heat-obj': ['human', 'food', 'location'],
    'human-cool-obj': ['human', 'food', 'location'],
    'human-slice-obj': ['human', 'food', 'tool', 'location'],
    'human-soak-obj': ['human', 'tool', 'location'],
    'human-clean-obj-at-location': ['human', 'movable_cleanable', 'tool', 'location'],
    'human-clean-location': ['human', 'tool', 'location'],
    'robot-move-obj-to-human': ['human', 'movable', 'position'],
    'robot-move-obj-from-rec-into-rec': ['human', 'movable', 'position', 'position'],
    'robot-move-obj-from-rec-onto-rec': ['human', 'movable', 'position', 'position'],
    'robot-toggle-on': ['human', 'all'],
    'robot-toggle-off': ['human', 'all'],
    'robot-heat-obj': ['human', 'food'],
    'robot-cool-obj': ['human', 'food'],
    'robot-slice-obj': ['human', 'food'],
    'robot-soak-obj': ['human', 'other'],
    'robot-clean-obj': ['human', 'all'],
}


def generate_relevant_partially_grounded_actions(
    session: Session,
    state: State,
    rel_types: Dict,
    action_names: Optional[Sequence[str]] = None,
    action_filter: Optional[Callable[[OperatorApplier], bool]] = None,
    filter_static: Optional[bool] = True
) -> List[OperatorApplier]:

    assert isinstance(session, Session)

    if action_names is not None:
        action_ops = [session.domain.operators[x] for x in action_names]
    else:
        action_ops = session.domain.operators.values()

    rel = dict()
    rel['human'] = ['h']
    rel['location'] = rel_types['LOCATION']
    rel['receptacle'] = rel_types['RECEPTACLE']
    rel['food'] = rel_types['FOOD']
    rel['tool'] = rel_types['TOOL']
    rel['thing'] = rel_types['THING']
    rel['position'] = rel['location'] + rel['receptacle']
    rel['other'] = rel_types['FOOD'] + rel_types['TOOL'] + rel_types['THING']
    rel['movable'] = rel['receptacle'] + rel['other']
    rel['all'] = rel['movable'] + rel['location']
    rel['movable_cleanable'] = rel['receptacle'] + rel['thing']

    actions = list()
    for op in action_ops:
        argument_candidates = list()
        for idx, arg in enumerate(op.arguments):
            if isinstance(arg.dtype, ObjectType):
                candidates = state.object_type2names[arg.dtype.typename]
                type_class = ACTION_ARGS[op.name][idx]
                relevant_candidates = rel_analysis(candidates, rel[type_class])
                argument_candidates.append(relevant_candidates)
            else:
                assert isinstance(arg.dtype, NamedValueType)
                argument_candidates.append([NamedValueTypeSlot(arg.dtype)])
        for comb in itertools.product(*argument_candidates):
            actions.append(op(*comb))

    if filter_static:
        actions = filter_static_grounding(session, state, actions)
    if action_filter is not None:
        actions = list(filter(action_filter, actions))
    return actions


def rel_analysis(candidates, relevant):
    result = list()
    for rel in relevant:
        if '#' not in rel:
            for cand in candidates:
                if cand[:len(rel)] == rel:
                    result.append(cand)
        else:
            result.append(rel)
    return result


def update_object_dict(object_dict, cur_extended_state, names):
    state = list(cur_extended_state)
    for predicate in state:
        if '_not' in predicate:
            continue
        if 'type' in predicate:
            continue
        if 'color' in predicate:
            continue
        if 'size' in predicate:
            continue
        args = predicate.split()
        if args[0] == 'inside':
            object_dict[names[int(args[1]) + 1]]['inside'] = names[int(args[2]) + 1]
            if 'ontop' in object_dict[names[int(args[1]) + 1]].keys():
                object_dict[names[int(args[1]) + 1]].pop('ontop')
            continue
        if args[0] == 'ontop':
            object_dict[names[int(args[1]) + 1]]['ontop'] = names[int(args[2]) + 1]
            if 'inside' in object_dict[names[int(args[1]) + 1]].keys():
                object_dict[names[int(args[1]) + 1]].pop('inside')
            continue
        for attr in OTHER_ATTRIBUTES:
            if args[0] == attr:
                object_dict[names[int(args[1]) + 1]]['states'][attr] = True
    return object_dict


def power_set(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def obj_in_dict(obj, any_dict):
    answer = True
    for key in any_dict:
        if key in ['type', 'subclass', 'class']:
            if any_dict[key] != obj[key]:
                answer = False
                break
        elif key in ['inside', 'ontop']:
            if key not in obj.keys():
                answer = False
                break
            elif obj[key] != any_dict[key]:
                answer = False
                break
        elif key not in obj['states'].keys():
            answer = False
            break
        elif obj['states'][key] != any_dict[key]:
            answer = False
            break
    return answer


def dict_dominate_dict(meaning, utterance, hierarchy):
    if meaning[0] != utterance[0]:
        return False
    if meaning[2] != utterance[2] and utterance[2] is not None:
        return False
    answer = True
    dict1 = meaning[1]
    dict2 = utterance[1]
    for key in dict2:
        if key in ['class', 'subclass', 'type']:
            if key == 'type':
                if key not in dict1.keys():
                    answer = False
                    break
                elif dict1[key] != dict2[key]:
                    answer = False
                    break
            elif key == 'subclass':
                if key not in dict1.keys():
                    tmp_list = list()
                    for cls in hierarchy.keys():
                        if dict2[key] in hierarchy[cls].keys():
                            tmp_list = hierarchy[cls][dict2[key]]
                    if 'type' not in dict1.keys():
                        answer = False
                        break
                    elif dict1['type'] not in tmp_list:
                        answer = False
                        break
                elif dict1[key] != dict2[key]:
                    answer = False
                    break
            elif key == 'class':
                if key not in dict1.keys():
                    if 'subclass' not in dict1.keys():
                        if 'type' not in dict1.keys():
                            answer = False
                            break
                        else:
                            found = False
                            for subclass in hierarchy[dict2[key]].keys():
                                if dict1['type'] in hierarchy[dict2[key]][subclass]:
                                    found = True
                                    break
                            if not found:
                                answer = False
                                break
                    elif dict1['subclass'] not in hierarchy[dict2[key]].keys():
                        answer = False
                        break
                elif dict1[key] != dict2[key]:
                    answer = False
                    break
            continue
        if key not in dict1.keys():
            answer = False
            break
        elif dict1[key] != dict2[key]:
            answer = False
            break
    return answer


def get_objects_from_description(object_dict, quest, positions=None):
    name_list = list()
    for name in object_dict.keys():
        if object_dict[name]['class'] == 'LOCATION':
            continue
        if obj_in_dict(object_dict[name], quest[1]):
            name_list.append(name)
    if quest[0] == 'bring_me':
        return [(name, None) for name in name_list]
    elif quest[0] == 'move_to':
        if quest[2] is None:
            result = []
            for pos in positions:
                result += [(name, pos) for name in name_list]
            return result
        return [(name, quest[2]) for name in name_list]
    else:
        return [(name, quest[2]) for name in name_list]


def get_hardness_level(ag, am, au, ar):
    if set(am) == set(au):
        return 1
    if set(am) == set(ag).intersection(set(au)):
        return 2
    if set(am) == set(ag).intersection(set(ar)):
        return 3
    return 4


