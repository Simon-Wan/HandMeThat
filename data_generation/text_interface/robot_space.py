import json
import pdsketch as pds

ACTION_ARGS = {
    'robot-move': ['robot', 'location', 'location'],
    'robot-pick-up-at-location': ['robot', 'movable', 'location'],
    'robot-pick-up-from-receptacle-at-location': ['robot', 'movable', 'receptacle', 'location'],
    'robot-put-inside-location': ['robot', 'movable', 'location'],
    'robot-put-ontop-location': ['robot', 'movable', 'location'],
    'robot-put-inside-receptacle-at-location': ['robot', 'movable', 'receptacle', 'location'],
    'robot-put-ontop-receptacle-at-location': ['robot', 'movable', 'receptacle', 'location'],
    'robot-bring-to-human': ['robot', 'movable', 'human'],
    'robot-take-from-human': ['robot', 'movable', 'human'],
    'robot-open-location': ['robot', 'location'],
    'robot-close-location': ['robot', 'location'],
    'robot-open-receptacle-at-location': ['robot', 'receptacle', 'location'],
    'robot-close-receptacle-at-location': ['robot', 'receptacle', 'location'],
    'robot-toggle-on-location': ['robot', 'location'],
    'robot-toggle-off-location': ['robot', 'location'],
    'robot-toggle-on-movable-at-location': ['robot', 'tool', 'location'],
    'robot-toggle-off-movable-at-location': ['robot', 'tool', 'location'],
    'robot-heat-obj': ['robot', 'food', 'location'],
    'robot-cool-obj': ['robot', 'food', 'location'],
    'robot-slice-obj': ['robot', 'food', 'tool', 'location'],
    'robot-soak-obj': ['robot', 'other', 'location'],
    'robot-clean-obj-at-location': ['robot', 'other', 'tool', 'location'],
    'robot-clean-location': ['robot', 'tool', 'location'],
}


def set_up_planning_env(object_dict, session):
    names, state, translator, rel_types = define_robot_space(object_dict, session)
    action_ops = session.domain.operators.values()
    strips_state = translator.compile_state(state)
    return names, state, translator, rel_types, action_ops, strips_state


def define_robot_space(object_dict, session):
    names = list(object_dict.keys())
    num_of_obj = len(names)
    names = ['r', 'h'] + names
    types = ['robot', 'phyobj'] + ['phyobj' for _ in range(num_of_obj)]
    state = pds.State([session.domain.types[t] for t in types], object_names=names)
    ctx = state.define_context(session.domain)
    predicates = robot_generate_predicates(ctx, object_dict)

    predicates.append(ctx.hand_empty('r'))
    predicates.append(ctx.robot_at('r', 'floor'))
    predicates.append(ctx.type_human('h'))

    ctx.define_predicates(predicates)
    translator = pds.strips.GStripsTranslator(session, use_string_name=True)
    rel_types = {'LOCATION': [], 'RECEPTACLE': [], 'FOOD': [], 'TOOL': [], 'THING': []}
    for obj in object_dict.keys():
        rel_types[object_dict[obj]['class']].append(obj)
    return names, state, translator, rel_types


def robot_generate_predicates(ctx, object_dict):
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
                predicates.append(ctx.valid_clean_pair(name, tool + '#0'))
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
