import json
import numpy as np


def get_obj_name_and_article(object_dict):
    obj_name_and_article = dict()
    abbr_obj_names = dict()
    for obj_name in object_dict.keys():
        only_one = False
        for another_name in object_dict.keys():
            if obj_name.split('#')[0] == another_name.split('#')[0]:
                if not only_one:
                    only_one = True     # the first time
                else:
                    only_one = False
                    break
        if only_one:
            name = obj_name.split('#')[0]
            abbr_obj_names[name] = obj_name
        else:
            name = obj_name.split('#')[0] + ' ({})'.format(obj_name)
        if object_dict[obj_name]['class'] == 'LOCATION':
            if 'inside' in object_dict[obj_name].keys():
                prep = 'in'
            else:
                prep = 'on'
            obj_name_and_article[obj_name] = {'name': name,
                                              'article': 'a',
                                              'prep': prep}
        elif object_dict[obj_name]['class'] == 'RECEPTACLE':
            if 'inside' in object_dict[obj_name].keys():
                prep = 'in'
            else:
                prep = 'on'
            static_attr = ''
            if 'size' in object_dict[obj_name]['states'].keys():
                static_attr += object_dict[obj_name]['states']['size'] + ' '
            if 'color' in object_dict[obj_name]['states'].keys():
                static_attr += object_dict[obj_name]['states']['color'] + ' '
            obj_name_and_article[obj_name] = {'name': static_attr + name,
                                              'article': 'a',
                                              'prep': prep}
        else:
            obj_name_and_article[obj_name] = {'name': name,
                                              'article': 'a'}
        if obj_name.split('#')[0] in ['oven', 'apparel', 'earphone']:
            obj_name_and_article[obj_name]['article'] = 'an'
        obj_name_and_article['h'] = {'name': 'human', 'article': 'the', 'prep': 'at'}
    return obj_name_and_article, abbr_obj_names


def print_obj_at_loc(object_dict, loc, obj_name_and_article, fully=False):
    # if loc == 'h':
    #     obs = '\nYou can see the human, who is currently holding {}.\n'.format(
    #         interaction.get_human_holding())
    # obs = "\nYou examine the {} closely{}.\n".format(loc, generate_states_expr(object_dict[loc]))
    obs = "\nYou see{} {}.\n".format(generate_states_expr(object_dict[loc]), loc)
    if 'open' in object_dict[loc]['states'].keys():
        if not object_dict[loc]['states']['open'] and not fully:
            return obs
    rec_info = ''
    if 'has-inside' in object_dict[loc]['ability']:
        obs += 'In {} you can see '.format(loc)
        for obj in object_dict.keys():
            if 'inside' in object_dict[obj].keys():
                if loc == object_dict[obj]['inside']:
                    obs += print_obj(object_dict, obj, obj_name_and_article)
                    rec_info += print_obj_at_rec(object_dict, obj, obj_name_and_article, fully=fully)
    else:
        obs += 'On {} you can see '.format(loc)
        for obj in object_dict.keys():
            if 'ontop' in object_dict[obj].keys():
                if loc == object_dict[obj]['ontop']:
                    obs += print_obj(object_dict, obj, obj_name_and_article)
                    rec_info += print_obj_at_rec(object_dict, obj, obj_name_and_article, fully=fully)
    if obs[-4:] == 'see ':
        obs += 'nothing'
    else:
        obs = obs[:-2]
    obs += '. ' + rec_info
    '''
    for obj in object_dict.keys():
        if 'inside' in object_dict[obj].keys():
            if loc == object_dict[obj]['inside']:
                obs += print_obj(object_dict, obj, obj_name_and_article)
                obs += print_obj_at_rec(object_dict, obj, obj_name_and_article, fully=fully)
        elif 'ontop' in object_dict[obj].keys():
            if loc == object_dict[obj]['ontop']:
                obs += print_obj(object_dict, obj, obj_name_and_article)
                obs += print_obj_at_rec(object_dict, obj, obj_name_and_article, fully=fully)
    '''
    return obs


def print_obj(object_dict, obj, obj_name_and_article, indent=False):
    expr = ''
    states_expr = generate_states_expr(object_dict[obj])
    if 'inside' in object_dict[obj].keys():
        expr += "{} {}, ".format(states_expr, obj)
        # expr += "There is {} inside the {}{}.".format(name_wrap(obj, obj_name_and_article, True),
        #                                     name_wrap(object_dict[obj]['inside'], obj_name_and_article), states_expr)
    elif 'ontop' in object_dict[obj].keys():
        expr += "{} {}, ".format(states_expr, obj)
        # expr += "There is {} on top of the {}{}.".format(name_wrap(obj, obj_name_and_article, True),
        #                                     name_wrap(object_dict[obj]['ontop'], obj_name_and_article), states_expr)
    else:
        expr += '{} {}'.format(states_expr, obj)
        # expr += '{}{}'.format(name_wrap(obj, obj_name_and_article, True), states_expr)
        return expr
    obs = expr
    # if indent:
    #     obs = '\t' + obs
    return obs


def print_obj_at_rec(object_dict, rec, obj_name_and_article, fully=False):
    obs = ''
    if object_dict[rec]['class'] != 'RECEPTACLE':
        return ''
    if 'open' in object_dict[rec]['states'].keys():
        if not object_dict[rec]['states']['open'] and not fully:
            return obs
    if 'has-inside' in object_dict[rec]['ability']:
        obs += 'In {} you can see '.format(rec)
        for obj in object_dict.keys():
            if 'inside' in object_dict[obj].keys():
                if rec == object_dict[obj]['inside']:
                    obs += print_obj(object_dict, obj, obj_name_and_article)
    else:
        obs += 'On {} you can see '.format(rec)
        for obj in object_dict.keys():
            if 'ontop' in object_dict[obj].keys():
                if rec == object_dict[obj]['ontop']:
                    obs += print_obj(object_dict, obj, obj_name_and_article)
    if obs[-4:] == 'see ':
        return ''
    else:
        obs = obs[:-2]
    obs += '. '
    '''
    for obj in object_dict:
        if 'inside' in object_dict[obj].keys():
            if rec == object_dict[obj]['inside']:
                obs += print_obj(object_dict, obj, obj_name_and_article, True)
        elif 'ontop' in object_dict[obj].keys():
            if rec == object_dict[obj]['ontop']:
                obs += print_obj(object_dict, obj, obj_name_and_article, True)
    '''
    return obs


def generate_states_expr(dictionary):
    states = list()
    for key in dictionary['states']:
        if key in ['size', 'color']:
            states.append(dictionary['states'][key])
    for key in dictionary['states']:
        if key not in ['size', 'color']:
            if dictionary['states'][key]:
                if key == 'toggled':
                    states.append('toggled-on')
                else:
                    states.append(key)
            else:
                if key == 'open':
                    states.append('closed')
                elif key == 'toggled':
                    states.append('toggled-off')

    if states:
        expr = ''
        # expr = np.random.choice([', which is ', ', that is ', ' and it is '])
        for state in states:
            expr += ' ' + state
    else:
        expr = ''
    return expr


def print_action(action, obj_name_and_article):
    name = action['name']
    raw_args = action['arguments']
    # args = name_wrap(raw_args, obj_name_and_article)
    args = raw_args
    expr = ''
    if name == 'human-move':
        expr = 'Human moves to the {}.'.format(args[2])
    elif name == 'human-pick-up-at-location':
        expr = 'Human picks up the {} at the {}.'.format(args[1], args[2])
        # expr = expr.replace(' at ', ' ' + obj_name_and_article[raw_args[2]]['prep'] + ' ')
    elif name == 'human-pick-up-from-receptacle-at-location':
        expr = 'Human picks up the {} from the {} at the {}.'.format(args[1], args[2], args[3])
    elif name == 'human-put-inside-location':
        expr = 'Human puts the {} into the {}.'.format(args[1], args[2])
    elif name == 'human-put-ontop-location':
        expr = 'Human puts the {} on the {}.'.format(args[1], args[2])
    elif name == 'human-put-inside-receptacle-at-location':
        expr = 'Human puts the {} into the {}.'.format(args[1], args[2])
    elif name == 'human-put-ontop-receptacle-at-location':
        expr = 'Human puts the {} on the {}.'.format(args[1], args[2])
    elif name == 'human-open-location':
        expr = 'Human opens the {}.'.format(args[1])
    elif name == 'human-close-location':
        expr = 'Human closes the {}.'.format(args[1])
    elif name == 'human-open-receptacle-at-location':
        expr = 'Human opens the {} at the {}.'.format(args[1], args[2])
    elif name == 'human-close-receptacle-at-location':
        expr = 'Human closes the {} at the {}.'.format(args[1], args[2])
    elif name == 'human-toggle-on-location':
        expr = 'Human toggles the {} on.'.format(args[1])
    elif name == 'human-toggle-off-location':
        expr = 'Human toggles the {} off.'.format(args[1])
    elif name == 'human-toggle-on-movable-at-location':
        expr = 'Human toggles the {} on at the {}.'.format(args[1], args[2])
    elif name == 'human-toggle-off-movable-at-location':
        expr = 'Human toggles the {} off at the {}.'.format(args[1], args[2])
    elif name == 'human-heat-obj':
        expr = 'Human heats the {} up with the {}.'.format(args[1], args[2])
    elif name == 'human-cool-obj':
        expr = 'Human cools the {} down in the {}.'.format(args[1], args[2])
    elif name == 'human-slice-obj':
        expr = 'Human slices up the {} with the {} at the {}.'.format(args[1], args[2], args[3])
    elif name == 'human-soak-obj':
        expr = 'Human makes the {} soaked in the {}.'.format(args[1], args[2])
    elif name == 'human-clean-obj-at-location':
        expr = 'Human cleans up the {} with the {} at the {}.'.format(args[1], args[2], args[3])
    elif name == 'human-clean-location':
        expr = 'Human cleans up the {} with the {}.'.format(args[1], args[2])
    return expr


def print_utterance(utterance, obj_name_and_article):
    if utterance[0] == 'bring_me':
        expr = np.random.choice(['', 'please ', 'can you ']) + np.random.choice(['hand me ', 'bring me ', 'give me '])
        if len(utterance[1].keys()) == 0:
            expr += 'that'
        else:
            expr += get_object_description(utterance[1], obj_name_and_article)
        expr += '.'
    elif utterance[0] == 'move_to':
        if utterance[2] is None:
            expr = np.random.choice(['', 'please ', 'can you ']) + 'put '
            if len(utterance[1].keys()) == 0:
                expr += 'it'
            else:
                expr += get_object_description(utterance[1], obj_name_and_article)
            expr += ' over there.'
        else:
            expr = np.random.choice(['', 'please ', 'can you ']) + np.random.choice(['move ', 'put '])
            if len(utterance[1].keys()) == 0:
                expr += 'it'
            else:
                expr += get_object_description(utterance[1], obj_name_and_article)
            expr += ' to the {}.'.format(name_wrap(utterance[2], obj_name_and_article))
    else:
        expr = np.random.choice(['', 'please ', 'can you '])
        prep_mapping = {'toggle': ' on', 'clean': ' up', 'cool': ' down', 'heat': ' up', 'slice': ' up', 'soak': '', 'open': ''}
        if len(utterance[1].keys()) == 0:
            object_des_expr = 'it'
        else:
            object_des_expr = get_object_description(utterance[1], obj_name_and_article)
        if len(object_des_expr) >= 20:
            expr += utterance[2] + prep_mapping[utterance[2]] + ' ' + object_des_expr + '.'
        else:
            expr += utterance[2] + ' ' + object_des_expr + prep_mapping[utterance[2]] + '.'

    return expr.capitalize()


def get_object_description(obj_description, obj_name_and_article):
    expr = 'the '
    states_expr = ''
    for key in obj_description.keys():
        if key in ['inside', 'ontop', 'class', 'subclass', 'type']:
            continue
        if key in ['color', 'size']:
            states_expr += obj_description[key] + ', '
        elif obj_description[key]:
            if key == 'toggled':
                states_expr += 'toggled-off, '
            else:
                states_expr += key + ', '
        elif not obj_description[key]:
            if key in ['cooked', 'frozen', 'sliced']:
                states_expr += 'un' + key + ', '
            elif key == 'toggled':
                states_expr += 'toggled-on, '
            elif key == 'open':
                states_expr += 'closed, '
            elif key in ['dusty', 'stained']:
                states_expr += 'clean, '
            else:
                states_expr += 'not ' + key + ', '      # not soaked
    if states_expr:
        states_expr = states_expr[:-2] + ' '
        expr += states_expr
    if 'type' in obj_description.keys():
        expr += name_wrap(obj_description['type'], obj_name_and_article)
    elif 'subclass' in obj_description.keys():
        expr += name_wrap(obj_description['subclass'], obj_name_and_article)
    elif 'class' in obj_description.keys():
        expr += name_wrap(obj_description['class'], obj_name_and_article)
    else:
        expr += 'one'
    if 'inside' in obj_description.keys():
        expr += ' in the {}'.format(name_wrap(obj_description['inside'], obj_name_and_article))
    elif 'ontop' in obj_description.keys():
        expr += ' on the {}'.format(name_wrap(obj_description['ontop'], obj_name_and_article))
    return expr


def name_wrap(raw_args, obj_name_and_article, with_article=False, label=False):
    if isinstance(raw_args, list):
        args = raw_args.copy()
        for idx, arg in enumerate(args):
            if arg in obj_name_and_article.keys():
                args[idx] = obj_name_and_article[arg]['name']
                if with_article:
                    args[idx] = obj_name_and_article[raw_args[idx]]['article'] + ' ' + args[idx]
    else:
        args = raw_args
        if args in obj_name_and_article.keys():
            args = obj_name_and_article[args]['name']
            if with_article:
                args = obj_name_and_article[raw_args]['article'] + ' ' + args
            if label:
                if len(args.split()) > 1:
                    args = args.split()[-1]
                    if args[0] == '(' and args[-1] == ')':
                        args = args[1:-1]
    return args


MAPPING = {'slice': 'sliced', 'open': 'open', 'toggle': 'toggled', 'heat': 'cooked', 'cool': 'frozen', 'soak': 'soaked'}


def get_meaning_goal(meaning, valid_objects, state, translator):
    expr = '(or '
    objects = [valid[0] for valid in valid_objects]
    for obj in objects:
        if meaning[0] == 'bring_me':
            expr += '(human-holding {}) '.format(obj)
        elif meaning[0] == 'move_to':
            expr += '(inside {} {}) (ontop {} {}) '.format(obj, meaning[2], obj, meaning[2])
        else:
            if meaning[2] == 'clean':
                expr += '(and (not (stained {})) (not (dusty {})))'.format(obj, obj)
            else:
                expr += '({} {}) '.format(MAPPING[meaning[2]], obj)
    expr += ')'
    strips_goal = translator.compile_expr(expr, state)[0]
    return strips_goal


def get_all_locations(object_dict):
    locations = ['h']
    for obj in object_dict.keys():
        if object_dict[obj]['class'] == 'LOCATION':
            locations.append(obj)
    return locations
