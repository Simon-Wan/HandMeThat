from data_generation.text_interface.robot_space import ACTION_ARGS
from data_generation.text_interface.text_utils import *


class Interaction:
    def __init__(self, object_dict, obj_name_and_article, state, strips_state, action_ops, translator):
        self.action_names = list(ACTION_ARGS.keys())
        self.object_dict = object_dict
        self.obj_name_and_article = obj_name_and_article
        self.state = state
        self.strips_state = strips_state
        self.action_ops = action_ops
        self.translator = translator
        self.locations = get_all_locations(self.object_dict)

        self.current_location = 'floor'
        self.holding = None
        self.human_holding = list()

    def ground_an_action(self, name, arg):
        for op in self.action_ops:
            if op.name == name:
                try:
                    strips_op = self.translator.compile_operator(op(*tuple(arg)), self.state, is_relaxed=False)
                    strips_op.compile()
                    return strips_op
                except Exception as _:
                    return None
        return None

    def move(self, to_loc):
        from_loc = self.current_location
        if to_loc == 'human':
            to_loc = 'h'
        name = 'robot-move'
        arg = ['r', from_loc, to_loc]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.current_location = to_loc
        return 'You move from the {} to the {}.'.format(self.wrap(from_loc), self.wrap(to_loc))

    def pick_up_at_location(self, movable):
        location = self.current_location
        name = 'robot-pick-up-at-location'
        arg = ['r', movable, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.holding = movable
            if 'inside' in self.object_dict[movable].keys():
                self.object_dict[movable].pop('inside')
            elif 'ontop' in self.object_dict[movable].keys():
                self.object_dict[movable].pop('ontop')

        return 'You pick up the {} from the {}.'.format(self.wrap(movable), self.wrap(location))

    def pick_up_from_receptacle_at_location(self, movable, receptacle):
        location = self.current_location
        name = 'robot-pick-up-from-receptacle-at-location'
        arg = ['r', movable, receptacle, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.holding = movable
            if 'inside' in self.object_dict[movable].keys():
                self.object_dict[movable].pop('inside')
            elif 'ontop' in self.object_dict[movable].keys():
                self.object_dict[movable].pop('ontop')

        return 'You pick up the {} from the {} at the {}.'.format(self.wrap(movable), self.wrap(receptacle), self.wrap(location))

    def put_inside_location(self, movable, location):
        name = 'robot-put-inside-location'
        arg = ['r', movable, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.holding = None
            self.object_dict[movable]['inside'] = location

        return 'You put the {} into the {}.'.format(self.wrap(movable), self.wrap(location))

    def put_ontop_location(self, movable, location):
        name = 'robot-put-ontop-location'
        arg = ['r', movable, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.holding = None
            self.object_dict[movable]['ontop'] = location

        return 'You put the {} on the {}.'.format(self.wrap(movable), self.wrap(location))

    def put_inside_receptacle_at_location(self, movable, receptacle):
        location = self.current_location
        name = 'robot-put-inside-receptacle-at-location'
        arg = ['r', movable, receptacle, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.holding = None
            self.object_dict[movable]['inside'] = receptacle

        return 'You put the {} into the {} at the {}.'.format(self.wrap(movable), self.wrap(receptacle), self.wrap(location))

    def put_ontop_receptacle_at_location(self, movable, receptacle):
        location = self.current_location
        name = 'robot-put-ontop-receptacle-at-location'
        arg = ['r', movable, receptacle, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.holding = None
            self.object_dict[movable]['ontop'] = receptacle

        return 'You put the {} on the {} at the {}.'.format(self.wrap(movable), self.wrap(receptacle), self.wrap(location))

    def give_to_human(self, movable):
        location = self.current_location
        name = 'robot-bring-to-human'
        arg = ['r', movable, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.holding = None
            self.human_holding.append(movable)

        return 'You give the {} to human.'.format(self.wrap(movable))

    def take_from_human(self, movable):
        location = self.current_location
        name = 'robot-take-from-human'
        arg = ['r', movable, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.holding = movable
            self.human_holding.remove(movable)

        return 'You take the {} from human.'.format(self.wrap(movable))

    def open_location(self, location):
        name = 'robot-open-location'
        arg = ['r', location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[location]['states']['open'] = True

        return 'You open the {}.'.format(self.wrap(location))

    def close_location(self, location):
        name = 'robot-close-location'
        arg = ['r', location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[location]['states']['open'] = False

        return 'You close the {}.'.format(self.wrap(location))

    def open_receptacle_at_location(self, receptacle):
        location = self.current_location
        name = 'robot-open-receptacle-at-location'
        arg = ['r', receptacle, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[receptacle]['states']['open'] = True

        return 'You open the {}.'.format(self.wrap(receptacle))

    def close_receptacle_at_location(self, receptacle):
        location = self.current_location
        name = 'robot-close-receptacle-at-location'
        arg = ['r', receptacle, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[receptacle]['states']['open'] = False

        return 'You close the {}.'.format(self.wrap(receptacle))

    def toggle_on_location(self, location):
        name = 'robot-toggle-on-location'
        arg = ['r', location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[location]['states']['toggled'] = True

        return 'You toggle the {} on.'.format(self.wrap(location))

    def toggle_off_location(self, location):
        name = 'robot-toggle-off-location'
        arg = ['r', location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[location]['states']['toggled'] = False

        return 'You toggle the {} off.'.format(self.wrap(location))

    def toggle_on_movable_at_location(self, movable):
        location = self.current_location
        name = 'robot-toggle-on-movable-at-location'
        arg = ['r', movable, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[movable]['states']['toggled'] = True

        return 'You toggle the {} on.'.format(self.wrap(movable))

    def toggle_off_movable_at_location(self, movable):
        location = self.current_location
        name = 'robot-toggle-off-movable-at-location'
        arg = ['r', movable, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[movable]['states']['toggled'] = False

        return 'You toggle the {} off.'.format(self.wrap(movable))

    def heat_obj(self, food):
        location = self.current_location
        name = 'robot-heat-obj'
        arg = ['r', food, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[food]['states']['cooked'] = True
            self.object_dict[food]['states']['frozen'] = False

        return 'You heat the {} up with the {}.'.format(self.wrap(food), self.wrap(location))

    def cool_obj(self, food):
        location = self.current_location
        name = 'robot-cool-obj'
        arg = ['r', food, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[food]['states']['cooked'] = False
            self.object_dict[food]['states']['frozen'] = True

        return 'You cool the {} down with the {}.'.format(self.wrap(food), self.wrap(location))

    def slice_obj(self, food, tool):
        location = self.current_location
        name = 'robot-slice-obj'
        arg = ['r', food, tool, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[food]['states']['sliced'] = True

        return 'You slice up the {} with the {}.'.format(self.wrap(food), self.wrap(tool))

    def soak_obj(self, movable):
        location = self.current_location
        name = 'robot-soak-obj'
        arg = ['r', movable, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[movable]['states']['soaked'] = True

        return 'You make the {} soaked in the {}.'.format(self.wrap(movable), self.wrap(location))

    def clean_obj_at_location(self, movable, tool):
        location = self.current_location
        name = 'robot-clean-obj-at-location'
        arg = ['r', movable, tool, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[movable]['states']['dusty'] = False

        return 'You clean up the {} with the {}.'.format(self.wrap(movable), self.wrap(tool))

    def clean_location(self, tool, location):
        name = 'robot-clean-location'
        arg = ['r', tool, location]
        action = self.ground_an_action(name, arg)
        if not action:
            return 'I can\'t understand what you mean!'
        elif not action.applicable(self.strips_state):
            return 'You are not able to do so.'
        else:
            self.strips_state = action.apply(self.strips_state)
            self.object_dict[location]['states']['dusty'] = False
            self.object_dict[location]['states']['stained'] = False

        return 'You clean up the {} with the {}.'.format(self.wrap(location), self.wrap(tool))

    def wrap(self, obj, label=False):
        return name_wrap(obj, self.obj_name_and_article, label=label)

    def get_human_holding(self):
        expr = ''
        human_holding = self.wrap(self.human_holding)
        for name in human_holding:
            expr += 'the ' + name + ', '
        if expr:
            expr = expr[:-2]
        else:
            expr = 'nothing'
        return expr

    def get_applicable_actions(self):
        # TODO: take too much time
        applicable_actions = list()
        return applicable_actions

    def get_valid_actions(self, pick_and_place_only=False, extension=False):
        if extension:
            valid_actions = [
                'help',
                'look',
                'inventory',
                'question: can you say it clearly',
                'question: which type do you mean',
                'question: which color do you like',
                'question: which size do you like',
                'question: where is the object you want',
                'question: do you want a dusty one',
                'question: do you want a cooked one',
                'question: do you want a frozen one',
                'question: do you want a sliced one',
                'question: do you want a toggled one',
                'question: do you want a soaked one',
                'question: do you want a open one',
                'question: where do you want to place it',
            ]
        else:
            valid_actions = list()
        for loc in self.locations:
            if loc == 'h':
                valid_actions.append('move to human')
            else:
                valid_actions.append('move to {}'.format(loc))
        # pick and place, open, toggle, slice, clean
        if self.current_location == 'h':
            if self.holding:
                valid_actions.append('give {} to human'.format(self.wrap(self.holding, label=True)))
            else:
                for obj in self.human_holding:
                    valid_actions.append('take {} from human'.format(self.wrap(obj, label=True)))
        elif self.holding:
            if self.holding.split('#')[0] in ['knife']:
                knife = True
            else:
                knife = False
            if self.holding.split('#')[0] in ['rag', 'dishtowel', 'hand_towel', 'scrub_brush', 'vacuum', 'broom']:
                clean = True
            else:
                clean = False
            loc = self.current_location
            if 'has-inside' in self.object_dict[loc]['ability']:
                if 'openable' in self.object_dict[loc]['ability']:
                    if not self.object_dict[loc]['states']['open']:
                        valid_actions.append('open {}'.format(loc))
                        prep = None
                    else:
                        valid_actions.append('close {}'.format(loc))
                        prep = 'inside'
                else:
                    prep = 'inside'
            else:
                prep = 'ontop'
            if not pick_and_place_only:
                if 'toggleable' in self.object_dict[loc]['ability']:
                    if self.object_dict[loc]['states']['toggled']:
                        valid_actions.append('toggle off {}'.format(loc))
                    else:
                        valid_actions.append('toggle on {}'.format(loc))
                if clean:
                    if 'stainable' in self.object_dict[loc]['ability']:
                        valid_actions.append('clean {} with {}'.format(loc, self.wrap(self.holding, label=True)))
            if prep:
                m = {'inside': 'into', 'ontop': 'onto'}
                valid_actions.append('put {} {} {}'.format(self.wrap(self.holding, label=True), m[prep], loc))
                for rec in self.object_dict.keys():
                    if prep not in self.object_dict[rec].keys():
                        continue
                    if self.object_dict[rec][prep] != loc:
                        continue
                    if not pick_and_place_only:
                        if 'toggleable' in self.object_dict[rec]['ability']:
                            if self.object_dict[rec]['states']['toggled']:
                                valid_actions.append('toggle off {}'.format(self.wrap(rec, label=True)))
                            else:
                                valid_actions.append('toggle on {}'.format(self.wrap(rec, label=True)))
                        if knife:
                            if 'sliceable' in self.object_dict[rec]['ability']:
                                valid_actions.append('slice {} with {}'.format(self.wrap(rec, label=True),
                                                                               self.wrap(self.holding, label=True)))
                        if clean:
                            if 'dustyable' in self.object_dict[rec]['ability']:
                                valid_actions.append('clean {} with {}'.format(self.wrap(rec, label=True),
                                                                               self.wrap(self.holding, label=True)))

                    if 'has-inside' in self.object_dict[rec]['ability']:
                        if 'openable' in self.object_dict[rec]['ability']:
                            if not self.object_dict[rec]['states']['open']:
                                valid_actions.append('open {}'.format(rec))
                            else:
                                valid_actions.append('close {}'.format(rec))
                                if not self.object_dict[self.holding]['class'] == 'RECEPTACLE':
                                    valid_actions.append('put {} into {}'.format(self.wrap(self.holding, label=True), self.wrap(rec, label=True)))
                        else:
                            if not self.object_dict[self.holding]['class'] == 'RECEPTACLE':
                                valid_actions.append('put {} into {}'.format(self.wrap(self.holding, label=True), self.wrap(rec, label=True)))
                    elif 'has-ontop' in self.object_dict[rec]['ability']:
                        if not self.object_dict[self.holding]['class'] == 'RECEPTACLE':
                            valid_actions.append('put {} onto {}'.format(self.wrap(self.holding, label=True), self.wrap(rec, label=True)))
        else:
            loc = self.current_location
            if 'has-inside' in self.object_dict[loc]['ability']:
                if 'openable' in self.object_dict[loc]['ability']:
                    if not self.object_dict[loc]['states']['open']:
                        valid_actions.append('open {}'.format(loc))
                        prep = None
                    else:
                        valid_actions.append('close {}'.format(loc))
                        prep = 'inside'
                else:
                    prep = 'inside'
            else:
                prep = 'ontop'
            if not pick_and_place_only:
                if 'toggleable' in self.object_dict[loc]['ability']:
                    if self.object_dict[loc]['states']['toggled']:
                        valid_actions.append('toggle off {}'.format(loc))
                    else:
                        valid_actions.append('toggle on {}'.format(loc))

            if prep:
                for rec in self.object_dict.keys():  # rec or obj
                    if prep not in self.object_dict[rec].keys():
                        continue
                    if self.object_dict[rec][prep] != loc:
                        continue
                    if not pick_and_place_only:
                        if 'toggleable' in self.object_dict[rec]['ability']:
                            if self.object_dict[rec]['states']['toggled']:
                                valid_actions.append('toggle off {}'.format(self.wrap(rec, label=True)))
                            else:
                                valid_actions.append('toggle on {}'.format(self.wrap(rec, label=True)))

                    valid_actions.append('pick up {}'.format(self.wrap(rec, label=True)))
                    if 'has-inside' in self.object_dict[rec]['ability']:
                        for obj in self.object_dict.keys():
                            if 'inside' not in self.object_dict[obj].keys():
                                continue
                            if self.object_dict[obj]['inside'] != rec:
                                continue
                            if 'openable' in self.object_dict[rec]['ability']:
                                if not self.object_dict[rec]['states']['open']:
                                    valid_actions.append('open {}'.format(rec))
                                else:
                                    valid_actions.append('close {}'.format(rec))
                                    valid_actions.append('pick up {} from {}'.format(self.wrap(obj, label=True), self.wrap(rec, label=True)))
                            else:
                                valid_actions.append('pick up {} from {}'.format(self.wrap(obj, label=True), self.wrap(rec, label=True)))
                    elif 'has-ontop' in self.object_dict[rec]['ability']:
                        for obj in self.object_dict.keys():
                            if 'ontop' not in self.object_dict[obj].keys():
                                continue
                            if self.object_dict[obj]['ontop'] != rec:
                                continue
                            valid_actions.append('pick up {} from {}'.format(self.wrap(obj, label=True), self.wrap(rec, label=True)))
            if pick_and_place_only:
                valid_actions = list(set(valid_actions))
                return valid_actions

            if 'toggled' in self.object_dict[loc]['states'].keys() and self.object_dict[loc]['states']['toggled']:
                # heat
                if loc in ['microwave', 'oven']:
                    for obj in self.object_dict.keys():
                        if 'inside' not in self.object_dict[obj].keys():
                            continue
                        if self.object_dict[obj]['inside'] != loc:
                            continue
                        if 'cookable' in self.object_dict[obj]['ability']:
                            valid_actions.append('heat {}'.format(self.wrap(obj, label=True)))
                if loc == 'stove':
                    for obj in self.object_dict.keys():
                        if 'ontop' not in self.object_dict[obj].keys():
                            continue
                        if self.object_dict[obj]['ontop'] != loc:
                            continue
                        if 'cookable' in self.object_dict[obj]['ability']:
                            valid_actions.append('heat {}'.format(self.wrap(obj, label=True)))
                # cool
                if loc == 'refrigerator':
                    for obj in self.object_dict.keys():
                        if 'inside' not in self.object_dict[obj].keys():
                            continue
                        if self.object_dict[obj]['inside'] != loc:
                            continue
                        if 'freezable' in self.object_dict[obj]['ability']:
                            valid_actions.append('cool {}'.format(self.wrap(obj, label=True)))
                # soak
                if loc == 'sink':
                    if self.object_dict['sink']['states']['toggled']:
                        for obj in self.object_dict.keys():
                            if 'inside' not in self.object_dict[obj].keys():
                                continue
                            if self.object_dict[obj]['inside'] != loc:
                                continue
                            if 'soakable' in self.object_dict[obj]['ability']:
                                valid_actions.append('soak {}'.format(self.wrap(obj, label=True)))

        valid_actions = list(set(valid_actions))
        return valid_actions
