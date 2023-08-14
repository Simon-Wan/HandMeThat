import numpy as np
from baseline_models.base_agent import BaseAgent
STEP_LIMIT = 40


class RepeatAgent(BaseAgent):
    def __init__(self):
        super(RepeatAgent, self).__init__()

        self.solution = list()
        self.env = None
        self.game = None
        self.object_dict = None

    def act(self, ob, reward, done, info):
        self.score = info['score']
        self.moves = info['moves']
        self.question_cost = info['question_cost']
        if len(self.solution) > self.moves:
            action = self.solution[self.moves]
        else:
            action = 'stop'
        return action

    def reset(self, env):
        # import ipdb; ipdb.set_trace()
        self.score = 0
        self.moves = 0
        self.question_cost = 0

        self.env = env
        self.game = env.game
        self.object_dict = env.game.current_object_dict
        self.solution = list()
        utter = self.game.utterance
        if utter[0] == 'bring_me':
            used_objects = list()
            possible_objects = list()
            possible_types = list()
            for op in self.game.action_list:
                if 'pick-up' in op['name']:
                    obj = op['arguments'][1]
                    used_objects.append(obj)
                    possible_types.append(obj.split('#')[0])
            possible_types = list(set(possible_types))
            for obj in self.object_dict.keys():
                if obj in used_objects:
                    continue
                if self.object_dict[obj]['type'] in possible_types:
                    possible_objects.append(obj)
            satisfied_objects = self.game._objects_in_utterance
            satisfied_objects = [pair[0] for pair in satisfied_objects]

            possible_objects = [obj for obj in possible_objects if obj in satisfied_objects]
            if not possible_objects:
                possible_objects = satisfied_objects

            if len(self.solution) < STEP_LIMIT:
                obj = np.random.choice(possible_objects)
                possible_objects.remove(obj)
                self.solution += self.find_and_pick_up_obj(obj)
                self.solution += self.give_to_human(obj)
        elif utter[0] == 'move_to':
            target_pos = utter[2]
            used_objects = list()
            possible_objects = list()
            possible_types = list()
            for op in self.game.action_list:
                if 'pick-up' in op['name']:
                    obj = op['arguments'][1]
                    used_objects.append(obj)
                    possible_types.append(obj.split('#')[0])
            possible_types = list(set(possible_types))
            for obj in self.object_dict.keys():
                if obj in used_objects:
                    continue
                if self.object_dict[obj]['type'] in possible_types:
                    possible_objects.append(obj)
            satisfied_objects = self.game.get_objects_in_utterance()
            satisfied_objects = [pair[0] for pair in satisfied_objects]
            possible_objects = [obj for obj in possible_objects if obj in satisfied_objects]

            if not possible_objects:
                possible_objects = satisfied_objects

            possible_positions = list()
            if not target_pos:
                for op in self.game.action_list:
                    if 'put-inside' in op['name'] or 'put-ontop' in op['name']:
                        pos = op['arguments'][2]
                        possible_positions.append(pos)
            else:
                possible_positions.append(target_pos)

            possible_objects = list(set(possible_objects))
            possible_positions = list(set(possible_positions))
            cache = list()
            # todo: delete this part later
            # possible_positions = [pos for pos in possible_positions if '#' not in pos]
            if not possible_positions:
                return
            # pos = np.random.choice(possible_positions)
            # possible_positions = [pos]

            # todo: part end
            if len(self.solution) < STEP_LIMIT:
                obj = np.random.choice(possible_objects)

                pos = np.random.choice(possible_positions)
                self.solution += self.find_and_pick_up_obj(obj)
                self.solution += self.move_to_place(obj, pos)

    def find_and_pick_up_obj(self, obj):
        actions = list()
        if 'inside' in self.game.current_object_dict[obj].keys():
            pos = self.object_dict[obj]['inside']
        elif 'ontop' in self.game.current_object_dict[obj].keys():
            pos = self.object_dict[obj]['ontop']
        else:
            return []
        if self.object_dict[pos]['class'] == 'LOCATION':
            actions.append('move to {}'.format(pos))
            if 'openable' in self.object_dict[pos]['ability'] and not self.object_dict[pos]['states']['open']:
                actions.append('open {}'.format(pos))
            actions.append('pick up {}'.format(self.env.wrap(obj, label=True)))
        else:
            if 'inside' in self.game.current_object_dict[pos].keys():
                loc = self.object_dict[pos]['inside']
            elif 'ontop' in self.game.current_object_dict[pos].keys():
                loc = self.object_dict[pos]['ontop']
            else:
                return []
            if self.object_dict[loc]['class'] == 'LOCATION':
                actions.append('move to {}'.format(loc))
                if 'openable' in self.object_dict[loc]['ability'] and not self.object_dict[loc]['states']['open']:
                    actions.append('open {}'.format(loc))
                if 'openable' in self.object_dict[pos]['ability'] and not self.object_dict[pos]['states']['open']:
                    actions.append('open {}'.format(pos))
                actions.append('pick up {} from {}'.format(self.env.wrap(obj, label=True), self.env.wrap(pos, label=True)))
            else:
                return []
        return actions

    def give_to_human(self, obj):
        actions = [
            'move to human',
            'give {} to human'.format(self.env.wrap(obj, label=True))
        ]
        return actions

    def move_to_place(self, obj, pos):
        actions = list()
        if 'has-inside' in self.object_dict[pos]['ability']:
            prep = 'into'
        elif 'has-ontop' in self.object_dict[pos]['ability']:
            prep = 'onto'
        else:
            return []
        if self.object_dict[pos]['class'] == 'LOCATION':
            actions.append('move to {}'.format(pos))
            if 'openable' in self.object_dict[pos]['ability'] and not self.object_dict[pos]['states']['open']:
                actions.append('open {}'.format(pos))
            actions.append('put {} {} {}'.format(self.env.wrap(obj, label=True), prep, pos))
        else:
            if 'inside' in self.game.current_object_dict[pos].keys():
                loc = self.object_dict[pos]['inside']
            elif 'ontop' in self.game.current_object_dict[pos].keys():
                loc = self.object_dict[pos]['ontop']
            else:
                return []
            actions.append('move to {}'.format(loc))
            if 'openable' in self.object_dict[loc]['ability'] and not self.object_dict[loc]['states']['open']:
                actions.append('open {}'.format(loc))
            if 'openable' in self.object_dict[pos]['ability'] and not self.object_dict[pos]['states']['open']:
                actions.append('open {}'.format(pos))
            actions.append('put {} {} {}'.format(self.env.wrap(obj, label=True), prep, self.env.wrap(pos, label=True)))

        return actions