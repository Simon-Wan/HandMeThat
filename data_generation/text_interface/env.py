from data_generation.text_interface.robot_actions import *
from data_generation.text_interface.robot_space import *
from data_generation.text_interface.text_utils import *
from data_generation.data_class import load_from_json
import pdsketch as pds
import os.path as osp

DOMAIN_FILE = osp.join(osp.dirname(__file__), 'robot_domain.pddl')
DOMAIN = pds.load_domain_file(DOMAIN_FILE)
reward_shaping = False   # Set True to enable reward shaping


class HMTEnv:
    def __init__(self, json_file, fully):
        self.game = load_from_json(json_file)
        self.demo_actions = self.game.demo_actions
        self.object_dict = self.game.current_object_dict.copy()    # unchanged dict
        self.meaning = self.game.get_meaning()
        self.bindings = None

        self.locations = get_all_locations(self.object_dict)
        self.obj_name_and_article, self.abbr_obj_names = get_obj_name_and_article(self.object_dict)
        self.domain = DOMAIN
        self.session = pds.Session(self.domain)
        self.names, self.state, self.translator, self.rel_types, self.action_ops, self.strips_state \
            = set_up_planning_env(self.object_dict, self.session)
        self.goal = get_meaning_goal(self.meaning, self.game.get_objects_in_meaning(), self.state, self.translator)
        self.interaction = None

        self.info = dict()
        self.set_info()
        self.fully = fully
        self.task_description = None
        self.counter = 0

    def reset(self, level1=False):
        self.interaction = Interaction(self.game.current_object_dict.copy(), self.obj_name_and_article, self.state,
                                       self.strips_state.copy(), self.action_ops, self.translator)
        obs = self.get_initial_message(level1)
        self.set_info()
        self.info['next_expert_action'] = self.demo_actions[0]
        self.counter = 0

        return obs, self.info

    def step(self, action_str):
        reward = -1
        done = False
        if action_str == 'help':
            obs = self.help()
        elif action_str == 'stop':
            obs = 'You stop the game.\n'
            done = True
        elif action_str == 'look' or action_str == 'examine':
            obs = self.get_look(fully=self.fully)
            reward = 0      # Set negative value to punish observations
        elif action_str == 'inventory':
            obs = self.get_inventory()
        elif 'question:' in action_str.lower():
            obs, reward = self.question(action_str)
        else:
            obs, reward, done = self.interact(action_str, self.fully)

        self.info['moves'] += 1
        if reward_shaping:
            if self.demo_actions:
                if action_str == self.demo_actions[0]:
                    print(action_str)
                    self.counter += 1
                    reward += self.counter * 10
                    self.demo_actions.pop(0)
        else:
            if self.demo_actions:
                self.demo_actions.pop(0)
        if self.demo_actions:
            self.info['next_expert_action'] = self.demo_actions[0]
        else:
            self.info['next_expert_action'] = 'give up'
        self.info['score'] += reward

        return obs, reward, done, self.info

    def get_valid_actions(self):
        return self.interaction.get_valid_actions()

    def get_initial_message(self, level1=False):
        scene_message = self.scene_msg()
        obs = 'Welcome to the world!\n' + scene_message
        self.task_description = self.get_task_description(level1)
        obs += self.task_description
        obs += 'Now you are standing on the floor.\n' + self.get_look(fully=self.fully)
        obs += 'Now it is your turn to help human to achieve the goal!\n'
        return obs

    def get_task_description(self, level1=False):
        if self.game.action_list:
            obs = '\nThe human agent has taken a list of actions towards a goal, which includes:\n'
            for action in self.game.action_list:
                obs += print_action(action, self.obj_name_and_article) + '\n'
        else:
            obs = '\nYou cannot see whether the human agent has taken any actions towards the goal.\n'
        if not level1:
            obs += 'Human stops and says, \'' + print_utterance(self.game.utterance, self.obj_name_and_article) + '\'\n'
        else:
            obs += 'Human stops and says, \'' + print_utterance(self.game.get_meaning(), self.obj_name_and_article) + '\'\n'
        return obs

    def help(self):
        obs = 'The valid actions you can perform:\n' \
                    'move to [LOCATION];\n' \
                    'pick up [MOVABLE];\n' \
                    'pick up [MOVABLE] from [RECEPTACLE];\n' \
                    'put [MOVABLE] into [RECEPTACLE/LOCATION];\n' \
                    'put [MOVABLE] onto [RECEPTACLE/LOCATION];\n' \
                    'take [MOVABLE] from human;\n' \
                    'give [MOVABLE] to human;\n' \
                    'open [RECEPTACLE/LOCATION];\n' \
                    'close [RECEPTACLE/LOCATION];\n' \
                    'toggle on [TOGGLEABLE];\n' \
                    'toggle off [TOGGLEABLE];\n' \
                    'heat [COOKABLE];\n' \
                    'cool [FREEZABLE];\n' \
                    'soak [SOAKABLE];\n' \
                    'slice [SLICEABLE] with [TOOL];\n' \
                    'clean [CLEANABLE] with [TOOL].\n' \
                    '\nThe valid questions you can ask:\n' \
                    'question: can you say it clearly;\n' \
                    'question: which type do you mean;\n' \
                    'question: which [color/size] do you like;\n' \
                    'question: where is the object you want;\n' \
                    'question: do you want a [dusty/cooked/frozen/sliced/toggled/soaked/open] one;\n' \
                    'question: where do you want to place it.\n'
        return obs

    def get_look(self, fully=False, html_used=False):
        if html_used:
            loc = self.interaction.current_location
            if loc == 'h':
                obs = '\nYou are in front of the human, who is currently holding {}.\n'.format(
                    self.interaction.get_human_holding())
            else:
                obs = '\nYou are at the {}.\n'.format(loc) + \
                      print_obj_at_loc(self.interaction.object_dict, loc, self.obj_name_and_article, fully=True)
            return obs
        loc = self.interaction.current_location
        if loc == 'h':
            obs = '\nYou are in front of the human, who is currently holding {}.\n'.format(
                self.interaction.get_human_holding())
        else:
            obs = '\nYou are at the {}.\n'.format(loc) + \
                       print_obj_at_loc(self.interaction.object_dict, loc, self.obj_name_and_article)
        if fully:
            for location in self.locations:
                if location in [loc, 'h']:
                    continue
                obs += print_obj_at_loc(self.interaction.object_dict, location, self.obj_name_and_article, fully=True)
            if loc != 'h':
                obs += '\nHuman is currently holding {}.\n'.format(self.interaction.get_human_holding())
        return obs

    def get_inventory(self):
        if self.interaction.holding:
            obs = 'You are holding {}.\n'.format(
                print_obj(self.interaction.object_dict, self.interaction.holding, self.obj_name_and_article))
        else:
            obs = 'You are holding nothing.\n'

        obs += 'Recall your task:\n'
        obs += self.task_description
        return obs

    def interact(self, action, fully):
        action = action.replace('.', ' ')
        action = action.replace(',', ' ')
        action = action.replace(' # ', '#')
        action = action.replace(' 0', '#0')
        action = action.replace(' 1', '#1')
        action = action.replace(' 2', '#2')
        action = action.replace(' 3', '#3')
        action = action.replace(' _ ', '_')
        words = action.lower().split()
        obs = 'I cannot understand.'
        done = False
        reward = -1
        if not words:
            obs = 'I cannot understand.'
            return obs, reward, done
        for idx, word in enumerate(words):
            if '#' not in word:
                if word in self.abbr_obj_names.keys():
                    words[idx] = self.abbr_obj_names[word]
        if words[0] == 'move':
            if len(words) >= 3:
                if words[-1] in self.object_dict.keys():
                    obs = self.interaction.move(words[-1])
                elif words[-1] == 'human':
                    obs = self.interaction.move('human')
        elif words[0] == 'pick' and 'from' not in words:
            if len(words) >= 3:
                if words[-1] in self.object_dict.keys():
                    obs = self.interaction.pick_up_at_location(words[-1])
        elif words[0] == 'pick' and 'from' in words:
            idx = words.index('from')
            former = words[:idx]
            latter = words[idx:]
            if len(former) >= 3 and len(latter) >= 2:
                if former[-1] in self.object_dict.keys() and latter[-1] in self.object_dict.keys():
                    obs = self.interaction.pick_up_from_receptacle_at_location(former[-1], latter[-1])
        elif words[0] == 'put' and 'into' in words:
            idx = words.index('into')
            former = words[:idx]
            latter = words[idx:]
            if len(words) >= 2 and len(latter) >= 2:
                if former[-1] in self.object_dict.keys() and latter[-1] in self.object_dict.keys():
                    if self.object_dict[latter[-1]]['class'] == 'LOCATION':
                        obs = self.interaction.put_inside_location(former[-1], latter[-1])
                    elif self.object_dict[latter[-1]]['class'] == 'RECEPTACLE':
                        obs = self.interaction.put_inside_receptacle_at_location(former[-1], latter[-1])
        elif words[0] == 'put' and 'onto' in words:
            idx = words.index('onto')
            former = words[:idx]
            latter = words[idx:]
            if len(words) >= 2 and len(latter) >= 2:
                if former[-1] in self.object_dict.keys() and latter[-1] in self.object_dict.keys():
                    if self.object_dict[latter[-1]]['class'] == 'LOCATION':
                        obs = self.interaction.put_ontop_location(former[-1], latter[-1])
                    elif self.object_dict[latter[-1]]['class'] == 'RECEPTACLE':
                        obs = self.interaction.put_ontop_receptacle_at_location(former[-1], latter[-1])
        elif words[0] == 'take' and 'from' in words:
            idx = words.index('from')
            former = words[:idx]
            if len(former) >= 2:
                if former[-1] in self.object_dict.keys():
                    obs = self.interaction.take_from_human(former[-1])
        elif words[0] == 'give' and 'to' in words:
            idx = words.index('to')
            former = words[:idx]
            if len(former) >= 2:
                if former[-1] in self.object_dict.keys():
                    obs = self.interaction.give_to_human(former[-1])
        elif words[0] == 'open':
            if len(words) >= 2:
                if words[-1] in self.object_dict.keys():
                    if self.object_dict[words[-1]]['class'] == 'LOCATION':
                        obs = self.interaction.open_location(words[-1])
                    elif self.object_dict[words[-1]]['class'] == 'RECEPTACLE':
                        obs = self.interaction.open_receptacle_at_location(words[-1])
        elif words[0] == 'close':
            if len(words) >= 2:
                if words[-1] in self.object_dict.keys():
                    if self.object_dict[words[-1]]['class'] == 'LOCATION':
                        obs = self.interaction.close_location(words[-1])
                    elif self.object_dict[words[-1]]['class'] == 'RECEPTACLE':
                        obs = self.interaction.close_receptacle_at_location(words[-1])
        elif words[0] == 'toggle' and 'on' in words:
            if len(words) >= 3:
                if words[-1] in self.object_dict.keys():
                    if self.object_dict[words[-1]]['class'] == 'LOCATION':
                        obs = self.interaction.toggle_on_location(words[-1])
                    else:
                        obs = self.interaction.toggle_on_movable_at_location(words[-1])
        elif words[0] == 'toggle' and 'off' in words:
            if len(words) >= 3:
                if words[-1] in self.object_dict.keys():
                    if self.object_dict[words[-1]]['class'] == 'LOCATION':
                        obs = self.interaction.toggle_off_location(words[-1])
                    else:
                        obs = self.interaction.toggle_off_movable_at_location(words[-1])
        elif words[0] == 'heat':
            if len(words) >= 2:
                if words[-1] in self.object_dict.keys():
                    obs = self.interaction.heat_obj(words[-1])
        elif words[0] == 'cool':
            if len(words) >= 2:
                if words[-1] in self.object_dict.keys():
                    obs = self.interaction.cool_obj(words[-1])
        elif words[0] == 'soak':
            if len(words) >= 2:
                if words[-1] in self.object_dict.keys():
                    obs = self.interaction.soak_obj(words[-1])
        elif words[0] == 'clean' and 'with' in words:
            idx = words.index('with')
            former = words[:idx]
            latter = words[idx:]
            if len(former) >= 2 and len(latter) >= 2:
                if former[-1] in self.object_dict.keys() and latter[-1] in self.object_dict.keys():
                    if self.object_dict[former[-1]]['class'] == 'LOCATION':
                        obs = self.interaction.clean_location(latter[-1], former[-1])
                    else:
                        obs = self.interaction.clean_obj_at_location(former[-1], latter[-1])
        elif words[0] == 'slice' and 'with' in words:
            idx = words.index('with')
            former = words[:idx]
            latter = words[idx:]
            if len(former) >= 2 and len(latter) >= 2:
                if former[-1] in self.object_dict.keys() and latter[-1] in self.object_dict.keys():
                    obs = self.interaction.slice_obj(former[-1], latter[-1])
        if self.check_goal():
            obs += '\nYou reach the goal!\n'
            done = True
            reward += 100       # Set winning reward

        return obs, reward, done

    def question(self, question):
        obs = 'I cannot understand.'
        reward = -10            # Set negative value to punish questions
        self.info['question_cost'] += 10
        if 'can you say it clearly' in question.lower():
            obs = 'I mean {}.'.format(get_object_description(self.meaning[1], self.obj_name_and_article))
            reward = -30
            self.info['question_cost'] += 20
        elif 'which type do you mean' in question.lower():
            if 'type' in self.meaning[1].keys():
                obs = 'I mean the {}.'.format(self.meaning[1]['type'])
            elif 'subclass' in self.meaning[1].keys():
                obs = 'I just want a {}.'.format(self.meaning[1]['subclass'])
            elif 'class' in self.meaning[1].keys():
                obs = 'I just want a {}.'.format(self.meaning[1]['class'].lower())
            else:
                obs = 'Not any specific type.'
        elif 'which color do you like' in question.lower():
            if 'color' in self.meaning[1].keys():
                obs = 'The {} one.'.format(self.meaning[1]['color'])
            else:
                obs = 'Not any specific color.'
        elif 'which size do you like' in question.lower():
            if 'size' in self.meaning[1].keys():
                obs = 'The {} one.'.format(self.meaning[1]['size'])
            else:
                obs = 'Not any specific size.'
        elif 'where is the object you want' in question.lower():
            if 'inside' in self.meaning[1].keys():
                obs = 'Inside the {}.'.format(name_wrap(self.meaning[1]['inside'], self.obj_name_and_article))
            elif 'ontop' in self.meaning[1].keys():
                obs = 'On top of the {}.'.format(name_wrap(self.meaning[1]['ontop'], self.obj_name_and_article))
            else:
                obs = 'Not any specific position.'
        elif 'do you want a ' in question.lower():
            attr = question.split()[4]
            if attr in ['dusty', 'cooked', 'frozen', 'sliced', 'toggled', 'soaked', 'open']:
                if attr in self.meaning[1].keys():
                    if self.meaning[1][attr]:
                        obs = 'Yes, I mean the {} one.'.format(attr)
                    else:
                        obs = 'No, I mean the not {} one.'.format(attr)
                else:
                    obs = 'I don\'t care.'
        elif 'where do you want to place it' in question.lower():
            if self.meaning[0] != 'move_to':
                obs = 'I cannot understand.'
            else:
                obs = 'Put it at the {}.'.format(name_wrap(self.meaning[2], self.obj_name_and_article))
        return obs, reward

    def scene_msg(self):
        obs = 'In the room, there is '
        for loc in self.locations:
            if loc == 'floor':
                continue
            if loc == 'h':  
                obs += 'the human, '
                continue
            obs += self.obj_name_and_article[loc]['article'] + ' ' + loc + ', '
        obs = obs[:-2] + '.\n'
        return obs

    def check_goal(self):
        goal_func = self.goal.compile()
        return goal_func(self.interaction.strips_state)

    def set_info(self, info=None):
        if info is None:
            self.info = {'moves': 0, 'score': 0, 'question_cost': 0}
        else:
            self.info = info
        return

    def wrap(self, name, label=False):
        return name_wrap(name, self.obj_name_and_article, label=label)
