import numpy as np
from data_generation.sampling.goal_utils import *


# goal input example
# ['exists_y_forall_x_x_at_y [fruit]1 (sliced) [jar]1',
# 'exists_y_forall_x_x_at_y [fruit]2 (sliced) [jar]2']


class Goal:
    def __init__(self, sub_goals_args, object_dict, subclasses, types):
        self.valid = True

        self.sub_goals_args = sub_goals_args  # a list of arguments for each sub-goal
        self.object_dict = object_dict  # a dict of all objects
        self.subclasses = subclasses  # a dict of all subclasses and their children's names
        self.types = types  # a dict of the number of each object type

        self.locations = [name for name in object_dict.keys() if object_dict[name]['class'] == 'LOCATION']

        self.sample_replace = dict()  # a temp dict to store all sampling results
        self.sub_goals_args_replaced = sub_goals_args.copy()
        self.sub_goals = list()

        for sub_goal_idx, args in enumerate(self.sub_goals_args):
            args_replaced = args
            for arg_idx, arg in enumerate(args.split()):
                if arg[0] == '[':
                    try:
                        self.sample_replace[arg] = self.sample(arg)
                    except ValueError as err:
                        print(repr(err))
                        self.valid = False
                        break
                    args_replaced = args_replaced.replace(arg, '[' + self.sample_replace[arg] + ']')
            self.sub_goals_args_replaced[sub_goal_idx] = args_replaced
            if not self.valid:
                break
        for args in self.sub_goals_args_replaced:
            self.sub_goals.append(self.get_sub_goal(args))

        self.expr = '(and '
        self.expr_list = list()
        self.rel_types = {'LOCATION': [], 'RECEPTACLE': [], 'FOOD': [], 'TOOL': [], 'THING': []}
        self.rel_targets = {'MOVABLE': [], 'POSITION': []}
        for goal in self.sub_goals:
            self.expr += goal.expr
            self.expr_list.append(goal.expr)
            for key in self.rel_types.keys():
                self.rel_types[key] += goal.rel_types[key]
            for key in self.rel_targets.keys():
                self.rel_targets[key] += goal.rel_targets[key]
        self.expr += ')'
        if 'sliced' in self.expr:
            update_rel_types(self.rel_types, self.object_dict, 'knife')
            update_rel_targets(self.rel_targets, 'knife', '', self.object_dict)
        if 'dusty' in self.expr or 'stained' in self.expr:
            update_rel_types(self.rel_types, self.object_dict, 'sink')
            update_rel_types(self.rel_types, self.object_dict, 'rag')
            update_rel_types(self.rel_types, self.object_dict, 'dishtowel')
            update_rel_types(self.rel_types, self.object_dict, 'hand_towel')
            update_rel_types(self.rel_types, self.object_dict, 'scrub_brush')
            update_rel_targets(self.rel_targets, '', 'sink', self.object_dict)
            update_rel_targets(self.rel_targets, 'rag', '', self.object_dict)
            update_rel_targets(self.rel_targets, 'dishtowel', '', self.object_dict)
            update_rel_targets(self.rel_targets, 'hand_towel', '', self.object_dict)
            update_rel_targets(self.rel_targets, 'scrub_brush', '', self.object_dict)
        if 'cooked' in self.expr:
            update_rel_types(self.rel_types, self.object_dict, 'microwave')
            update_rel_types(self.rel_types, self.object_dict, 'oven')
            update_rel_types(self.rel_types, self.object_dict, 'stove')
            update_rel_targets(self.rel_targets, '', 'microwave', self.object_dict)
            update_rel_targets(self.rel_targets, '', 'oven', self.object_dict)
            update_rel_targets(self.rel_targets, '', 'stove', self.object_dict)
        if 'frozen' in self.expr:
            update_rel_types(self.rel_types, self.object_dict, 'refrigerator')
            update_rel_targets(self.rel_targets, '', 'refrigerator', self.object_dict)
        for key in self.rel_types.keys():
            self.rel_types[key] = list(set(self.rel_types[key]))

        self.rel_targets['MOVABLE'] = list(set(self.rel_targets['MOVABLE']))
        self.rel_targets['POSITION'] = list(set(self.rel_targets['POSITION']))

    def sample(self, arg):
        if arg[1:-2] in self.locations:
            return arg[1:-2]
        if arg in self.sample_replace.keys():
            return self.sample_replace[arg]  # already sampled
        name = arg[1:-2]
        idx = arg[-1]
        if idx == '*':  # new name AND no need to be grounded
            if name in self.subclasses.keys():
                if self.subclasses[name]:
                    result = np.random.choice(self.subclasses[name])  # sample type from subclass
                else:
                    result = None
            else:
                result = name  # directly use type
            if result not in self.types.keys():
                raise ValueError('No object in type {}.'.format(result))
            return result
        else:
            if name in self.types.keys():
                choices = [name + '#' + str(i) for i in range(self.types[name])]
                for value in self.sample_replace.values():
                    if value in choices:
                        choices.remove(value)  # avoid duplicate
                if not choices:
                    raise ValueError('Not enough objects in type {}.'.format(name))
                return np.random.choice(choices)
            elif name in self.subclasses.keys():
                choices = self.subclasses[name].copy()  # possible types
                for value in self.sample_replace.values():
                    if value in choices:
                        choices.remove(value)  # avoid duplicate
                if not choices:
                    raise ValueError('Not enough types in subclass {}.'.format(name))
                return np.random.choice(choices)
            else:
                raise ValueError('Unvalid input!')

    def get_sub_goal(self, args):
        arg_list = args.split()
        if arg_list[0] == 'x_at_y':
            return x_at_y(self.object_dict, args)
        elif arg_list[0] == 'forall_x_x_at_y':
            return forall_x_x_at_y(self.object_dict, args)
        elif arg_list[0] == 'exists_x_x_at_y':
            return exists_x_x_at_y(self.object_dict, args)
        elif arg_list[0] == 'exists_y_x_at_y':
            return exists_y_x_at_y(self.object_dict, args)
        elif arg_list[0] == 'exists_y_forall_x_x_at_y':
            return exists_y_forall_x_x_at_y(self.object_dict, args)
        elif arg_list[0] == 'forall_x_exists_y_x_at_y':
            return forall_x_exists_y_x_at_y(self.object_dict, args)
        elif arg_list[0] == 'exists_y_exists_x_x_at_y':
            return exists_y_exists_x_x_at_y(self.object_dict, args)
        elif arg_list[0] == 'forall_y_exists_x_x_at_y':
            return forall_y_exists_x_x_at_y(self.object_dict, args)
        elif arg_list[0] == 'x_state_x':
            return x_state_x(self.object_dict, args)
        elif arg_list[0] == 'forall_x_state_x':
            return forall_x_state_x(self.object_dict, args)

        return


# all predicates:
# x_at_y
# forall_x_x_at_y
# exists_x_x_at_y
# exists_y_x_at_y
# exists_y_forall_x_x_at_y
# forall_x_exists_y_x_at_y
# exists_y_exists_x_x_at_y
# forall_y_exists_x_x_at_y
# x_state_x
# forall_x_state_x


class SubGoal:
    def __init__(self, object_dict, args):
        self.object_dict = object_dict
        self.args = args
        self.rel_types = {'LOCATION': [], 'RECEPTACLE': [], 'FOOD': [], 'TOOL': [], 'THING': []}
        self.rel_targets = {'MOVABLE': [], 'POSITION': []}
        self.expr = ''
        self.locations = list()
        for key in object_dict.keys():
            if object_dict[key]['class'] == 'LOCATION':
                self.locations.append(key)

    def get_target_state_expr(self, state, var):
        """
        format:
        (attr1,not_attr2,...)
        """
        state = state
        state_list = state.split(',')
        expr = ''
        for attr in state_list:
            if attr == '':
                continue
            if attr[:4] == 'not_':
                expr += '(not ({} {})) '.format(attr[4:], var)
            else:
                expr += '({} {}) '.format(attr, var)
        return expr


class x_at_y(SubGoal):  # solvable for sure
    """
    template before sampling:
    'x_at_y (state) [subclass/type]? <prep> (state) [subclass/type]?'
    template after sampling:
    'x_at_y (state) [obj] <prep> (state) [obj]'
    """

    def __init__(self, object_dict, args):
        super().__init__(object_dict, args)
        self.arg_list = args.split()
        self.x = self.arg_list[2][1:-1]
        self.y = self.arg_list[5][1:-1]
        self.prep = self.arg_list[3][1:-1]
        self.x_state = self.arg_list[1][1:-1]
        self.y_state = self.arg_list[4][1:-1]

        self.get_rel_types()
        self.expr = self.get_expr()

    def get_rel_types(self):
        update_rel_types(self.rel_types, self.object_dict, self.x)
        update_rel_types(self.rel_types, self.object_dict, self.y)
        update_rel_targets(self.rel_targets, self.x, self.y, self.object_dict)

    def get_expr(self):
        return '(and ({} {} {}){}{})'.format(
            self.prep,
            self.x,
            self.y,
            self.get_target_state_expr(self.x_state, self.x),
            self.get_target_state_expr(self.y_state, self.y),
        )


class forall_x_x_at_y(SubGoal):  # solvable for sure
    """
    template before sampling:
    'forall_x_x_at_y (state) [subclass/type]? <prep> (state) [subclass/type]?'
    template after sampling:
    'forall_x_x_at_y (state) [type] <prep> (state) [obj]'
    """

    def __init__(self, object_dict, args):
        super().__init__(object_dict, args)
        self.arg_list = args.split()
        self.x = self.arg_list[2][1:-1]
        self.y = self.arg_list[5][1:-1]
        self.prep = self.arg_list[3][1:-1]
        self.x_state = self.arg_list[1][1:-1]
        self.y_state = self.arg_list[4][1:-1]

        self.get_rel_types()
        self.expr = self.get_expr()

    def get_rel_types(self):
        update_rel_types(self.rel_types, self.object_dict, self.x)
        update_rel_types(self.rel_types, self.object_dict, self.y)
        update_rel_targets(self.rel_targets, self.x, self.y, self.object_dict)

    def get_expr(self):
        return '(forall (?x - phyobj) (or (not (type-{} ?x)) (and ({} ?x {}){}{})))'.format(
            self.x,
            self.prep,
            self.y,
            self.get_target_state_expr(self.x_state, '?x'),
            self.get_target_state_expr(self.y_state, self.y),
        )


class exists_x_x_at_y(SubGoal):  # solvable for sure
    """
    template before sampling:
    'exists_x_x_at_y (state) [subclass/type]? <prep> (state) [subclass/type]?'
    template after sampling:
    'exists_x_x_at_y (state) [type] <prep> (state) [obj]'
    """

    def __init__(self, object_dict, args):
        super().__init__(object_dict, args)
        self.arg_list = args.split()
        self.x = self.arg_list[2][1:-1]
        self.y = self.arg_list[5][1:-1]
        self.prep = self.arg_list[3][1:-1]
        self.x_state = self.arg_list[1][1:-1]
        self.y_state = self.arg_list[4][1:-1]

        self.get_rel_types()
        self.expr = self.get_expr()

    def get_rel_types(self):
        update_rel_types(self.rel_types, self.object_dict, self.x)
        update_rel_types(self.rel_types, self.object_dict, self.y)
        update_rel_targets(self.rel_targets, self.x, self.y, self.object_dict)

    def get_expr(self):
        return '(exists (?x - phyobj) (and (type-{} ?x) ({} ?x {}){}{}))'.format(
            self.x,
            self.prep,
            self.y,
            self.get_target_state_expr(self.x_state, '?x'),
            self.get_target_state_expr(self.y_state, self.y),
        )


class exists_y_x_at_y(SubGoal):  # solvable for sure
    """
    template before sampling:
    'exists_y_x_at_y (state) [subclass/type]? <prep> (state) [subclass/type]?'
    template after sampling:
    'exists_y_x_at_y (state) [obj] <prep> (state) [type]'
    """

    def __init__(self, object_dict, args):
        super().__init__(object_dict, args)
        self.arg_list = args.split()
        self.x = self.arg_list[2][1:-1]
        self.y = self.arg_list[5][1:-1]
        self.prep = self.arg_list[3][1:-1]
        self.x_state = self.arg_list[1][1:-1]
        self.y_state = self.arg_list[4][1:-1]

        self.get_rel_types()
        self.expr = self.get_expr()

    def get_rel_types(self):
        update_rel_types(self.rel_types, self.object_dict, self.x)
        update_rel_types(self.rel_types, self.object_dict, self.y)
        update_rel_targets(self.rel_targets, self.x, self.y, self.object_dict)

    def get_expr(self):
        return '(exists (?y - phyobj) (and (type-{} ?y) ({} {} ?y){}{}))'.format(
            self.y,
            self.prep,
            self.x,
            self.get_target_state_expr(self.x_state, self.x),
            self.get_target_state_expr(self.y_state, '?y'),
        )


class exists_y_forall_x_x_at_y(SubGoal):  # solvable for sure
    """
    template before sampling:
    'exists_y_forall_x_x_at_y (state) [subclass/type]? <prep> (state) [subclass/type]?'
    template after sampling:
    'exists_y_forall_x_x_at_y (state) [type] <prep> (state) [type]'
    """

    def __init__(self, object_dict, args):
        super().__init__(object_dict, args)
        self.arg_list = args.split()
        self.x = self.arg_list[2][1:-1]
        self.y = self.arg_list[5][1:-1]
        self.prep = self.arg_list[3][1:-1]
        self.x_state = self.arg_list[1][1:-1]
        self.y_state = self.arg_list[4][1:-1]

        self.get_rel_types()
        self.expr = self.get_expr()

    def get_rel_types(self):
        update_rel_types(self.rel_types, self.object_dict, self.x)
        update_rel_types(self.rel_types, self.object_dict, self.y)
        update_rel_targets(self.rel_targets, self.x, self.y, self.object_dict)

    def get_expr(self):
        return '(exists (?y - phyobj) (and (type-{} ?y) (forall (?x - phyobj) (or (not (type-{} ?x)) (and ({} ?x ?y){}{})))))'.format(
            self.y,
            self.x,
            self.prep,
            self.get_target_state_expr(self.x_state, '?x'),
            self.get_target_state_expr(self.y_state, '?y'),
        )


class forall_x_exists_y_x_at_y(SubGoal):  # solvable for sure
    """
    template before sampling:
    'forall_x_exists_y_x_at_y (state) [subclass/type]? <prep> (state) [subclass/type]?'
    template after sampling:
    'forall_x_exists_y_x_at_y (state) [type] <prep> (state) [type]'
    """

    def __init__(self, object_dict, args):
        super().__init__(object_dict, args)
        self.arg_list = args.split()
        self.x = self.arg_list[2][1:-1]
        self.y = self.arg_list[5][1:-1]
        self.prep = self.arg_list[3][1:-1]
        self.x_state = self.arg_list[1][1:-1]
        self.y_state = self.arg_list[4][1:-1]

        self.get_rel_types()
        self.expr = self.get_expr()

    def get_rel_types(self):
        update_rel_types(self.rel_types, self.object_dict, self.x)
        update_rel_types(self.rel_types, self.object_dict, self.y)
        update_rel_targets(self.rel_targets, self.x, self.y, self.object_dict)

    def get_expr(self):
        return '(forall (?x - phyobj) (or (not (type-{} ?x)) (exists (?y - phyobj) (and (type-{} ?y) ({} ?x ?y){}{}))))'.format(
            self.x,
            self.y,
            self.prep,
            self.get_target_state_expr(self.x_state, '?x'),
            self.get_target_state_expr(self.y_state, '?y'),
        )


class exists_y_exists_x_x_at_y(SubGoal):
    """
    template before sampling:
    'exists_y_exists_x_x_at_y (state) [subclass/type]? <prep> (state) [subclass/type]?'
    template after sampling:
    'exists_y_exists_x_x_at_y (state) [type] <prep> (state) [type]'
    """

    def __init__(self, object_dict, args):
        super().__init__(object_dict, args)
        self.arg_list = args.split()
        self.x = self.arg_list[2][1:-1]
        self.y = self.arg_list[5][1:-1]
        self.prep = self.arg_list[3][1:-1]
        self.x_state = self.arg_list[1][1:-1]
        self.y_state = self.arg_list[4][1:-1]

        self.get_rel_types()
        self.expr = self.get_expr()

    def get_rel_types(self):
        update_rel_types(self.rel_types, self.object_dict, self.x)
        update_rel_types(self.rel_types, self.object_dict, self.y)
        update_rel_targets(self.rel_targets, self.x, self.y, self.object_dict)

    def get_expr(self):
        return '(exists (?y - phyobj) (and (type-{} ?y) (exists (?x - phyobj) (and  (type-{} ?x) ({} ?x ?y){}{})))'.format(
            self.y,
            self.x,
            self.prep,
            self.get_target_state_expr(self.x_state, '?x'),
            self.get_target_state_expr(self.y_state, '?y'),
        )


class forall_y_exists_x_x_at_y(SubGoal):  # may not be solvable, better to ground with other template
    """
    template before sampling:
    'forall_y_exists_x_x_at_y (state) [subclass/type]? <prep> (state) [subclass/type]?'
    template after sampling:
    'forall_y_exists_x_x_at_y (state) [type] <prep> (state) [type]'
    """

    def __init__(self, object_dict, args):
        super().__init__(object_dict, args)
        self.arg_list = args.split()
        self.x = self.arg_list[2][1:-1]
        self.y = self.arg_list[5][1:-1]
        self.prep = self.arg_list[3][1:-1]
        self.x_state = self.arg_list[1][1:-1]
        self.y_state = self.arg_list[4][1:-1]

        self.get_rel_types()
        self.expr = self.get_expr()

    def get_rel_types(self):
        update_rel_types(self.rel_types, self.object_dict, self.x)
        update_rel_types(self.rel_types, self.object_dict, self.y)
        update_rel_targets(self.rel_targets, self.x, self.y, self.object_dict)

    def get_expr(self):
        return '(forall (?y - phyobj) (or (not (type-{} ?y)) (exists (?x - phyobj) (and  (type-{} ?x) ({} ?x ?y){}{})))'.format(
            self.y,
            self.x,
            self.prep,
            self.get_target_state_expr(self.x_state, '?x'),
            self.get_target_state_expr(self.y_state, '?y'),
        )


class x_state_x(SubGoal):  # solvable for sure
    """
    template before sampling:
    'x_state_x (state) [subclass/type]?'
    template after sampling:
    'x_state_x (state) [obj]'
    """

    def __init__(self, object_dict, args):
        super().__init__(object_dict, args)
        self.arg_list = args.split()
        self.x = self.arg_list[2][1:-1]
        self.x_state = self.arg_list[1][1:-1]

        self.get_rel_types()
        self.expr = self.get_expr()

    def get_rel_types(self):
        update_rel_types(self.rel_types, self.object_dict, self.x)
        update_rel_targets(self.rel_targets, self.x, '', self.object_dict)

    def get_expr(self):
        return '(and {})'.format(
            self.get_target_state_expr(self.x_state, self.x),
        )


class forall_x_state_x(SubGoal):  # solvable for sure
    """
    template before sampling:
    'forall_x_state_x (state) [subclass/type]?'
    template after sampling:
    'forall_x_state_x (state) [type]'
    """

    def __init__(self, object_dict, args):
        super().__init__(object_dict, args)
        self.arg_list = args.split()
        self.x = self.arg_list[2][1:-1]
        self.x_state = self.arg_list[1][1:-1]

        self.get_rel_types()
        self.expr = self.get_expr()

    def get_rel_types(self):
        update_rel_types(self.rel_types, self.object_dict, self.x)
        update_rel_targets(self.rel_targets, self.x, '', self.object_dict)

    def get_expr(self):
        return '(forall (?x - phyobj) (or (not (type-{} ?x)) (and {})))'.format(
            self.x,
            self.get_target_state_expr(self.x_state, '?x'),
        )


def generate_one_goal(object_dict, object_names, goal_idx=None):
    subclasses, types = get_each_hierarchy_level(object_names)
    all_goals = list()
    behavior_goals = [
        # 0
        # assembling_gift_baskets
        replace_subclass_by_same_type([
            'x_at_y () [illumination_tool]0 <inside> () [basket]0',
            'x_at_y () [snack]0 <inside> () [basket]0',
            'x_at_y () [baked_food]0 <inside> () [basket]0',
            'x_at_y () [decoration]0 <inside> () [basket]0',
            'x_at_y () [illumination_tool]1 <inside> () [basket]1',
            'x_at_y () [snack]1 <inside> () [basket]1',
            'x_at_y () [baked_food]1 <inside> () [basket]1',
            'x_at_y () [decoration]1 <inside> () [basket]1',
        ], subclasses),
        # 1
        # bottling_fruit
        [
            'forall_x_x_at_y (sliced) [fruit]0 <inside> () [jar]0',
            'forall_x_x_at_y (sliced) [fruit]1 <inside> () [jar]1',
        ],
        # 2
        # boxing_books_up_for_storage
        [
            'exists_y_forall_x_x_at_y () [paper_product]* <inside> () [box]*',
        ],
        # 3
        # bringing_in_wood
        [
            'forall_x_x_at_y () [building_materials]* <ontop> () [floor]*',
        ],
        # 4
        # brushing_lint_off_clothing
        [
            'forall_x_x_at_y (not_dusty) [clothing]* <ontop> () [bed]*',
        ],
        # 5
        # chopping_vegetables
        [
            'forall_x_exists_y_x_at_y (sliced) [fruit]0 <inside> () [tableware]0',
            'forall_x_exists_y_x_at_y (sliced) [fruit]1 <inside> () [tableware]0',
            'forall_x_exists_y_x_at_y (sliced) [vegetable]0 <inside> () [tableware]0',
            'forall_x_exists_y_x_at_y (sliced) [vegetable]1 <inside> () [tableware]0',
        ],
        # pass
        # cleaning_a_car

        # pass
        # cleaning_barbecue_grill

        # 6
        # cleaning_bathrooms
        [
            'x_state_x (not_stained) [toilet]*',
            'x_state_x (not_stained) [bathtub]*',
            'x_state_x (not_stained) [sink]*',
            'x_state_x (not_stained) [floor]*',
            'exists_y_x_at_y (soaked) [rag]0 <inside> () [bucket]*',
        ],
        # pass
        # cleaning_bathtub

        # 7
        # cleaning_bedroom
        [
            'forall_x_x_at_y () [clothing]* <inside> (not_dusty) [cabinet]*',
            'forall_x_x_at_y () [decoration]* <inside> () [cabinet]*',
            'forall_x_x_at_y () [toiletry]* <inside> () [cabinet]*',
            'forall_x_x_at_y () [paper_product]* <ontop> () [bed]*',
        ],
        # pass
        # cleaning_carpets

        # 8
        # cleaning_closet
        [
            'forall_x_x_at_y () [decoration]* <inside> (not_dusty) [cabinet]*',
            'forall_x_x_at_y () [headwear]* <ontop> (not_dusty) [shelf]*',
            'forall_x_x_at_y () [footwear]* <ontop> (not_dusty) [floor]*',
        ],
        # pass
        # cleaning_cupboards

        # 9
        # cleaning_floors
        [
            'x_state_x (not_stained,not_dusty) [floor]*',
        ],
        # pass
        # cleaning_freezer

        # 10
        # cleaning_garage
        [
            'x_state_x (not_dusty) [floor]*',
            'x_state_x (not_dusty) [cabinet]*',
            'x_state_x (not_stained) [cabinet]*',
            'forall_x_x_at_y () [paper_product]* <inside> () [ashcan]0',
            'forall_x_x_at_y () [vessel]* <ontop> () [table]*',
        ],
        # 11
        # cleaning_high_chair
        [
            'forall_x_state_x (not_dusty) [furniture]*',
        ],
        # 12
        # cleaning_kitchen_cupboard
        [
            'x_state_x (not_dusty) [cabinet]*',
            'forall_x_x_at_y () [tableware]0 <inside> () [cabinet]*',
            'forall_x_x_at_y () [tableware]1 <inside> () [cabinet]*',
        ],
        # 13
        # cleaning_microwave_oven
        [
            'x_state_x (not_dusty,not_stained) [microwave]*',
        ],
        # 14
        # cleaning_out_drawers
        [
            'forall_x_x_at_y () [piece_of_cloth]* <inside> () [sink]*',
            'forall_x_x_at_y () [tableware]* <inside> () [sink]*',
            'forall_x_x_at_y () [cutlery]* <inside> () [sink]*',
        ],
        # 15
        # cleaning_oven
        [
            'x_state_x (soaked) [rag]0',
            'x_state_x (not_stained) [oven]*',
        ],
        # 16
        # cleaning_shoes
        [
            'forall_x_x_at_y (not_dusty) [footwear]* <ontop> () [floor]*',
            'forall_x_x_at_y () [footwear]* <ontop> () [floor]*',
        ],
        # pass
        # cleaning_sneakers

        # 17
        # cleaning_stove
        [
            'x_state_x (not_dusty,not_stained) [stove]*',
            'forall_x_x_at_y () [rag]* <inside> () [sink]*',
            'forall_x_x_at_y () [dishtowel]* <inside> () [sink]*',
        ],
        # 18
        # cleaning_table_after_clearing
        [
            'x_state_x (not_stained) [table]*',
        ],
        # pass
        # cleaning_the_hot_tub

        # 19
        # cleaning_the_pool
        [
            'x_state_x (not_stained) [pool]*',
            'forall_x_x_at_y () [scrub_brush]* <ontop> () [shelf]*',
            'forall_x_x_at_y () [cleansing]* <ontop> () [floor]*',
        ],
        # 20
        # cleaning_toilet
        [
            'x_state_x (not_stained) [toilet]*',
            'forall_x_x_at_y () [scrub_brush]* <ontop> () [floor]*',
            'forall_x_x_at_y () [cleansing]* <ontop> () [floor]*',
        ],
        # 21
        # cleaning_up_after_a_meal      # todo: take long time to calculate robot action rewards
        [
            'forall_x_state_x (not_dusty) [tableware]0',
            'forall_x_state_x (not_dusty) [tableware]1',
            'forall_x_state_x (not_dusty) [tableware]2',
            'exists_y_forall_x_x_at_y () [snack]* <inside> () [bag]*',
            'x_state_x (not_stained) [table]*',
            'x_state_x (not_stained) [floor]*',
            'forall_x_state_x (not_dusty) [furniture]*',
        ],
        # 22
        # cleaning_up_refrigerator      # todo: take long time to calculate robot action rewards
        [
            'forall_x_x_at_y () [rag]* <inside> () [sink]*',
            'forall_x_x_at_y () [cleansing]* <inside> () [sink]*',
            'forall_x_x_at_y (not_dusty) [tray]* <inside> (not_stained) [refrigerator]*',
            'forall_x_x_at_y (not_dusty) [tableware]* <inside> () [sink]*',
        ],
        # 23
        # cleaning_up_the_kitchen_only
        [
            'forall_x_x_at_y () [vessel]* <ontop> () [countertop]*',
            'forall_x_x_at_y () [cleansing]* <inside> () [sink]*',
            'forall_x_x_at_y () [flavorer]* <inside> () [cabinet]*',
            'forall_x_x_at_y (not_dusty) [tableware]* <inside> (not_dusty) [cabinet]*',
            'forall_x_x_at_y () [rag]* <inside> () [sink]*',
            'forall_x_x_at_y () [utensil]* <inside> () [refrigerator]*',
            'forall_x_x_at_y () [fruit]* <inside> () [refrigerator]*',
        ],
        # pass
        # cleaning_windows

        # 24
        # clearing_the_table_after_dinner
        [
            'forall_x_x_at_y () [cutlery]0 <inside> () [bucket]0',
            'forall_x_x_at_y () [cutlery]1 <inside> () [bucket]1',
            'exists_y_forall_x_x_at_y () [flavorer]* <inside> () [bucket]*',
        ],
        # 25
        # collect_misplaced_items
        [
            'forall_x_x_at_y () [footwear]* <ontop> () [table]*',
            'forall_x_x_at_y () [decoration]* <ontop> () [table]*',
            'forall_x_x_at_y () [paper_product]* <ontop> () [table]*',
        ],
        # 26
        # collecting_aluminum_cans
        [
            'forall_x_x_at_y () [drink]* <inside> () [ashcan]0',
        ],
        # pass
        # defrosting_freezer

        # 27
        # filling_a_Christmas_stocking
        [
            'exists_x_x_at_y () [plaything]0 <inside> () [xmas_stocking]0',
            'exists_x_x_at_y () [plaything]0 <inside> () [xmas_stocking]1',
            'exists_x_x_at_y () [snack]0 <inside> () [xmas_stocking]0',
            'exists_x_x_at_y () [snack]0 <inside> () [xmas_stocking]1',
            'exists_x_x_at_y () [writing_tool]0 <inside> () [xmas_stocking]0',
            'exists_x_x_at_y () [writing_tool]0 <inside> () [xmas_stocking]1',
        ],
        # 28
        # filling_an_Easter_basket
        [
            'forall_x_x_at_y () [basket]* <ontop> () [countertop]*',
            'exists_x_x_at_y () [protein]0 <inside> () [basket]0',
            'exists_x_x_at_y () [protein]0 <inside> () [basket]1',
            'exists_x_x_at_y () [snack]0 <inside> () [basket]0',
            'exists_x_x_at_y () [snack]0 <inside> () [basket]1',
            'exists_x_x_at_y () [paper_product]0 <inside> () [basket]0',
            'exists_x_x_at_y () [paper_product]0 <inside> () [basket]1',
            'exists_x_x_at_y () [plaything]0 <inside> () [basket]0',
            'exists_x_x_at_y () [plaything]0 <inside> () [basket]1',
            'exists_y_forall_x_x_at_y () [decoration]* <inside> () [basket]*',
        ],
        # 29
        # installing_a_fax_machine
        [
            'forall_x_x_at_y (toggled) [electric_equipment]* <ontop> () [table]*'
        ],
        # pass
        # installing_a_modem

        # pass
        # installing_a_printer

        # pass
        # installing_a_scanner

        # 30
        # installing_alarms
        [
            'exists_x_x_at_y () [electrical_device]0 <ontop> () [table]*',
            'exists_x_x_at_y () [electrical_device]0 <ontop> () [countertop]*',
            'exists_x_x_at_y () [electrical_device]0 <ontop> () [sofa]*',
        ],
        # 31
        # laying_tile_floors
        [
            'forall_x_x_at_y () [building_materials]* <ontop> () [floor]*'
        ],
        # pass
        # laying_wood_floors

        # 32
        # loading_the_dishwasher
        [
            'forall_x_x_at_y () [tableware]0 <inside> () [sink]*',
            'forall_x_x_at_y () [tableware]1 <inside> () [sink]*',
            'forall_x_x_at_y () [vessel]* <inside> () [sink]*',
        ],
        # pass
        # locking_every_door

        # pass
        # locking_every_window

        # pass
        # making_tea

        # pass
        # mopping_floors

        # 33
        # moving_boxes_to_storage
        [
            'forall_x_x_at_y () [box]* <ontop> () [floor]*',
        ],

        # 34
        # opening_packages
        [
            'forall_x_x_at_y (open) [package]* <ontop> () [floor]*',
        ],
        # pass
        # opening_presents

        # 35
        # organizing_boxes_in_garage
        [
            'forall_x_x_at_y () [plaything]* <inside> () [box]0',
            'forall_x_x_at_y () [cutlery]* <inside> () [box]1',
            'forall_x_x_at_y () [cleansing]* <inside> () [box]2',
            'forall_x_x_at_y () [box]* <ontop> () [floor]*',
        ],
        # 36
        # organizing_file_cabinet
        [
            'forall_x_x_at_y () [writing_tool]* <ontop> () [table]*',
            'forall_x_x_at_y () [paper_product]* <inside> () [cabinet]*',
        ],
        # 37
        # organizing_school_stuff
        [
            'exists_x_x_at_y () [paper_product]* <inside> () [backpack]0',
            'exists_x_x_at_y () [writing_tool]0 <inside> () [backpack]0',
            'exists_x_x_at_y () [writing_tool]1 <inside> () [backpack]0',
            'exists_x_x_at_y () [electrical_device]* <inside> () [backpack]0',
        ],
        # 38
        # packing_adult_s_bags
        [
            'forall_x_x_at_y () [decoration]* <inside> () [backpack]0',
            'exists_x_x_at_y () [toiletry]0 <inside> () [backpack]0',
            'exists_x_x_at_y () [toiletry]1 <inside> () [backpack]0',
            'exists_x_x_at_y () [electrical_device]* <inside> () [backpack]0',
        ],
        # 39
        # packing_bags_or_suitcase
        [
            'forall_x_x_at_y () [clothing]* <inside> () [briefcase]0',
            'exists_x_x_at_y () [toiletry]* <inside> () [briefcase]0',
            'exists_x_x_at_y () [cleansing]0 <inside> () [briefcase]0',
            'exists_x_x_at_y () [cleansing]1 <inside> () [briefcase]0',
            'exists_x_x_at_y () [paper_product]* <inside> () [briefcase]0',
        ],
        # 40
        # packing_boxes_for_household_move_or_trip
        [
            'forall_x_x_at_y () [cutlery]* <inside> () [box]0',
            'exists_x_x_at_y () [piece_of_cloth]* <inside> () [box]0',
            'forall_x_x_at_y () [book]* <inside> () [box]1',
            'exists_x_x_at_y () [clothing]* <inside> () [box]1',
        ],
        # pass
        # packing_car_for_trip

        # 41
        # packing_child_s_bag
        [
            'exists_x_x_at_y () [headwear]* <inside> () [backpack]0',
            'exists_x_x_at_y () [decoration]* <inside> () [backpack]0',
            'exists_x_x_at_y () [fruit]* <inside> () [backpack]0',
            'exists_x_x_at_y () [electrical_device]* <inside> () [backpack]0',
            'exists_x_x_at_y () [paper_product]* <inside> () [backpack]0',
        ],
        # 42
        # packing_food_for_work
        [
            'exists_x_x_at_y () [snack]0 <inside> () [box]0',
            'exists_x_x_at_y () [snack]1 <inside> () [box]0',
            'exists_x_x_at_y () [fruit]* <inside> () [box]0',
            'exists_x_x_at_y () [drink]* <inside> () [box]0',
        ],
        # 43
        # packing_lunches
        [
            'exists_x_x_at_y () [snack]0 <inside> () [box]0',
            'exists_x_x_at_y () [snack]0 <inside> () [box]1',
            'exists_x_x_at_y () [baked_food]0 <inside> () [box]0',
            'exists_x_x_at_y () [baked_food]0 <inside> () [box]1',
            'exists_x_x_at_y () [prepared_food]* <inside> () [box]0',
            'exists_x_x_at_y () [drink]* <inside> () [box]0',
            'exists_x_x_at_y () [fruit]* <inside> () [box]0',
            'exists_x_x_at_y () [snack]* <inside> () [box]1',
            'exists_x_x_at_y () [drink]* <inside> () [box]1',
            'exists_x_x_at_y () [fruit]* <inside> () [box]1',
        ],
        # 44
        # packing_picnics
        [
            'exists_x_x_at_y () [snack]0 <inside> () [box]0',
            'exists_x_x_at_y () [snack]1 <inside> () [box]0',
            'exists_x_x_at_y () [fruit]0 <inside> () [box]1',
            'exists_x_x_at_y () [fruit]1 <inside> () [box]1',
            'exists_x_x_at_y () [fruit]2 <inside> () [box]1',
            'exists_x_x_at_y () [drink]0 <inside> () [box]2',
            'exists_x_x_at_y () [drink]1 <inside> () [box]2',
            'exists_x_x_at_y () [drink]2 <inside> () [box]2',
        ],
        # 45
        # picking_up_take-out_food
        [
            'forall_x_x_at_y () [prepared_food]* <inside> () [box]0',
            'forall_x_x_at_y () [snack]* <inside> () [box]0',
            'forall_x_state_x (not_open) [box]* <ontop> () [floor]*'
        ],
        # 46
        # picking_up_trash
        [
            'exists_y_forall_x_x_at_y () [paper_product]* <inside> () [ashcan]*',
            'exists_y_forall_x_x_at_y () [drink]* <inside> () [ashcan]*',
        ],
        # 47
        # polishing_furniture
        [
            'x_state_x (not_dusty) [table]*',
            'x_state_x (not_dusty) [shelf]*',
        ],
        # 48
        # polishing_shoes
        [
            'forall_x_x_at_y (soaked) [rag]* <inside> () [sink]*',
            'forall_x_state_x (not_dusty) [footwear]*',
        ],
        # 49
        # polishing_silver
        [
            'forall_x_x_at_y (not_dusty) [cutlery]* <inside> () [cabinet]*',
            'forall_x_x_at_y () [rag]* <inside> () [cabinet]*',
        ],
        # pass
        # preparing_a_shower_for_child

        # 50
        # preparing_salad
        [
            'exists_x_x_at_y () [vegetable]0 <inside> () [plate]0',
            'exists_x_x_at_y () [vegetable]0 <inside> () [plate]1',
            'exists_x_x_at_y () [vegetable]1 <inside> () [plate]0',
            'exists_x_x_at_y () [vegetable]1 <inside> () [plate]1',
            'exists_x_x_at_y (sliced) [vegetable]2 <inside> () [plate]0',
            'exists_x_x_at_y (sliced) [vegetable]2 <inside> () [plate]1',
            'exists_x_x_at_y (sliced) [fruit]0 <inside> () [plate]0',
            'exists_x_x_at_y (sliced) [fruit]0 <inside> () [plate]1',
        ],
        # 51
        # preserving_food
        [
            'exists_y_forall_x_x_at_y (sliced,cooked) [fruit]* <inside> (not_open) [vessel]*',
            'forall_x_x_at_y (frozen) [protein]* <inside> () [refrigerator]*',
        ],
        # 52
        # putting_away_Christmas_decorations
        [
            'forall_x_x_at_y () [decoration]0 <inside> () [cabinet]*',
            'forall_x_x_at_y () [decoration]1 <inside> () [cabinet]*',
            'forall_x_x_at_y () [decoration]2 <inside> () [cabinet]*',
        ],
        # 53
        # putting_away_Halloween_decorations
        [
            'forall_x_x_at_y () [vegetable]* <inside> () [cabinet]*',
            'forall_x_x_at_y () [illumination_tool]* <inside> () [cabinet]*',
            'forall_x_x_at_y () [vessel]* <ontop> () [table]*',
        ],
        # 54
        # putting_away_toys
        [
            'exists_y_forall_x_x_at_y () [plaything]* <inside> (not_open) [box]*',
        ],
        # 55
        # putting_dishes_away_after_cleaning
        [
            'forall_x_x_at_y () [tableware]* <inside> () [cabinet]*',
        ],
        # 56
        # putting_leftovers_away
        [
            'forall_x_x_at_y () [prepared_food]* <inside> () [refrigerator]*',
            'forall_x_x_at_y () [flavorer]* <inside> () [refrigerator]*',
        ],
        # 57
        # putting_up_Christmas_decorations_inside
        [
            'forall_x_x_at_y () [illumination_tool]* <ontop> () [table]*',
            'forall_x_x_at_y () [decoration]0 <ontop> () [table]*',
            'forall_x_x_at_y () [decoration]1 <ontop> () [table]*',
            'forall_x_x_at_y () [decoration]2 <ontop> () [sofa]*',
        ],
        # 58
        # re-shelving_library_books
        [
            'forall_x_x_at_y () [paper_product]* <ontop> () [shelf]*',
        ],
        # pass
        # rearranging_furniture

        # 59
        # serving_a_meal
        [
            'forall_x_x_at_y () [dish]* <ontop> () [table]*',
            'forall_x_x_at_y () [bowl]* <ontop> () [table]*',
            'forall_x_x_at_y () [cutlery]0 <ontop> () [table]*',
            'forall_x_x_at_y () [cutlery]1 <ontop> () [table]*',
            'forall_x_x_at_y () [drink]* <ontop> () [table]*',
            'exists_x_x_at_y () [protein]* <inside> () [dish]0',
            'exists_x_x_at_y () [protein]* <inside> () [dish]1',
            'exists_x_x_at_y () [prepared_food]* <inside> () [dish]0',
            'exists_x_x_at_y () [prepared_food]* <inside> () [dish]1',
            'exists_x_x_at_y () [baked_food]* <inside> () [dish]0',
            'exists_x_x_at_y () [baked_food]* <inside> () [dish]1',
        ],
        # 60
        # serving_hors_d_oeuvres
        [
            'forall_x_x_at_y () [tray]* <ontop> () [table]*',
            'forall_x_x_at_y () [baked_food]* <ontop> () [table]*',
            'forall_x_x_at_y () [vegetable]* <ontop> () [table]*',
            'forall_x_x_at_y () [prepared_food]* <ontop> () [table]*',
        ],
        # pass
        # setting_mousetraps

        # 61
        # setting_up_candles
        replace_subclass_by_same_type([
            'x_at_y () [illumination_tool]0 <ontop> () [furniture]0',
            'x_at_y () [illumination_tool]1 <ontop> () [furniture]1',
            'x_at_y () [illumination_tool]2 <ontop> () [furniture]2',
        ], subclasses),
        # 62
        # sorting_books
        [
            'forall_x_x_at_y () [paper_product]0 <ontop> () [shelf]*',
            'forall_x_x_at_y () [paper_product]1 <ontop> () [shelf]*',
        ],
        # pass
        # sorting_groceries

        # pass
        # sorting_mail

        # 63
        # storing_food
        [
            'forall_x_x_at_y () [prepared_food]* <inside> () [cabinet]*',
            'forall_x_x_at_y () [snack]* <inside> () [cabinet]*',
            'forall_x_x_at_y () [flavorer]0 <inside> () [cabinet]*',
            'forall_x_x_at_y () [flavorer]1 <inside> () [cabinet]*',
        ],
        # 64
        # storing_the_groceries
        [
            'forall_x_x_at_y () [vegetable]0 <inside> () [refrigerator]*',
            'forall_x_x_at_y () [vegetable]1 <inside> () [refrigerator]*',
            'forall_x_x_at_y () [fruit]* <inside> () [refrigerator]*',
            'forall_x_x_at_y () [protein]* <inside> () [refrigerator]*',
        ],
        # 65
        # thawing_frozen_food
        [
            'forall_x_x_at_y () [vegetable]* <inside> () [sink]*',
            'forall_x_x_at_y () [fruit]* <inside> () [sink]*',
            'forall_x_x_at_y () [protein]* <inside> () [sink]*',
        ],
        # 66
        # throwing_away_leftovers
        [
            'exists_y_forall_x_x_at_y () [snack]* <inside> () [ashcan]*',
        ],
        # pass
        # unpacking_suitcase

        # pass
        # vacuuming_floors

        # pass
        # washing_cars_or_other_vehicles

        # 67
        # washing_dishes
        [
            'forall_x_state_x (not_dusty) [tableware]0',
            'forall_x_state_x (not_dusty) [tableware]1',
            'forall_x_state_x (not_dusty) [tableware]2',
        ],
        # pass
        # washing_floor

        # 68
        # washing_pots_and_pans
        [
            'forall_x_x_at_y (not_dusty) [utensil]* <inside> () [cabinet]*',
            'forall_x_x_at_y (not_dusty) [vessel]0 <inside> () [cabinet]*',
            'forall_x_x_at_y (not_dusty) [vessel]1 <inside> () [cabinet]*',
        ],
        # pass
        # watering_houseplants

        # pass
        # waxing_cars_or_other_vehicles

    ]
    if goal_idx is not None:
        return Goal(behavior_goals[goal_idx], object_dict, subclasses, types)
    else:
        for sub_goals in behavior_goals:
            new_goal = Goal(sub_goals, object_dict, subclasses, types)
            all_goals.append(new_goal)
        return np.random.choice(all_goals)
