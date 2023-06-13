import pdsketch as pds
import random


def solve_subgoal(expr, state, extended_state, strips_operators, translator):
    strips_goal = translator.compile_expr(expr, state)
    strips_goal = strips_goal[0]

    cur_task = pds.strips.GStripsTask(extended_state, strips_goal, strips_operators, is_relaxed=False)
    cur_task = translator.relevance_analysis(cur_task)
    cur_task = cur_task.compile()

    solvable = True
    if 'classifier' in cur_task.goal.__dict__.keys():
        if not cur_task.goal.is_disjunction:
            predicates = list(cur_task.goal.classifier)
            tmp_dict = dict()
            for pred in predicates:
                if pred.split()[0] in ['inside', 'ontop']:
                    if pred.split()[1] in tmp_dict.keys():
                        if tmp_dict[pred.split()[1]] != pred.split()[2]:
                            solvable = False
                    tmp_dict[pred.split()[1]] = pred.split()[2]

    if not solvable:
        return None, None

    if not cur_task.operators:
        return None, None

    heuristic = pds.strips.StripsHFFHeuristic(cur_task, translator)     # TODO
    try:
        plan = pds.strips.strips_heuristic_search(cur_task, heuristic, verbose=False, heuristic_weight=0.7)     # TODO
    except RuntimeError as err:
        plan = None

    if plan:
        for simplified_op in plan:
            extended_op = None
            for op in strips_operators:
                if simplified_op.raw_operator.name == op.raw_operator.name and simplified_op.raw_operator.arguments == op.raw_operator.arguments:
                    extended_op = op
                    break
            extended_state = extended_op.apply(extended_state)

    return plan, extended_state


def solve_goal(cur_goal, state, extended_state, strips_operators, translator):
    plan = list()
    sub_plans = list()
    random.shuffle(cur_goal.expr_list)

    for expr in cur_goal.expr_list:
        sub_plan, extended_state = solve_subgoal(expr, state, extended_state, strips_operators, translator)
        if sub_plan is None:
            plan = None
            break
        else:
            plan += sub_plan
            sub_plans.append(sub_plan)
    return plan, extended_state, sub_plans


def get_subgoal(strips_subgoals, cur_extended_state, meaning_op, strips_operators, translator, subgoal_steps):
    maximum = 0
    subgoal_idx = -1
    for idx, subgoal in enumerate(strips_subgoals):
        new_task = pds.strips.GStripsTask(meaning_op.apply(cur_extended_state), subgoal, strips_operators, is_relaxed=False)
        new_task = translator.relevance_analysis(new_task)
        new_task = new_task.compile()
        new_heuristic = pds.strips.StripsHFFHeuristic(new_task, translator)
        new_plan = pds.strips.strips_heuristic_search(new_task, new_heuristic, verbose=False, heuristic_weight=0.7)
        shorten = subgoal_steps[idx] - len(new_plan)
        if shorten > maximum:
            maximum = shorten
            subgoal_idx = idx
    return subgoal_idx

