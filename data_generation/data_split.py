import json
import os
import os.path as osp
import random

import numpy as np


def create_split():
    type_split = {
        'bring_me': list(),
        'move_to': list(),
        'change_state': list(),
    }
    level_split = {
        'level1': list(),
        'level2': list(),
        'level3': list(),
        'level4': list(),
    }
    goal_split = dict()
    for idx in range(69):
        goal_split[str(idx)] = list()

    train_split = {
        'train': list(),
        'validate': list(),
        'test': list()
    }
    return type_split, level_split, goal_split, train_split


def add_file(type_split, level_split, goal_split, train_split, filename):
    quest_type, goal_idx, task_idx, hardness_level = filename[5:-5].split('-')
    type_split[quest_type].append(filename)
    goal_split[goal_idx].append(filename)
    if hardness_level == '1':
        level_split['level1'].append(filename)
    elif hardness_level == '2':
        level_split['level2'].append(filename)
    elif hardness_level == '3':
        level_split['level3'].append(filename)
    elif hardness_level == '0' or hardness_level == '4':
        level_split['level4'].append(filename)
    rd = random.uniform(0, 1)
    if rd < 0.9:
        train_split['train'].append(filename)
    elif rd > 0.95:
        train_split['test'].append(filename)
    else:
        train_split['validate'].append(filename)


def print_count(type_split, level_split, goal_split, train_split):
    print('bring_me: {}, move_to: {}, change_state: {}'.format(
        len(type_split['bring_me']), len(type_split['move_to']), len(type_split['change_state'])
    ))
    print('level1: {}, level2: {}, level3: {}, level4: {}'.format(
        len(level_split['level1']), len(level_split['level2']), len(level_split['level3']), len(level_split['level4'])
    ))
    print('goal:')
    print([len(goal_split[str(idx)]) for idx in range(69)])
    print('train: {}, validate: {}, test: {}'.format(
        len(train_split['train']), len(train_split['validate']), len(train_split['test']),
    ))


if __name__ == '__main__':
    type_split, level_split, goal_split, train_split = create_split()
    data_info = './datasets/v2/HandMeThat_data_info.json'
    data_dir = './datasets/v2/HandMeThat_with_expert_demonstration'
    if not osp.exists(data_dir):
        raise ValueError('Expert demonstration not exists!')
    pap_tasks = [2, 3, 24, 25, 26, 30, 31, 32, 33, 35, 36, 46, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66]
    counter = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            name = file.split('-')
            counter += 1
            # task - quest type - goal idx - task idx - hardness level
            add_file(type_split, level_split, goal_split, train_split, file)

    print_count(type_split, level_split, goal_split, train_split)

    with open(data_info, 'w+') as f:
        json.dump([type_split, level_split, goal_split, train_split], f)

