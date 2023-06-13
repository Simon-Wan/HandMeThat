import numpy as np
from data_generation.sampling.scene_utils import *


# Generate the scene by sampling locations, receptacles, objects, as well as their positions and states.
def sample_scene(hierarchy, nodes, valid_positions):
    object_dict = dict()
    location_name_dict = sample_locations(hierarchy, nodes, object_dict)
    receptacle_name_dict = sample_receptacles(hierarchy, nodes, object_dict, valid_positions, location_name_dict)
    food_name_dict, tool_name_dict, thing_name_dict = \
        sample_objs(hierarchy, nodes, object_dict, valid_positions, location_name_dict, receptacle_name_dict)
    object_names = {
        'LOCATION': location_name_dict,
        'RECEPTACLE': receptacle_name_dict,
        'FOOD': food_name_dict,
        'TOOL': tool_name_dict,
        'THING': thing_name_dict
    }
    return object_dict, object_names


# Sample the locations.
def sample_locations(hierarchy, nodes, object_dict):
    location_name_dict = dict()
    for subclass in hierarchy['LOCATION']:
        location_name_dict[subclass] = list()
        for loc_type in hierarchy['LOCATION'][subclass]:
            new_obj = nodes[loc_type].copy()
            new_state_dict = sample_other_states('LOCATION', subclass, loc_type, new_obj)
            new_obj['states'] = new_state_dict
            object_dict[loc_type] = new_obj.copy()
            location_name_dict[subclass].append(loc_type)
    return location_name_dict


# Sample the receptacles.
def sample_receptacles(hierarchy, nodes, object_dict, valid_positions, location_name_dict):
    receptacle_name_dict = dict()
    for subclass in hierarchy['RECEPTACLE']:
        receptacle_name_dict[subclass] = dict()

        # sample the receptacle types
        num_of_types = np.random.randint(1, len(hierarchy['RECEPTACLE'][subclass]) + 1)
        # (more types in tableware/bag)
        if subclass in ['tableware', 'bag']:
            num_of_types = np.random.randint(3, len(hierarchy['RECEPTACLE'][subclass]) + 1)
        types = np.random.choice(hierarchy['RECEPTACLE'][subclass], num_of_types, replace=False)

        # sample the receptacles
        for rec_type in types:
            receptacle_name_dict[subclass][rec_type] = list()
            if len(types) == 1:
                num_of_obj = np.random.randint(3, 4)
            else:
                num_of_obj = np.random.randint(1, 4)

            for idx in range(num_of_obj):
                rec_name = '{}#{}'.format(rec_type, idx)
                new_obj = nodes[rec_type].copy()
                prep, position = sample_position(nodes, valid_positions[rec_type], location_name_dict)
                new_obj[prep] = position
                new_state_dict = sample_other_states('RECEPTACLE', subclass, rec_type, new_obj)
                new_obj['states'] = new_state_dict
                object_dict[rec_name] = new_obj.copy()
                receptacle_name_dict[subclass][rec_type].append(rec_name)
    return receptacle_name_dict


# Sample other objects.
def sample_objs(hierarchy, nodes, object_dict, valid_positions, location_name_dict, receptacle_name_dict):
    obj_name_dict = {'FOOD': {}, 'TOOL': {}, 'THING': {}}
    for cls in ['FOOD', 'TOOL', 'THING']:
        for subclass in hierarchy[cls]:
            obj_name_dict[cls][subclass] = dict()

            # sample the object types
            if subclass in ['piece_of_cloth', 'cleaning_tool', 'cutlery']:
                num_of_types = 3
            else:
                num_of_types = np.random.randint(1, len(hierarchy[cls][subclass]) + 1)
            types = np.random.choice(hierarchy[cls][subclass], num_of_types, replace=False)

            # sample the objects
            for obj_type in types:
                obj_name_dict[cls][subclass][obj_type] = list()
                if cls == 'TOOL':
                    if subclass in ['toiletry', 'writing_tool', 'cutlery', 'cleansing', 'electrical_device', 'illumination_tool', 'paper_product']:
                        num_of_obj = np.random.randint(2, 4)
                    else:
                        num_of_obj = np.random.randint(1, 4)
                elif cls == 'FOOD':
                    if num_of_types > 1:
                        num_of_obj = np.random.randint(2, 4)
                    else:
                        num_of_obj = np.random.randint(3, 4)
                else:
                    if num_of_types > 1:
                        num_of_obj = np.random.randint(1, 4)
                    else:
                        num_of_obj = np.random.randint(2, 4)
                for idx in range(num_of_obj):
                    obj_name = '{}#{}'.format(obj_type, idx)
                    new_obj = nodes[obj_type].copy()
                    prep, position = \
                        sample_position(nodes, valid_positions[obj_type], location_name_dict, receptacle_name_dict)
                    new_obj[prep] = position
                    new_state_dict = sample_other_states(cls, subclass, obj_type, new_obj)
                    new_obj['states'] = new_state_dict
                    object_dict[obj_name] = new_obj.copy()
                    obj_name_dict[cls][subclass][obj_type].append(obj_name)
    return obj_name_dict['FOOD'], obj_name_dict['TOOL'], obj_name_dict['THING']


# Sample the initial state of each object.
def sample_other_states(cls, subclass, obj_type, new_obj):
    state_dict = dict()
    if 'ability' not in new_obj.keys():
        return state_dict
    # open
    if 'openable' in new_obj['ability']:
        state_dict['open'] = False
    # cooked
    if 'cookable' in new_obj['ability']:
        if 'inside' in new_obj.keys():
            if new_obj['inside'] in ['stove', 'oven', 'microwave']:
                state_dict['cooked'] = True
            else:
                state_dict['cooked'] = False
        else:
            state_dict['cooked'] = False
    # frozen
    if 'freezable' in new_obj['ability']:
        if 'inside' in new_obj.keys():
            if new_obj['inside'] in ['refrigerator']:
                state_dict['frozen'] = True
            else:
                state_dict['frozen'] = False
        else:
            state_dict['frozen'] = False
    # sliced
    if 'sliceable' in new_obj['ability']:
        state_dict['sliced'] = False
    # soaked
    if 'soakable' in new_obj['ability']:
        state_dict['soaked'] = False
    # toggled
    if 'toggleable' in new_obj['ability']:
        if obj_type == 'refrigerator':
            state_dict['toggled'] = True
        else:
            state_dict['toggled'] = False
    # dusty
    if 'dustyable' in new_obj['ability']:
        dusty = np.random.choice([True, False], p=[1/3, 2/3])
        if dusty:
            state_dict['dusty'] = True
        else:
            state_dict['dusty'] = False
    # stained
    if 'stainable' in new_obj['ability']:
        dusty = np.random.choice([True, False], p=[1/3, 2/3])
        if dusty:
            state_dict['stained'] = True
        else:
            state_dict['stained'] = False
    # color
    if 'has-color' in new_obj['ability']:
        color = np.random.choice(['red', 'green', 'blue'])
        state_dict['color'] = color
    # size
    if 'has-size' in new_obj['ability']:
        size = np.random.choice(['large', 'small'])
        state_dict['size'] = size
    return state_dict


# Sample the initial position of each movable object.
def sample_position(nodes, obj_valid_positions, location_name_dict, receptacle_name_dict=None):
    all_possibles = list()
    for name in obj_valid_positions:
        if name in location_name_dict['inside']:
            all_possibles.append('inside ' + name)
        if name in location_name_dict['ontop']:
            all_possibles.append('ontop ' + name)
        if receptacle_name_dict:
            for subclass in receptacle_name_dict:
                for rec_name in receptacle_name_dict[subclass].keys():
                    if rec_name == name:
                        if 'has-inside' in nodes[rec_name]['ability']:
                            prep = 'inside'
                        else:
                            prep = 'ontop'
                        all_possibles.append(prep + ' ' + np.random.choice(receptacle_name_dict[subclass][rec_name]))
    sample = np.random.choice(all_possibles)
    sample = sample.split()
    return sample[0], sample[1]     # prep, position


if __name__ == '__main__':
    # Try scene sampling.
    hierarchy, nodes = hierarchy2json('object_hierarchy', True)
    valid_positions = get_valid_positions(nodes, 'object_sample_space')
    object_dict, object_names = sample_scene(hierarchy, nodes, valid_positions)
    print("finish sampling")
    # print(object_dict)
    # print(object_names)

