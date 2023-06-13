import json


# Parse the ./object_hierarchy file.
def hierarchy2json(filename, save_json=False):
    classes = ['LOCATION', 'RECEPTACLE', 'FOOD', 'TOOL', 'THING']       # change the class names if needed
    hierarchy = dict()
    nodes = dict()
    for cls in classes:
        hierarchy[cls] = dict()

    current_class = None
    with open(filename) as f:
        start_states_info = False
        for line in f.readlines():
            words = line.split()
            if not words:
                continue
            if words[0] == '---END---':
                break
            if words[0][0] == '_':
                current_class = None
            if current_class:
                obj_type = words[0]
                subclass = words[1]
                if subclass in hierarchy[current_class].keys():
                    hierarchy[current_class][subclass].append(obj_type)
                else:
                    hierarchy[current_class][subclass] = [obj_type]
                nodes[obj_type] = {'class': current_class, 'subclass': subclass, 'type': obj_type, 'ability': []}
            elif start_states_info:
                static_state = words[0]
                for word in words[1:]:
                    if word in classes:
                        for subclass in hierarchy[word].keys():
                            for obj_type in hierarchy[word][subclass]:
                                nodes[obj_type]['ability'].append(static_state)
                    else:
                        tmp = True
                        for cls in classes:
                            if word in hierarchy[cls].keys():
                                for obj_type in hierarchy[cls][word]:
                                    nodes[obj_type]['ability'].append(static_state)
                                tmp = False
                                break
                        if tmp:
                            nodes[word]['ability'].append(static_state)
            if words[0] in classes:
                current_class = words[0]
            elif words[0] == 'STATIC':
                start_states_info = True
    if save_json:
        with open('./hierarchy.json', 'w+') as js1:
            json.dump(hierarchy, js1)
        with open('./nodes.json', 'w+') as js2:
            json.dump(nodes, js2)
    return hierarchy, nodes


# Parse the ./object_sample_space file.
def get_valid_positions(nodes, filename):
    valid_positions = dict()
    with open(filename, 'r') as f:
        start_reading = False
        for line in f.readlines():
            words = line.split()
            if not words:
                continue
            if words[0][0] == '_':
                start_reading = None
            if start_reading:
                valid_positions[words[0]] = words[1:]
            if words[0] == 'POSITION':
                start_reading = True
        for node in nodes.keys():
            if node not in valid_positions.keys() and nodes[node]['class'] != 'LOCATION':
                if nodes[node]['class'] in valid_positions.keys():
                    valid_positions[node] = valid_positions[nodes[node]['class']]
                if nodes[node]['subclass'] in valid_positions.keys():
                    valid_positions[node] = valid_positions[nodes[node]['subclass']]
    return valid_positions


