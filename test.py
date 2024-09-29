import os
import json

dir1 = '/Users/lly/Downloads/img_data-1/'

for file in os.listdir(dir1):
    if file.endswith('.json'):
        json_path = os.path.join(dir1, file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        shape_copy = data['shapes'].copy()
        data['shapes'] = []
        new_shape = []
        for idx, shape in enumerate(shape_copy):
            if shape['shape_type'] == 'rectangle':
                shape['label'] = str(idx)
                new_shape.append(shape)
        
        data['shapes'] = new_shape
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
