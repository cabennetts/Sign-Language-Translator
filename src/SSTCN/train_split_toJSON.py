import json

data = {
    'test_count' : 3,
    'test_file_name' : ['4', 'A', 'E'],
    'test_label' : [0, 1, 2],
    'train_count' : 3,
    'train_file_name' : ['4', 'A', 'E'],
    'train_label' : [0, 1, 2],
}

with open('train_split.json', 'w+') as outfile:
    json.dump(data, outfile)
    outfile.flush()
