import json
import csv

#Format of the json file
'''
data = {
    'test_count' : 3,
    'test_file_name' : ['4', 'A', 'E'],
    'test_label' : [0, 1, 2],
    'train_count' : 3,
    'train_file_name' : ['4', 'A', 'E'],
    'train_label' : [0, 1, 2],
}
'''

def read_labels_csv(train_labels_path, test_labels_path):
    train_filenames = []
    train_values = []
    test_filenames = []
    test_values = []

    # Read train labels
    with open(train_labels_path, 'r') as train_labels_file:
        csv_reader = csv.reader(train_labels_file)
        next(csv_reader) # skip header row
        for row in csv_reader:
            train_filenames.append(row[0])
            train_values.append(row[1])

    # Read test labels
    with open(test_labels_path, 'r') as test_labels_file:
        csv_reader = csv.reader(test_labels_file)
        next(csv_reader) # skip header row
        for row in csv_reader:
            test_filenames.append(row[0])
            test_values.append(row[1])

    return train_filenames, train_values, test_filenames, test_values, len(train_filenames), len(test_filenames)

def get_data(train_filenames, train_values, test_filenames, test_values, num_train_files, num_test_files):
    data = {}
    data['test_count'] = num_test_files
    data['test_file_name'] = test_filenames
    data['test_label'] = test_values
    data['train_count'] = num_train_files
    data['train_file_name'] = train_filenames
    data['train_label'] = train_values
    return data

if __name__ == '__main__':
    train_labels_path = 'E:/ASL_Data/train_labels.csv'
    test_labels_path = 'E:/ASL_Data/test_labels.csv'

    train_filenames, train_values, test_filenames, test_values, num_train_files, num_test_files = read_labels_csv(train_labels_path, test_labels_path)
    data = get_data(train_filenames, train_values, test_filenames, test_values, num_train_files, num_test_files)
    #print(data)
    with open('train_split.json', 'w+') as outfile:
        
        json.dump(data, outfile)
        outfile.flush()
