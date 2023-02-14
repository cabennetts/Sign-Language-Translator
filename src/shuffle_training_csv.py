import random
import os

#loop through lines of provided csv dile, and randomly shuffle them
def shuffle_csv(csv_file):
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(csv_file, 'w') as f:
        f.writelines(lines)

#print current directory
print(os.getcwd())

shuffle_csv("E:/ASL_Data/train_labels.csv")