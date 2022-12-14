
import csv
import os
import random

path_to_letters = "C:/Users/bencl/Desktop/Senior Fall Semester/EECS_581/ASL Project/archive (2)//SigNN Character Database"
path_to_j_z = "C:/Users/bencl/Desktop/Senior Fall Semester/EECS_581/ASL Project/archive (1)/SigNN Video Data"
path_to_numbers = "C:/Users/bencl/Desktop/Senior Fall Semester/EECS_581/ASL Project/Sign-Language-Digits-Dataset-master/Dataset"
test_folder = "C:/Users/bencl/Desktop/data/Test"
train_folder = "C:/Users/bencl/Desktop/data/Train"
#path_to_numbers = "C:/Users/bencl/Desktop/Senior Fall Semester/EECS_581/ASL Project"
paths = [path_to_letters, path_to_j_z, path_to_numbers]

def generate_csv():
    #open csv file
    test_data = []
    with open('training_data.csv', 'w', newline='') as csvfile:
        #create csv writer
        writer = csv.writer(csvfile, delimiter=',')
        #write header
        writer.writerow(['video_name', 'tag'])
        #iterate through all folders
        for path in paths:
            for folder in os.listdir(path):
                num_files = len(os.listdir(path + "/" + folder))
                #generate five random numbers
                test_indexes = []
                for i in range(5):
                    test_indexes.append(random.randint(0, num_files - 1))

                #iterate through all files in folder
                for i, filename in enumerate(os.listdir(path + "/" + folder)):
                    #create a string conatining the last three folders and the file name
                    #video_name = path.split("/")[-2] + "/" + path.split("/")[-1] + "/" + folder + "/" + filename
                    #write row to csv
                    if i in test_indexes:
                        os.rename(path + "/" + folder + "/" + filename, test_folder + "/" + folder + str(i) + "." + filename.split(".")[-1])
                        test_data.append([folder + str(i) + "." + filename.split(".")[-1], folder])
                    else:
                        #move the file to C:\Users\bencl\Desktop\Data\Train
                        os.rename(path + "/" + folder + "/" + filename, train_folder + "/" + folder + str(i) + "." + filename.split(".")[-1])
                        writer.writerow([folder + str(i) + "." + filename.split(".")[-1], folder])
    #write test data to csv
    with open('test_data.csv', 'w', newline='') as csvfile:
        #create csv writer
        writer = csv.writer(csvfile, delimiter=',')
        #write header
        writer.writerow(['video_name', 'tag'])
        #write rows
        for row in test_data:
            writer.writerow(row)
    
generate_csv()