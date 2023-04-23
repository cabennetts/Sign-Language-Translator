'''
This file contains all the code needed to go from a newly recorded video
to a prediction of the sign language word being signed.

The file must be in the ensemble folder, it depends on the contained code.
'''

# This path should point to a folder containing only one mp4 video
# and a lot of files will be saved here as well
path_to_video = '/Users/cabennetts/Documents/GitHub/Sign-Language-Translator/Backend/controllers/files/'

import os
import shutil
import sys

# change the current working directory to the directory of this file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from collections import OrderedDict

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Python sucks so I have to do this to import from other folders
sys.path.append(os.path.abspath('DataPreparation'))
sys.path.append(os.path.abspath('models'))
sys.path.append(os.path.abspath('FeatureExtraction'))
sys.path.append(os.path.abspath('SL-GCN/data_gen/'))
import decouple_gcn_attn
import demo
import gen_bone_data
import gen_frames
import gen_motion_data
import sign_27
import sign_gendata
import split_video
import wholepose_features_extraction
from T_Pose_model import T_Pose_model

import SentenceShuffler
# Looks like lots of errors, but it's just because I'm importing from other folders
from Conv3D import r2plus1d_18

# The following functions are used to prepare the video and extract the
# data we will need for the predictions.
# NOTE: These functions are in the order they should be called

# This function calls the split_video function to split the video into 
# one video for every 56 frames, and creates the folders that will contain
# the data needed for each 56 frame 'set'
def video_to_sets():
    #loop through files in path_to_video searching for .mp4 files
    for file in sorted(os.listdir(path_to_video)):
        if file.endswith(".mp4"):
            #split the video into sets of 56 frames
            split_video.run_sv(path_to_video + file)

# This function runs demo.py to extract the keypoints from each set.
# It then saves the keypoints as a .npy file in a /npy/ folder in each set folder
def generate_npy_keypoints():
    for folder in sorted(os.listdir(path_to_video)):
        if os.path.isdir(path_to_video + folder):
            os.mkdir(path_to_video + folder + '/npy/')
            demo.run(path_to_video + folder, path_to_video + folder + '/npy/')

# This function runs gen_frames.py to extract the frames from each set
# and saves them in the /frames/ folder in each set folder.
def extract_frames():
    for folder in sorted(os.listdir(path_to_video)):
        if os.path.isdir(path_to_video + folder):
            os.mkdir(path_to_video + folder + '/frames/')
            gen_frames.run(path_to_video + folder, path_to_video + folder + '/frames/', path_to_video + folder + '/npy/')

# This function runs wholepose_features_extraction.py to extract the features
# from each set and saves them in the /pt/ folder in each set folder.
def generate_pt_features():
    for folder in sorted(os.listdir(path_to_video)):
        if os.path.isdir(path_to_video + folder):
            os.mkdir(path_to_video + folder + '/pt/')
            wholepose_features_extraction.run(path_to_video + folder, path_to_video + folder + '/pt/', False)

# This function runs sign_gendata.py to generate the data needed for the
# joint GCN model and saves it in the /sign_gen/ folder in each set folder.
def generate_sign_gcn_data():
    for folder in sorted(os.listdir(path_to_video)):
        if os.path.isdir(path_to_video + folder):
            os.mkdir(path_to_video + folder + '/sign_gen/')
            sign_gendata.run(path_to_video + folder + '/npy/', path_to_video + folder + '/sign_gen/')

# This function runs gen_bone_data.py to generate the data needed for the
# bone GCN model and saves it in the /sign_gen/ folder in each set folder.
def generate_bone_gcn_data():
    for folder in sorted(os.listdir(path_to_video)):
        if os.path.isdir(path_to_video + folder):
            gen_bone_data.run(path_to_video + folder + '/sign_gen/')

# This function runs gen_motion_data.py to generate the data needed for the joint and
# bone motion GCN models and saves it in the /sign_gen/ folder in each set folder.
def generate_motion_gcn_data():
    for folder in sorted(os.listdir(path_to_video)):
        if os.path.isdir(path_to_video + folder):
            gen_motion_data.run(path_to_video + folder + '/sign_gen/')

# This function is given a checkpoint and removes the 'module.' from
# the beginning of each key in the state dict so they can be loaded.
def generate_state_dict(model):
    state_dict = OrderedDict()
    for k, v in model.items():
        #name = k[7:] # remove 'module.'
        name = k.replace('module.', '')
        state_dict[name]=v
    return state_dict


# Globals for predictions

# transform used on images for 3D CNN
transform = transforms.Compose([transforms.Resize([240, 240]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

# Labels to map class integers to words
class_labels = ["car", "go", "have", "hello", "my", "name", "school", "self", "tomorrow", "we", "what", "yesterday", "you"]

# Load checkpoints for each model
rgb_model = generate_state_dict(torch.load('sign_resnet2d+1_5_epoch009.pth', map_location=torch.device('cpu')))
joint_model = generate_state_dict(torch.load('sign_joint_final-24-95.pt', map_location=torch.device('cpu')))
joint_motion_model = generate_state_dict(torch.load('sign_joint_motion_final-32-86.pt', map_location=torch.device('cpu')))
bone_model = generate_state_dict(torch.load('sign_bone_final-25.pt', map_location=torch.device('cpu')))
bone_motion_model = generate_state_dict(torch.load('sign_bone_motion_final-25.pt', map_location=torch.device('cpu')))
tcn_model = generate_state_dict(torch.load('T_Pose_model_16_99.0.pth', map_location=torch.device('cpu')))


# The following functions are used to make the predictions

def rgb_cnn_prediction():

    # Load the checkpoint for the 3D CNN
    model = r2plus1d_18(pretrained=True, num_classes=13)
    model.load_state_dict(rgb_model)
    model.eval()
    rgb_cnn_predictions = []

    # Loop through each set and make predictions
    for folder in sorted(os.listdir(path_to_video)):
        if os.path.isdir(path_to_video + folder + '/frames/0'):
            print(folder)
            images = []
            input_clips = []
            for i, file in enumerate(sorted(os.listdir(path_to_video + folder + '/frames/0'))):
                if i < 4:
                    continue
                image = Image.open(path_to_video + folder + '/frames/0/' + file)
                image = transform(image)
                images.append(image)
                if len(images) == 16:
                    images = torch.stack(images, dim=0)
                    images = images.permute(1, 0, 2, 3)
                    images = torch.Tensor(images)
                    images = images.unsqueeze(0)
                    input_clips.append(images)
                    images = []

            # Make predictions on each set and store them in a list
            set_predictions = []
            for set in input_clips:
                top_five = []
                output = None
                with torch.no_grad():
                    output = model(set)
                # Convert the predictions to probabilities using softmax
                probs = torch.nn.functional.softmax(output, dim=1)

                # Append top k probabilities and their corresponding class labels
                for i in range(len(probs[0])):
                    top_five.append((class_labels[i], (probs[0][i]*100).item()))
                set_predictions.append(top_five)
         # returns array in form [[[(class, prob), (class, prob), (class, prob), (class, prob), (class_prob)],
            #                      [(class, prob), (class, prob), (class, prob), (class, prob), (class_prob)],
            #                      [(class, prob), (class, prob), (class, prob), (class, prob), (class_prob)]],
            #
            #                     [[(class, prob), (class, prob), (class, prob), (class, prob), (class_prob)],
            #                      [(class, prob), (class, prob), (class, prob), (class, prob), (class_prob)],
            # etc. Where the inner most arrays are the top 5 predictions, the second most inner arrays are each 
            # set of 16 frames, and the outer most arrays are the predictions for each set
            rgb_cnn_predictions.append(set_predictions)

    return rgb_cnn_predictions

def gcn_predictions():

    # Load model architechure
    Model_j = decouple_gcn_attn.Model(13, 27, 1, 16, 41,  "sign_27.Graph", {"labeling_mode": 'spatial'}, 3)
    Model_b = decouple_gcn_attn.Model(13, 27, 1, 16, 41, "sign_27.Graph", {"labeling_mode": 'spatial'}, 3)
    Model_jm = decouple_gcn_attn.Model(13, 27, 1, 16, 41, "sign_27.Graph", {"labeling_mode": 'spatial'}, 3)
    Model_bm = decouple_gcn_attn.Model(13, 27, 1, 16, 41, "sign_27.Graph", {"labeling_mode": 'spatial'}, 3)

    # Load model states from checkpoints
    Model_j.load_state_dict(joint_model)
    Model_b.load_state_dict(bone_model)
    Model_jm.load_state_dict(joint_motion_model)
    Model_bm.load_state_dict(bone_motion_model)

    # Set the model to evaluation mode
    Model_j.eval()
    Model_b.eval()
    Model_jm.eval()
    Model_bm.eval()

    joint_predictions = []
    bone_predictions = []
    joint_motion_predictions = []
    bone_motion_predictions = []

    for folder in sorted(os.listdir(path_to_video)):
        if os.path.isdir(path_to_video + folder + '/sign_gen/'):
            bone_npy = np.load(path_to_video + folder + '/sign_gen/test_data_bone.npy')
            joint_npy = np.load(path_to_video + folder + '/sign_gen/test_data_joint.npy')
            bone_motion_npy = np.load(path_to_video + folder + '/sign_gen/test_data_bone_motion.npy')
            joint_motion_npy = np.load(path_to_video + folder + '/sign_gen/test_data_joint_motion.npy')

            # Load the data onto the GPU if available
            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device("cpu")
            bone_npy = torch.from_numpy(bone_npy).to(device)
            joint_npy = torch.from_numpy(joint_npy).to(device)
            bone_motion_npy = torch.from_numpy(bone_motion_npy).to(device)
            joint_motion_npy = torch.from_numpy(joint_motion_npy).to(device)

            # Make predictions using the four models
            with torch.no_grad():
                joint_output = Model_j(joint_npy)
                bone_output = Model_b(bone_npy)
                joint_motion_output = Model_jm(joint_motion_npy)
                bone_motion_output = Model_bm(bone_motion_npy)

            # Print the top 5 predictions and their confidence percentages for each model
            def get_top_five(output):
                predictions = []
                probabilities = torch.nn.functional.softmax(output, dim=1)
                for i in range(len(probabilities[0])):
                    predictions.append((class_labels[i], (probabilities[0][i]*100).item()))
                return predictions
            
            joint_predictions.append(get_top_five(joint_output))
            bone_predictions.append(get_top_five(bone_output))
            joint_motion_predictions.append(get_top_five(joint_motion_output))
            bone_motion_predictions.append(get_top_five(bone_motion_output))
    
    # returns four arrays in form [[(class, prob), (class, prob), ...], [(class, prob), (class, prob), ...], ...]
    # where each inner array contains the top 5 predictions for the model on that set
    return joint_predictions, bone_predictions, joint_motion_predictions, bone_motion_predictions

def tcn_predictions():
    device = torch.device("cpu")
    model = T_Pose_model(frames_number=60,joints_number=33, n_classes=13)
    model = model.to(device)

    # Add weights from checkpoint model
    model.load_state_dict(tcn_model)
    model.eval()

    tcn_predictions = []

    for folder in sorted(os.listdir(path_to_video)):
        if os.path.isdir(path_to_video + folder + '/pt/'):
            pt_file = path_to_video + folder + '/pt/0.mp4.pt'
            data = torch.load(pt_file)
            #data = data.contiguous().view(1,-1,24,24)
            data_in = torch.autograd.Variable(data.to(device), requires_grad=False)
            with torch.no_grad():
                pred=model(data_in)
            #pred = pred.cpu().detach().numpy()
            predictions = []
            def get_top_five(output):
                probabilities = torch.nn.functional.softmax(output, dim=1)
                for i in range(len(probabilities[0])):
                    predictions.append((class_labels[i], (probabilities[0][i]*100).item()))
                return predictions
            tcn_predictions.append(get_top_five(pred))
    
    # returns array in form [[(class, prob), (class, prob), ...], [(class, prob), (class, prob), ...], ...]
    # where each inner array contains the top 5 predictions for the model on that set
    return tcn_predictions




def remove_files_in_dir(path):
    """
    This function removes all files and folders within a directory
    """
    # Check if path exists
    if not os.path.exists(path):
        print(f"{path} does not exist")
        return
    
    # Remove files in the directory
    for filename in sorted(os.listdir(path)):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    print(f"All files and folders in {path} have been removed.")
    

# run through every step defined above 
def run_ensemble():
    # Run data preprocessing
    print("Starting Processing")
    
    video_to_sets()
    generate_npy_keypoints()
    extract_frames()
    generate_pt_features()
    generate_sign_gcn_data()
    generate_bone_gcn_data()
    generate_motion_gcn_data()
    
    print("Data Preprocessing Complete")
    print("Starting Predictions")

    cnn_preds = rgb_cnn_prediction()
    tcn_preds = tcn_predictions()
    joint_preds, bone_preds, joint_motion_preds, bone_motion_preds = gcn_predictions()

    final_preds = []
    for set_i in range(len(cnn_preds)):
        averaged_preds = [0 for i in range(len(cnn_preds[0][0]))]
        for pred_j in range(len(cnn_preds[0][0])):
            averaged_preds[pred_j] += cnn_preds[set_i][0][pred_j][1]
            averaged_preds[pred_j] += cnn_preds[set_i][1][pred_j][1]
            averaged_preds[pred_j] += cnn_preds[set_i][2][pred_j][1]
            averaged_preds[pred_j] += tcn_preds[set_i][pred_j][1]
            averaged_preds[pred_j] += joint_preds[set_i][pred_j][1]
            averaged_preds[pred_j] += bone_preds[set_i][pred_j][1]
            averaged_preds[pred_j] += joint_motion_preds[set_i][pred_j][1]
            averaged_preds[pred_j] += bone_motion_preds[set_i][pred_j][1]
            averaged_preds[pred_j] /= 8
        print("Set " + str(set_i) + " Averaged Predictions: ")
        for i, pred in enumerate(averaged_preds):
            print(class_labels[i] + ": " + str(pred))
        final_preds.append(max(zip(averaged_preds, class_labels)))
        print('Final Preds:', final_preds)

    print("Final Predictions: ")
    # object to return results
    import json
    result = ""
    data =  { "result": "",
              "words": {},
              "english": "",
            }  

    for i, pred in enumerate(final_preds):
        result += pred[1] + " "
        data['words'][pred[1]] = pred[0]
    
    result = result.replace("self", "I")

    # convert result to proper english
    english = SentenceShuffler.convertToEnglish(result)

    data['result'] = result
    data['english'] = english
    json_object = json.dumps(data)
    print(data)
    print(json_object)
    # clear contents in upload folder
    remove_files_in_dir(path_to_video)
    return json_object
 
if __name__ == "__main__":
    print("Starting Program")
    run_ensemble()