import os
import shutil

import cv2


def split_video_frames(video_path):
    # remove the file from the end of the path
    folder_name = os.path.dirname(video_path)
    try:
        os.makedirs(folder_name + "/set_0" , exist_ok=True)
    except Exception as e:
        print (e)
    print(folder_name + "/set_0")
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    total_sets = 0
    frame_count = 0
    frame_list = []
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        # Save the frame as a .jpg file
        #frame_path = os.path.join(folder_name + "/set_" + str(total_sets), f"{frame_count}.jpg")
        #cv2.imwrite(frame_path, frame)
        
        # Add the frame to the list
        frame_list.append(frame)
        
        # If we have 16 frames, save them to a new .mp4 file
        if len(frame_list) == 52:
            save_frames_to_mp4(frame_list, folder_name + "/set_" + str(total_sets))
            frame_list = []
            total_sets += 1
            os.makedirs(folder_name + "/set_" + str(total_sets) , exist_ok=True)
            
        frame_count += 1
    
    # NOTE: I'm throwing out the remaining ending frames
    #if len(frame_list) > 0:
    #    save_frames_to_mp4(frame_list, folder_name)
    # check if number of files in folder is 16, if not delete the folder
    if len(os.listdir(folder_name + "/set_" + str(total_sets))) < 16:
        shutil.rmtree(folder_name + "/set_" + str(total_sets))

    # Release the video file
    cap.release()

def save_frames_to_mp4(frame_list, folder_name):
    # Get the dimensions of the frames
    height, width, _ = frame_list[0].shape
    
    # Create a VideoWriter object to save the frames to a new .mp4 file
    video_path = os.path.join(folder_name, f"{len(os.listdir(folder_name))}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    
    # Write the frames to the new .mp4 file
    for frame in frame_list:
        out.write(frame)
    
    # Release the VideoWriter object
    out.release()

def run_sv(vid_path = None):
    # Replace this with the path to your .mp4 file
    video_path = vid_path or "D:/output/we-go-school-tomorrow/we-go-school-tomorrow.mp4"
    # video_path = vid_path 
    split_video_frames(video_path)