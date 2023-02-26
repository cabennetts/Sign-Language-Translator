import os
import time

UserVideo = ["UserVideo"]

#change parent_dir to os.getcwd()
parent_dir = r"C:/Users/Alex Anderson/Documents/EECS_581/Sign-Language-Translator-main/Sign-Language-Translator/RecordData/gen2-record-replay"
for Gloss in UserVideo:
    glossPath = os.path.join(parent_dir, Gloss)
    #print(glossPath)
    try:
        os.mkdir(glossPath)
    except:
        print("Folder for Gloss Already Exists, moving on")
    #os.chdir(glossPath)
    fileNum = 1
    while True:
        videoPath = os.path.join(glossPath, str(fileNum))
        if(os.path.exists(videoPath)):
            fileNum+=1
        else:
            break
    

    videoPath = os.path.join(glossPath, str(fileNum))
    os.mkdir(videoPath)
    print("recording user video in 3 seconds...")
    time.sleep(1)
    print("recording user video in 2 seconds...")
    time.sleep(1)
    print("recording user video in 1 second...")
    time.sleep(1)
    #record the video
    os.system('python record.py -p ' + "\"" + videoPath + "\"")
    print("done recording user video")
    #This function takes in length of the segments in seconds(a frame is about 0.04 seconds)
    def sliceVideo(seconds):    
        os.system("ffmpeg -i color.mp4 -c copy -map 0 -segment_time 00:00:" + seconds + " -f segment -reset_timestamps 1 output%03d.mp4")
    #put the video in the new folder
    os.chdir(parent_dir)
    #change the parameters to alter the length of each video segment.
    #0.64 corresponds to about 16 frames 
    sliceVideo(0.64)
