import os
import time
import string

ListOfSigns = list(string.ascii_uppercase)
parent_dir = os.getcwd()
for Gloss in ListOfSigns:
    glossPath = os.path.join(parent_dir, Gloss)
    #print(glossPath)
    try:
        os.mkdir(glossPath)
    except:
        print("Folder for Gloss Already Exists, moving on")
    os.chdir(glossPath)
    fileNum = 1
    while True:
        videoPath = os.path.join(glossPath, str(fileNum))
        if(os.path.exists(videoPath)):
            fileNum+=1
        else:
            break
    
    for iteration in range(1,5):
        videoPath = os.path.join(glossPath, str(fileNum+iteration))
        os.mkdir(videoPath)
        print("recording " + Gloss + " in 2 sec")
        time.sleep(2)
        print("recording video #" + str(iteration) + " in 2 sec")
        #record the video
        os.system('python3 record.py -p '+ videoPath)
        print("done recording video #"+ str(iteration)+ " of "+ Gloss)
    print("")
    #put the video in the new folder
    os.chdir(parent_dir)
    

    
    

    
    
    