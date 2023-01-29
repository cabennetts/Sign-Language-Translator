import cv2
import mediapipe as mp
import time

inFile = '/dev/video0'

capture = cv2.VideoCapture(inFile)
FramesVideo = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) # Number of frames inside video
FrameCount = 0 # Currently playing frame
prevTime = 0

# some objects for mediapipe
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

while True:
    FrameCount += 1
    #read image and convert to rgb
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #process image
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        #get landmark positions
        landmarks = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape 
            cx, cy = int(lm.x * w), int(lm.y * h) 
            cv2.putText(img, str(id), (cx,cy), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
            landmarks.append((cx,cy))
 
    # calculate and print fps
    frameTime = time.time()
    fps = 1/(frameTime-prevTime)
    prevTime = frameTime
    cv2.putText(img, str(int(fps)), (30,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    #show image
    cv2.imshow('Video', img)
    cv2.waitKey(1)
    if FrameCount == FramesVideo-1:
        capture.release()
        cv2.destroyAllWindows()
        break