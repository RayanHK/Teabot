import numpy, cv2, mediapipe
from module import findnameoflandmark,findpostion
from collections import Counter

### C:\Users\invate\AppData\Local\Programs\Python\Python38\Hackathon   <-- python files stored here
# In that folder there is "code-for-hand-identification somethig someting." THat does stuff individually (like counting fingers), might be useful if stuck
#
#https://www.youtube.com/watch?v=a7B5EZVHHkw&t=274s  <-- Very useful video
#



drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

tip=[8,12,16,20]
tipname=[8,12,16,20]

def detectHandGesture(frame): #Detects points of hands, its the skeleton thing. You can access those coordinates somehow, check the folder I mentioned at top of code, dont have time to figure it out
    with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7,min_tracking_confidence=0.7,max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks != None:
            return results.multi_hand_landmarks
        else:
            return None



def checkFingersUp(pos, fingerName): #Checks how many fingers up and which one

    #Appends target finger to list which is returned at end
    finger=[]
    fingers=[]
    fingersSeen = []
    if len(pos and fingerName)!=0:
        #Checks if thumb is up
        if pos[0][1:] < pos[4][1:]: 
           finger.append(1)
           print (fingerName[4])
           fingersSeen.append(fingerName[4])
          
        else:
           finger.append(0)   
        
         #Checks all other fingers
        for id in range(0,4):
            if pos[tip[id]][2:] < pos[tip[id]-2][2:]:

               print(fingerName[tipname[id]])

               fingersSeen.append(fingerName[tipname[id]])
                
               fingers.append(1)
    
            else:
               fingers.append(0)
     #Below will print to the terminal the number of fingers that are up or down          
    x=fingers + finger
    c=Counter(x)
    up=c[1]
    down=c[0]
    print('This many fingers are up - ', up)
    print('This many fingers are down - ', down)

    print(fingersSeen)
    return fingersSeen
        

def checkCamera(ticks, checkForFinger="THUMB"):  #ticks = how many loops,checkForFinger = which finger to check for (thumb by default). (INDEX FINGER, MIDDLE FINGER, RING FINGER, THUMB, PINKY)
    capture = cv2.VideoCapture(0)
    scaleFactor = 1.3
    minNeighbours = 5

    ticksChecking = 0

    fingerFound = False

    seenSolo = []

    while ticksChecking<ticks:
        ret, frame = capture.read()
        
        faces = faceDetection(frame, scaleFactor, minNeighbours) # faces stores location of face
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pos=findpostion(frame)
        fingerName=findnameoflandmark(frame)
        fingerList = checkFingersUp(pos,fingerName)
        if len(fingerList) == 1:
            print("I am here and seeing " + checkForFinger + " TIP")
            print("Compared to", fingerList[0])
            if fingerList[0] == (checkForFinger + " TIP"):
                print(checkForFinger, "FOUND")
                seenSolo.append(checkForFinger)

        mhl = detectHandGesture(frame)
        if mhl != None:
            for handLandmarks in mhl:
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)

        #Draws rectangles around face/eyes.
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5) #Draw rectangle around face
            faceRegion_gray = grayFrame[y:y+w, x:x+w]
            faceRegion_colour = frame[y:y+w, x:x+w]

            eyes = detectEye(faceRegion_gray, scaleFactor, minNeighbours)

            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(faceRegion_colour, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5) #Draw rectangle around eyes
        
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
        print(ticksChecking)
        ticksChecking+=1

        

    
    capture.release()
    cv2.destroyAllWindows()
    return len (seenSolo)  #seensolo is list which is appended to everytime target finger is found. length list is how many times its found. THis is returned
    #return len(fingersSeenAlone)

#Detcts face
def faceDetection(frame, scaleFactor, minNeighbours): #Frame = camera, scaleFactor = bigger -> more accurate but slower compute time, minNeighbours = idk, something about size it detects at. Keep it at 5 thats recommended
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor, minNeighbours) #Change 1.3 scale factor. Trade off accuracy for speed
    return faces

#Detects eyes
def detectEye(frame, scaleFactor, minNeighbours):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(frame, scaleFactor, minNeighbours)
    return eyes

checkCamera(25)

#TESTING CODE FOR TESTING PURPOSES
#test = 0
#for i in range(100):
 #   timesSeen = checkCamera(15)
  #  print(timesSeen, "times")

   # if (timesSeen >= 1):
    #    test+=1
     #   print("\n\n\n\n\n\n")

#print("FOUND IT", test, " TIMES")
