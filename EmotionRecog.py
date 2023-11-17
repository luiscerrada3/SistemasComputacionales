from deepface import DeepFace
from pathlib import Path
import numpy as np
import cv2
import os


#Recorre la lista de videos en la carpeta y va agregando los frames con la emocion detectada 
#y la pregunta a la que corresponde cuando hay una separacion de negro en el video es que cambia
#a la siguiente pregunta las carpetas y los videos no estan
for x in range(1,12):
    print(x)
    cam = cv2.VideoCapture("VIDEO"+str(x)+".mp4") 
    flag = True
    currentframe = 0
    questionNumber=0
    lastEmotion=""
    questions=[]

    while(True):      

        ret,frame = cam.read()   
    
        if ret:
            if np.sum(frame) == 0:
                if(flag):
                    questionNumber +=1
                    flag=False

            else:
                if (currentframe % 5 == 0):
                    flag = True
                    name = 'user'+str(x)+'/'+str(questionNumber)+ '-'+lastEmotion+'-'+str(currentframe) + '.jpg'
                    print ('Creating...' + name) 

                    cv2.imwrite(name, frame) 
                    objs = DeepFace.analyze(img_path=name, actions = ['emotion'] ,enforce_detection=False )
                    lastEmotion = objs[0]['dominant_emotion']
  
            currentframe += 1
        else: 
            break
    cam.release() 
    cv2.destroyAllWindows() 





print("final")