from imports import *


def detectfaces(frame_count,gray,first_frame,frame):
    #Locations of the haar casacades are loaded
    threshold_area=2000
    cascPath = [r"E:\EDUCATION\PROJECTS\Theft_Scanner\Cascade\haarcascade_profileface.xml",r"E:\EDUCATION\PROJECTS\Theft_Scanner\Cascade\haarcascade_frontalface_default.xml"]
    
    #Compute the absolute difference between cureent frame and the first frame
    delta_frame=cv2.absdiff(first_frame,gray)
                            #cv2.imshow('delta_frame',delta_frame)
    
    #Threshold the result of absolute difference
    thresh_frame=cv2.threshold(delta_frame,60,255,cv2.THRESH_BINARY)[1]
                            #cv2.imshow('thresh',thresh_frame)
    
    #Clean the frame     
    thresh_frame=cv2.dilate(thresh_frame,None,iterations=2)
    
    #Find contours after thresholding and if the countour area is greater than the thresholdof 5000
    #motion is detected 
    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    motion=0
    for contours in cnts:
        if cv2.contourArea(contours)< threshold_area :
            continue
        motion=1
      #  (x,y,h,w)=cv2.boundingRect(contours)
      #  cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),2)
       
        #To loop through the haar cascades               
        for j in cascPath:
          #  print(j)
            #We detect face in the frame , scalefactor and minNeighbors have the respective 
            #values as they were the most efficient.
            faceCascade=cv2.CascadeClassifier(j)
            faces = faceCascade.detectMultiScale(
                                                    gray,
                                                    scaleFactor=1.07,#If scale factor is greater the faster the program is
                                                    minNeighbors=14,
               
                                                )
            
            if len(faces)!=0: #If faces are detected
              #  for (x, y, w, h) in faces:
              #      cv2.rectangle(frame, (x, y), ((x+h)*2, (y+w)*2), (255, 255, 0), 2)  
                
                return True,motion,frame
  
            
        

    return False,motion,frame