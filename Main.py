
# imports has all the common required imports 
from imports import *
from Face_Recog_Func import Face_Recognition_func
from Face_Detect_Func import detectfaces
from Face_Scanner_Func import create_folder
from Face_Scanner_Func import create_known_face

def get_info():
    
    if input("Do  you want to change any location details\n Press 1 if yes \n Press 2 if No\n")==2:
        Face_Recognition_func(UNKNOWN_FACES_DIR = location_output)
    else :
        Face_Recognition_func(Destination_T=input("Destination_Recognized Faces:"),
                              Destination_F=input("Destination_UnRecognized Faces:"),
                              KNOWN_FACES_DIR = input("Location_known Faces:"),
                              UNKNOWN_FACES_DIR = location_output
                             )
    

def face_database():

    while True:
        if input("Enter 1 to create database for new person:")=='1':
        
            location=create_folder()
            create_known_face(location)
            print("/nFace database has been created for the given input")
        else:break
    print("\nFace and motion detection is starting\n")


#Call's the Face_Recognition_func 
def face_recog():
    Face_Recognition_func()
    
    
# Converts seconds to hours/minutes/seconds/milliseconds and returns it
def convert_seconds_to_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)
    milliseconds = int(milliseconds * 1000)
    return "%d.%d.%d.%d" % (int(hours), int(minutes), int(seconds),milliseconds)

def customize_values():# used to customizee values like motion_frame_max, tolerance, etc...
    pass



if __name__=="__main__":
    
    os.system('cls')
   
    print("**********************Welcome to Theft_Scanner*******************************\n\n")
    
    print("**********************You Have Accessed The Facial And Motion Scannner and Detector*******************************\n\n")
    
    """
     
    print("Enter the video file location")
    location_input=input()
    print("Enter the location to store the images where a face is detected")
    location_output=input()
    print("Enter the location to store the images where motion is detected")
    motion_file=input()
    """
    #Function called to create face database for new people or strangers.
    face_database()
    
    location_output='E:\\EDUCATION\\PROJECTS\\Theft_Scanner\\unknown_faces\\'
    motion_output="E:\\EDUCATION\\PROJECTS\\Theft_Scanner\\Motion_Detected\\"
    frame_count=0
    
    #intialize the video frame
    file_name=input("\nEnter the input video file name with the extension:")
    video_capture = cv2.VideoCapture("E:\\EDUCATION\\PROJECTS\\Theft_Scanner\\Input_Video"+"\\"+file_name)
    #intialize variables  
    first_frame=None    #Used for motion detection
    frame_count_act=0   #Used to count no of frames
    motion_frame_max=99 #Used to save frames in which motion has been detected
    motion_frame=1      #Used as a BOOL , if motion is detected or not

    while True:
        # Capture frame-by-frame
       
        frame_count_act+=1
        frame_count+=1
        
        #reads the video frame
        check,frame=video_capture.read()
        if not check:
            print("The video frame is unread able or ended \n\nFace Recognition has started......")
            break
         
        #Convert the frame from colour/BGR to grayscalew
        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        #Saving the first frame to detect motion
        if first_frame is None:
            first_frame=gray 
            continue
        
        #Used to time stamp the frames 
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        time=str(convert_seconds_to_time(frame_count_act/fps))
        
        #Function called from Face_Detect_Func.py to detect motion
        faces,motion,frame=detectfaces(frame_count,gray,first_frame,frame)
  
        #Saving the frame in which frame has been detected  
        if faces==True:
            cv2.imwrite(location_output+time+".jpg", frame)
            motion_frame+=1
            
        #Only 1 frame is saved for every 99(motion_frame_max) frames in which motion was detected 
        elif motion==1 and motion_frame%motion_frame_max==0: 
            motion_frame+=1
            cv2.imwrite(motion_output+time+".jpg", frame)
            
            
        elif motion==1 :
            motion_frame+=1
        
        #Resetting the the first frame if motion is not detected   
        else :
            motion=10
            if(frame_count>1000):
                first_frame=gray
                frame_count=0
          
    
#Function called from Face_Recog_Func.py to recognize faces    
print("face detection is over")
face_recog()

print("\n*******************Program Has Ended********************")
video_capture.release()
cv2.destroyAllWindows()
