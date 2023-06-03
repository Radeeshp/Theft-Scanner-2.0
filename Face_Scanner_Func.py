#This code is used to generate the known face database from a video input of the user which 
#contains the video of a known face


from imports import *
# Provide the location path
def create_folder():
    location = "E:\\EDUCATION\\PROJECTS\\Theft_Scanner\\known_faces\\"
    name=input("Enter your name: ")
# Define the path for the new folder
    try:
        # Create the folder in the given path
        os.mkdir(location+name)
        print(location+name)
    except OSError:
        print ("Creation of the directory %s failed" % location+name)
    else:
        print ("Successfully created the directory %s " % location+name)
    return location+name
#To create a database of photos of a person from the video the input.
def create_known_face(database_location):
    #path to the input video
    
    name=input("\nEnter the file name with its extension:")
    video_capture = cv2.VideoCapture("E:\EDUCATION\PROJECTS\Theft_Scanner\Input_vid_known_face"+"\\"+name)
    frame_count=0
    no_saved_frame=0
    check = True
    while no_saved_frame<100 and check is True:
        frame_count+=1
        #read the video
        check,frame=video_capture.read()
        if check is True and frame_count%5==0:
        #writing the video as frames in the video as frame in the folder 
            cv2.imwrite(database_location+"\\"+str(frame_count)+".jpg",frame)
            no_saved_frame+=1
    print("Extracted")    
            