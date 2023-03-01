# Theft Scanner 2.0

A python and computer vision project to detect faces and motion, and check if thatface i present in the database and store the faces that have been detected,and when it was detected in the video

## 1 Problem Statement:
With the increasing number of cameras installed for surveillance in India, the task of reviewing and analyzing footage has become increasingly time-consuming and resource-intensive for law enforcement agencies. Indian city has the highest number of cameras in the world that is 2,75,000 cameras in Delhi [9]. There are 1.54 million cameras in India, 2,80,000 in Chennai [8] that is 1200000hrs of footage to be reviewed . However, our product offers a solution to this problem. Our product is specifically designed to streamline the process of viewing and analysing footage. With its advanced algorithms and user-friendly interface, our product allows officers to quickly and efficiently analyse footage from multiple cameras, identify key events, and generate a concise summary for official reports. Whether it's for criminal investigations, incident reports, or other purposes, this product is the perfect solution for any department looking to improve their efficiency and effectiveness in the face of the vast amount of footage in India.

## 2 Requirements:
      -computer
      -python
      -cv2
      -Video Footage
      -HaarCascade files as required
      -Packages as  required.
## 3 Steps To Run The Code:
        -Install the required packages 
        -Dont change the project structure
        -Run the code using a code editor
        -Enter the video file Location
        -Enter the location to store the image file
        -Create the folders ,Unknown_faces,Known_faces,Detected,Not_Detected.
        -Create a database of  the known faces if required.
        -Install the required packages.
## 4 Literature Survey: 
        [1] Proposed a method to analyse a video taken from the user and enhanced , if the file is of the type mpeg they carryon forward and then they perform shot segmentation ,background separation, blob detection and then the video is summarised. Summarize the videos by using the methodology of shot based key frame extraction. It is the process that can be extracted some videos that can be divided into frames and then we can reconstruct the extracted key frames into video segment. 

        [2] Proposed a method to detect motion in a video using background subtraction. In this method, the presence of moving objects is determined by comparing 2 successive frames. The previous frame is compared and then subtracted with the current frame. This allows us to obtain only those areas in the scene where motion is detected. The calculation is simple and it has a wide adaptability.
        Algorithm used by them 1. Calculate the Difference between the current Frame and previous Frame 2. Using the threshold value as a Threshold for the image calculated in (1), we calculate the areas which have changed in the current Frame from the previous Frame. 3. Resulting image from (2) is then highlighted in the current Frame to indicate areas of motion.
        [3] A new approach is proposed which is a combination of both background subtraction method and consecutive frame subtraction method to detect motion. As in this method background image is obtained by taking mean of previous consecutive frames and then this background image is compared pixel wise with current image to detect motion. Their experimental results showed that the proposed method is more robust in nature as it can avoid the noise in motion detection and it’s useful to reduce the number of false positive alarms. The methods used in detection of motion are background subtraction method, consecutive frames and threshold comparison method.
        [4]  Proposed a  method of human based keyframe extraction. They showed the different keyframe extraction method for detecting the motion can be screwed by various algorithms such as spatiotemporal filter, optical flow and background subtraction. The combined work of spatiotemporal method (HOG and SVM) identifying the human as rectangular box with certain keyframe extraction methods evaporates the required keyframe with the person suspected in the crime.
        [5] A face detection algorithm is proposed using the HCC face detector. In the first stage, the full frame is scanned by the HCC face detector, combining the close positive results to produce a single detection outcome. The HCC uses a 4-split cart as a weak classifier and sets the required true positive ratio of each point to 0.999. The HCC works in conjunction with the conciliatory AdaBoost. The positive learning set for the HCC face detector includes 2500 face images of Kathak and Kuchipudi dances. The negative set is created by randomly selecting 3500 images without any faces. Errors with a score less than 0.1 are considered as true positives, while others are considered false positives.
        [6] An efficient real-time face recognition system has been created using the VGG16 architecture of Convolutional Neural Networks (CNNs) and transfer learning methodology. This system can identify faces within a continuous video stream. The development process involved collecting face samples, preprocessing images, training the model, and performing face recognition. The average recognition accuracy achieved was between 97-99%. The system was evaluated based on factors such as the number of speakers, lighting conditions, facial expressions, occlusion, low-resolution images, shadow, and noise. The results showed that the system is capable of accurately recognizing faces for up to 20 users. In conclusion, using the VGG16 CNNs with transfer learning provides a cost-effective solution that can be run on low-end devices, while still achieving high accuracy with a limited number of images.



        
## 5 MODULES
### 5.1 Motion Detection:
Motion detection using background subtraction is a technique used to detect moving objects in a video stream. The basic idea is to subtract the current frame of the video from a reference background frame, which is usually the first frame of the video or a combination of multiple frames. The resulting difference image is thresholded to produce a binary image, where pixels representing moving objects are white and the background pixels are black. This binary image is then processed to extract contours, and the contours are analyzed to determine if there is motion in the video. If the area of a contour is above a certain threshold, it is considered to be motion. This technique is simple, efficient, and widely used for motion detection in surveillance systems and other video processing applications. However, it can be affected by changes in lighting conditions, camera noise, and other factors, which may result in false detections or missed detections. 
### 5.2 Face Detection: 
Face detection is performed using Haarcascades. Haarcascades is a machine learning object detection method used for face and object recognition. It uses Haar-like features, simple rectangular filters that can be applied to image data to extract features. The process involves training a classifier on a large set of positive and negative samples, then using the classifier to identify objects in new images. This is a module for detecting faces in a video stream using the OpenCV library. The function takes as inputs the frame count, the current grayscale frame, the first frame of the video, and the current frame. The absolute difference between the first frame and the current frame is computed and thresholded . Contours are then found in the thresholded frame, and if the contour area is greater than a certain threshold (2000), motion is detected. This module uses Haar cascades from two XML files to detect faces in the current frame. If faces are detected in the current frame, the function returns True, the motion flag, and the current frame with faces marked with rectangles. If no faces are detected, the function returns False, the motion flag, and the current frame.
### 5.3 Face Recognition:
The "face_recognition" module [7] in Python is a library that is used for face recognition and facial recognition tasks. It uses deep learning algorithms to compare a target face to a database of known faces, and it can also be used to identify specific individuals in real-time.
This is a python module  that uses the face_recognition library to perform face recognition. The script defines the Face_Recognition_func function that takes in 4 parameters, the destinations for detected and not detected faces, and the directories for known and unknown faces.

The function starts by loading all the known faces and names, where the faces are stored in subdirectories of KNOWN_FACES_DIR with the subdirectory names being the labels (names) of the faces. It then processes the unknown faces in the UNKNOWN_FACES_DIR by finding the locations of the faces in the images and then encoding those faces.
Next, the function performs face comparison between the unknown faces and the known faces using the face_recognition.compare_faces method. If the comparison returns a positive result, the function finds the name with the most positive results, draws a rectangle around the face and writes the name on the image using the cv2.rectangle and cv2.putText functions from the OpenCV library.
Finally, the function moves the image to the specified destination folder depending on whether a face was detected or not. The change_location method is used for this purpose.

## 6 ANALYSIS
The python version used in this project is  3.10.9
### 6.1 Data Set Description:
        1.	File type: mp4 file
        2.	File size: 7.55 mb
        3.	Video length: 00:1:25sec
        4.	Frame width: 640
        5.	Frame height: 352
        6.	Frame per second: 30.00fps

### 6.2 Modules Used:
        1.	 asyncio.windows_events 
        2.	 cv2
        3.	 sys
        4.	 CascadeClassifier
        5.	 numpy 
        6.	 face_recognition
        7.	os
        8.	 shutil
### 6.3 Functions Used:		
        1.	os.path.exists()
        2.	os.replace()
        3.	os.listdir()
        4.	face_recognition.load_image_file()
        5.	face_recognition.face_encodings()
        6.	face_recognition.face_locations()
        7.	cv2.cvtColor()
        8.	face_recognition.compare_faces()
        9.	cv2.rectangle()
        10.	cv2.putText()
        11.	cv2.absdiff()
        12.	cv2.threshold()
        13.	cv2.dilate()
        14.	cv2.findContours()
        15.	cv2.contourArea()
        16.	cv2.CascadeClassifier()
        17.	faceCascade.detectMultiScale()
        18.	video_capture.get(cv2.CAP_PROP_FPS)
### 6.4 Important Variable Used and Their Values:
        1.	threshold_area:
        It is a variable in Face_Detect_Func.py, its used to as a threshold to filter out contours that are smaller that it in area. It has been given a value of 2000 as it was big enough to avoid small leaves moving from being detected as motion, and small enough that it did not miss  any big objects (humans,animal,cars,etc) moving.
        2.	scaleFactor:
        It is a variable in Face_Detect_Func.py, its used in the function faces = faceCascade.detectMultiScale() {6.3.17}
        The value of 1.07 has been selected as it is the most efficient and accurate. This value was selected after values greater and lower than these were verified.
        3.	minNeighbors:
        It is a variable in Face_Detect_Func.py, its used in the function faces = faceCascade.detectMultiScale() {6.3.17}
        The value of 14 has been selected as it is the most efficient and accurate. This value was selected after values greater and lower than these were verified.
        4.	TOLERANCE:
        It is a variable in Face_Recog_Func.py, its used  it the function {6.3.8} face_recognition.compare_faces().
        This value is used to as a threshold as to decide how similar 2 images have to for it to be classified as recogonized.

## 7 REFERENCES:
        [1] Video Summarization of Surveillance Videos Using Key Frame Extraction
        1K.S.R.Manjusha, 2K. Pavan Kumar, 3D. Uma
        https://drive.google.com/file/d/1UHqtL5Xk2T_P6ewbcYdqe_CG61rj12cR/view

        [2] Motion detection algorithm based on Background Subtraction 
        Shivam Shah1 , Vivek Adhikari2 , Vineet Pokhriyal3
        https://www.ijser.org/researchpaper/Motion-detection-algorithm-based-on-Background-Subtraction.pdf

        [3] Real Time Motion Detection Using Background Subtraction Method and Frame Difference Lavanya M P
        https://www.ijsr.net/archive/v3i6/MDIwMTQ2NjE=.pdf

        [4] A COMPARATIVE STUDY FOR HUMAN BASED KEYFRAME
        EXTRACTION FOR CRIME SCENE INVESTIGATION IN CCTV SURVEILLANCE
        Rajeshwari , Victoria Priscilla 
        https://drive.google.com/file/d/1R1gqzNJS8LwKS2JrH6EYFxbKFYahPBMy/view

        [5] Face Detection Using Haar Cascade Classifiers 
        Bhavana R. Maale  ,  Dr. Suvarna Nandya
        https://www.ijsr.net/archive/v10i3/SR21306204717.pdf

        [6] Real Time Face Recognition System Using Convolutional Neural Network 
        Dr.Vinayak Bharadi, Mr.Rutik Sansare, Mr.Tushar Padelkar, Mr.Vishant Shinde
        https://ijcrt.org/papers/IJCRTO020015.pdf

        [7] https://github.com/ageitgey/face_recognition

        [8] https://thewire.in/rights/cctv-surveillance-is-rising-in-india-world-but-crime-rates-remain-unaffected
        [9] https://www.livemint.com/news/india/this-indian-city-has-the-highest-number-of-cctvs-in-the-world-beats-london-paris-11638517557065.html
        [10] opencv.com
        [11] google.com
        [12] github.com
