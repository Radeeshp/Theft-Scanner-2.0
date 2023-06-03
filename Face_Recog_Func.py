from imports import *
def Face_Recognition_func(Destination_T='E:\\EDUCATION\\PROJECTS\\Theft_Scanner\\Detected\\',
                          Destination_F="E:\\EDUCATION\\PROJECTS\\Theft_Scanner\\Not Detected\\",
                          KNOWN_FACES_DIR = "E:\\EDUCATION\\PROJECTS\\Theft_Scanner\\known_faces",
                          UNKNOWN_FACES_DIR = "E:\\EDUCATION\\PROJECTS\\Theft_Scanner\\unknown_faces",
                          Face_Not_Clear="E:\\EDUCATION\\PROJECTS\\Theft_Scanner\\Face_not_clear\\"):
    
    #This variable is used to set the tolerance level for comparing the faces and 
    #Lower the number higher the accuracy(Lower false positive,higher false negative)
    TOLERANCE = 0.5
    MODEL = 'cnn'  
    
    
    #To find which name has the most positive results.
    def find_results(known_names,results_T_F,no_faces):
             
            #We are converting the  results which are in the form of a list containing True/False to 1/0  
            results_1_0 = list(map(lambda x: 1 if x else 0, results_T_F))
            
            #Flag is used to indicate if the number of matchs of the unknown faces with the given database 
            #is less than threshold
            flag=True
            current=0
            Matches_betw_kn_and_un=known_names.copy()
            
            
            for K in  known_names.keys():
                    j = known_names[K]
                    Matches_betw_kn_and_un[K]=sum(results_1_0[current:current+j+1])
                    current=current+j    
          
        
            Matches_betw_kn_and_un_copy=dict(Matches_betw_kn_and_un)
            for key,values in Matches_betw_kn_and_un_copy.items():
                if Matches_betw_kn_and_un[key]<20:
                     Matches_betw_kn_and_un.pop(key)    
            
            if len(Matches_betw_kn_and_un)==0 : flag=False    
            
            
            
            sorted_dict = sorted(Matches_betw_kn_and_un.items(), key=lambda x: x[1], reverse=True)
            top_n_salaries = sorted_dict[:no_faces] 
            # Extract the names from the top N entries
            top_n_names = [entry[0] for entry in top_n_salaries]
            print(known_names)
            print(Matches_betw_kn_and_un)
            print(flag)
            print(top_n_names)
            return(top_n_names,flag)

    #To reloate the frames location based on whether the face was recogonized or not.
    def change_location(UNKNOWN_FACES_DIR,Destination):
            try:
                if os.path.exists(Destination):
                    print("Already exists")
                else :
                    os.replace(UNKNOWN_FACES_DIR,Destination)  
                    print("Done")  
            except FileNotFoundError:
                print(UNKNOWN_FACES_DIR+"was not found")
            
            
    # Returns (R, G, B) from name
    def name_to_color(name):
        # Take 3 first letters, tolower()
        # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
        color = [(ord(c.lower())-97)*8 for c in name[:3]]
        return color


    print('Loading known faces...')
    known_faces = []
    known_names = {}
    
    # We oranize known faces as subfolders of KNOWN_FACES_DIR
    # Each subfolder's name becomes our label (name)

    for name in os.listdir(KNOWN_FACES_DIR):
        intial=0
        print(name,'is loading')
        print("There are ",len(os.listdir(f'{KNOWN_FACES_DIR}/{name}')),"for",name)
    # Next we load every file of faces of known person
        for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

            # Load an image
            
            image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

            # Get 128-dimension face encoding
            # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                encoding = encodings[0]
            else:
               
                continue

            # Append encodings and name
            known_faces.append(encoding)
            intial+=1
        known_names[name]=intial
        print(known_names)

    
    print('\n\n\nProcessing unknown faces...')
    # Now let's loop over a folder of faces we want to label
    for filename in os.listdir(UNKNOWN_FACES_DIR):

        # Load image
        print(f'Filename {filename}', end='')
        image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}\\{filename}')

        # This time we first grab face locations - we'll need them to draw boxes
        locations = face_recognition.face_locations(image, model=MODEL)
        
        # Now since we know loctions, we can pass them to face_encodings as second argument
        # Without that it will search for faces once again slowing down whole process
        encodings = face_recognition.face_encodings(image, locations)

        # We passed our image through face_locations and face_encodings, so we can modify it
        # First we need to convert it from RGB to BGR as we are going to work with cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
        print(f', found {len(encodings)} face(s)')
        no_faces=len(encodings)
        if no_faces==0:
            print(UNKNOWN_FACES_DIR+"\\"+filename ,Face_Not_Clear+filename)
            change_location(UNKNOWN_FACES_DIR+"\\"+filename ,Face_Not_Clear+filename)
            continue
        
        for face_encoding, face_location in zip(encodings, locations):

            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face withing a tolerance
            match=None
            match,flag = find_results(known_names,results,no_faces)
            if flag:  # If at least one is true, get a name of first of found labels
              #  change_location(UNKNOWN_FACES_DIR+'\\'+filename ,Destination_T+name)
                             
                match = ', '.join(match)
                
                change_location(UNKNOWN_FACES_DIR+"\\"+filename ,Destination_T+match+filename)
            else:
                print(UNKNOWN_FACES_DIR+"\\"+filename ,Destination_F+filename)
                change_location(UNKNOWN_FACES_DIR+"\\"+filename ,Destination_F+filename)
        
        # Show image
        #cv2.imshow(filename, image)
        #cv2.waitKey(0)
        #cv2.destroyWindow(filename)