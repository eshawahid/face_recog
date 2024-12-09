# [ Start ]
#   |
#   V
# [ Receive Image File via API Endpoint ]
#   |
#   V
# [ Save the Received Image as `temp_image.jpg` ]
#   |
#   V
# [ Load Known Face Encodings and Names from the `images` Directory ]
#   |
#   V
# [ Read `temp_image.jpg` into Memory ]
#   |
#   V
# [ Detect Faces in the Image Using Face Recognition ]
#   |
#   V
# [ Check if Faces are Detected ]
#   |  
#   |--(No Faces Detected)---> [ Respond with 404 (No Faces Detected) ]
#   |  
#   |--(Faces Detected)---------> [ Compute Face Encodings for Detected Faces ]
#                                   |
#                                   V
#                                 [ Compare Detected Encodings with Known Encodings ]
#                                   |
#                                   |--(Match Found)------> [ Respond with 200 (Name of Matched Face) ]
#                                   |                        [ Log Name and Timestamp in CSV ]
#                                   |
#                                   |--(No Match Found)----> [ Respond with 404 (Face Not Matched) ]
#   |
#   V
# [ Delete `temp_image.jpg` ]
#   |
#   V
# [ End ]

import cv2
import face_recognition
import os
from datetime import datetime
import csv
import time

def construct_path(*args):
    return os.path.join(os.path.dirname(__file__), *args)


def load_face_encodings_from_directory(directory_path):
    encodings = []
    names = []

    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            name = os.path.splitext(filename)[0]  # Extract name from filename (without extension)

            
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)

            if face_encoding:
                encodings.append(face_encoding[0])
                names.append(name)
            else:
                print(f"No face detected in {image_path}")

    return encodings, names


images_directory = construct_path('images')


known_face_encodings, known_face_names = load_face_encodings_from_directory(images_directory)

now = datetime.now()
current_date = now.strftime('%Y-%m-%d')
csv_file_path = os.path.join(os.getcwd(), f'{current_date}.csv')  # Save CSV in the current directory


video_capture = cv2.VideoCapture(0)


video_capture.set(cv2.CAP_PROP_FPS, 30)


with open(csv_file_path, 'w+', newline='') as f:
    lnwriter = csv.writer(f)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to grab frame")
            break

        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

       
        face_locations = face_recognition.face_locations(rgb_frame)

  
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
           
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

         
            if name != "Unknown":
                current_time = datetime.now().strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

           
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        
        cv2.imshow('Webcam Face Recognition', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
        time.sleep(1 / 30)


    video_capture.release()
    cv2.destroyAllWindows()
