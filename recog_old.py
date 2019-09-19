#!/usr/bin/python3

import sys
import os
import dlib
import glob
import cv2
import json
import numpy as np

PATH = os.path.abspath(__file__ + "/..")
face_file = "faces.dat"
enc_file = PATH + "/models/" + face_file

pose_predictor = dlib.shape_predictor(PATH + "/models/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1(PATH + "/models/dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()

_CAM_ID = 6
dark_threshold = 50
certainty = 0.5
# TODO: Add back my face to get test
# record_mode = False
record_mode = True

cap = cv2.VideoCapture(_CAM_ID)

encodings = []
try:
    models = json.load(open(enc_file))
    for model in models:
        print(model)
        # TODO: Store model name as well
        print(model["name"])
        encodings += model["data"]
        print(encodings)
except FileNotFoundError:
    print("No encodings file")
    sys.exit(10)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    hist = cv2.calcHist([frame], [0], None, [8], [0, 256])
    hist_total = np.sum(hist)

    if hist_total == 0 or (hist[0] / hist_total * 100 > dark_threshold):
        continue

    faces = detector(frame, 1)

    for face in faces:

        face_landmark = pose_predictor(frame, face)
        face_encodering = np.array(face_encoder.compute_face_descriptor(frame, face_landmark, 1))
        
        x = int((face.right() - face.left()) / 2) + face.left()
        y = int((face.bottom() - face.top()) / 2) + face.top()

        # Get the raduis from the with of the square
        r = (face.right() - face.left()) / 2
        # Add 20% padding
        r = int(r + (r * 0.2))

        # Draw the Circle in green
        cv2.circle(frame, (x, y), r, (0, 0, 230), 2)

        # TODO: Add completely separate add user mode
        if record_mode:
            accept_prompt = input("Would you like to register this face (y/n): ")
            if accept_prompt == "y":
                name = input("What name would you like to register this face as: ")
                with open(enc_file, "w") as datafile:
                    model_info = {"name" : name, "data" : None}
                    model_info["data"] = face_encodering.tolist()
                    json.dump([model_info], datafile)
        else:
            matches = np.linalg.norm([encodings] - face_encodering, axis=1)
            match_index = np.argmin(matches)
            match = matches[match_index]

            print(match)
            if 0 < match < certainty:
                print("Hi there Thomas!")

    # Display the resulting frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
