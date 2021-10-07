import face_recognition as fr
import cv2
import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import helper
import imutils
import time
import dlib

def get_home():
	return str(os.getenv('pictures', default=Path.home()))

def encode_faces():
    encoded = {}
    home = get_home()
    for path, dname,name in os.walk(home+"/pictures/"):
        for f in name:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file(home+"/pictures/"+f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded

def recognize_face():
    faces =  encode_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    # FPS
    prev_frame_time = 0
    new_frame_time = 0

    capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        ret, frame = capture.read()
        rgb_frame = frame[:,:,::-1]

        face_locations = fr.face_locations(rgb_frame)
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        # FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = "".join(["fps: ", str(int(fps))])
        cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

        for (top,right,bottom,left), face_encoding in zip(face_locations,face_encodings):
            
            matches = fr.compare_faces(faces_encoded, face_encoding)
            
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            cv2.rectangle(frame,(left-20,top-20),(right+20,bottom+20), (0,0,225),2)
            cv2.putText(frame,name,(left+6,top-6), cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

def detect_face():
    # HOG + Linear SVM
    detector = dlib.get_frontal_face_detector()
    # MMOD CNN
    # detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    capture = cv2.VideoCapture(0)

    # FPS
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        isTrue, frame = capture.read()

        frame = imutils.resize(frame, width=600)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # dlib's face detector
        dlib_faces = detector(rgb, 1)

        # FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = "".join(["fps: ", str(int(fps))])
        cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

        # convert dlib rectangle to opencv bounding boxes
        faces = []
        for dlib_face in dlib_faces:
            faces.append(helper.get_bounding_box(frame, dlib_face))

        # draw the bounding box on our frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

detect_face()
# recognize_face()