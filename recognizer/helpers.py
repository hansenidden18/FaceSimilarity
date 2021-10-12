import face_recognition
import os
import constants
from PIL import Image
import numpy as np

def is_image(filepath):
    try:
        image = Image.open(filepath)
        return True
    except:
        return False

def get_home_path():
    return os.path.expanduser("~")

def get_path(dirpath, filename):
    return os.path.join(dirpath, filename)

def get_face_encoding(filepath):
    face_image = face_recognition.load_image_file(filepath)
    face_enconding = face_recognition.face_encodings(face_image)[0]
    return face_enconding

def get_face_name(known_face_encodings, known_face_names, face_encoding):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     return known_face_names[first_match_index]

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        return known_face_names[best_match_index]

    return constants.UNREGISTERED_LABEL