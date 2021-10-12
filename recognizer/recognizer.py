import face_recognition
import numpy as np
import os
import constants
from . import helpers

class Recognizer:
    def __init__(self, faces_dirname):
        result = dict()
        home_path = helpers.get_home_path()
        faces_dirpath = helpers.get_path(home_path, faces_dirname)

        for dirpath, dirnames, filenames in os.walk(faces_dirpath):
            for filename in filenames:
                filepath = helpers.get_path(dirpath, filename)

                if helpers.is_image(filepath):
                    filename_root = os.path.splitext(filename)[0]
                    result[filename_root] = helpers.get_face_encoding(filepath)

        self.known_face_encodings = list(result.values())
        self.known_face_names = list(result.keys())

    def get_face_names(self, frame, face_locations):
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names = helpers.get_face_name(face_encodings, self.known_face_encodings, self.known_face_names)
        return face_names