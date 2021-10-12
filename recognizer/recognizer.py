import face_recognition
import os
from . import helpers

class Recognizer:
    def __init__(self):
        self.known_face_encodings = list()
        self.known_face_names = list()

    def get_known_face_encodings(self, faces_dirname, userid):
        result = dict()
        home_path = helpers.get_home_path()
        faces_dirpath = helpers.get_path(home_path, faces_dirname)
        userface_dirpath = helpers.get_path(faces_dirpath, userid)

        if os.path.isdir(userface_dirpath) == False:
            raise OSError("User does not exist")

        for dirpath, dirnames, filenames in os.walk(userface_dirpath):
            for filename in filenames:
                filepath = helpers.get_path(dirpath, filename)

                if helpers.is_image(filepath):
                    filename_root = os.path.splitext(filename)[0]
                    result[filename_root] = helpers.get_face_encoding(filepath)

        self.known_face_encodings = list(result.values())
        self.known_face_names = list(result.keys())
        return

    def get_face_names(self, frame, face_locations):
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names = list()
        for face_encoding in face_encodings:
            face_name = helpers.get_face_name(self.known_face_encodings, self.known_face_names, face_encoding)
            face_names.append(face_name)
        return face_names