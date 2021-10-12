import face_recognition

class Detector:
    def __init__(self, upsample=1, model="hog"):
        self._upsample = upsample
        self._model = model

    @property
    def model(self):
        return self._model

    def get_face_locations(self, frame):
        return face_recognition.face_locations(frame, self._upsample, self._model)