import face_recognition

class Detector:
    def __init__(self, upsample=1, model="hog"):
        self.upsample = upsample
        self.model = model
    def get_face_locations(self, frame):
        return face_recognition.face_locations(frame, self.upsample, self.model)