import face_recognition

class Detector:
    
    def get_face_locations(self, frame, upsample, model):
        return face_recognition.face_locations(frame, upsample, model)