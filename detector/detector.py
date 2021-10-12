import face_recognition

def get_face_locations(frame, upsample, model):
    return face_recognition.face_locations(frame, upsample, model)