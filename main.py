import cv2
from recognizer import recognizer
from detector import detector
from utility import utility

def recognize_face(face_detector, face_recognizer):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        preprocessed_frame = utility.preprocess(frame)
        
        face_locations = face_detector.get_face_locations(preprocessed_frame, 1, "hog")
        face_names = face_recognizer.get_face_names(preprocessed_frame, face_locations)
        utility.mark_faces(frame, face_locations, face_names)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    face_recognizer = recognizer.Recognizer()
    
    try:
        face_recognizer.get_known_face_encodings(faces_dirname="pictures", userid="05111940000075")
    except OSError as err:
        print("".join(["INFO: ", str(err)]))

    face_detector = detector.Detector()

    recognize_face(face_detector, face_recognizer)

if __name__ == "__main__":
    main()