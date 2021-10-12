import cv2
from recognizer import recognizer
from detector import detector
from utility import utility

def recognize_face(face_recognizer):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = detector.get_face_locations(rgb_frame, 1, "hog")
        face_names = face_recognizer.get_face_names(rgb_frame, face_locations)
        utility.mark_faces(frame, face_locations, face_names)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    face_recognizer = recognizer.Recognizer(faces_dirname="pictures")
    recognize_face(face_recognizer)

if __name__ == "__main__":
    main()