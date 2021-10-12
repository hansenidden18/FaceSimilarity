import cv2
from recognizer import recognizer
from detector import detector
from utility import utility
import logging

def identify(face_detector, face_recognizer, frame_count_limit=1, model="hog"):
    """
    Start identifying faces in video

    :param face_detector: Detector object
    :param face_recognizer: Recognizer object
    :param frame_count_limit: proccess only the `frame_count_limit`th frame
    :param model: model used for face detection
    """
    video_capture = cv2.VideoCapture(0)

    frame_count = 1

    while True:
        ret, frame = video_capture.read()

        preprocessed_frame = utility.preprocess(frame)
        
        if frame_count % frame_count_limit == 0:
            face_locations = face_detector.get_face_locations(preprocessed_frame)
            face_names = face_recognizer.get_face_names(preprocessed_frame, face_locations)
            logging.info(utility.get_log_message(face_locations, face_names))
            utility.mark_faces(frame, face_locations, face_names)
            frame_count = 1

        frame_count += 1

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return

def main():
    face_recognizer = recognizer.Recognizer()

    logging.basicConfig(level=logging.INFO)
    userid = ""
    # userid = input("Enter your NIM: ")
    
    while True:
        try:
            face_recognizer.get_known_face_encodings(faces_dirname="pictures", userid=userid)
        except OSError as err:
            logging.info(str(err))
            # userid = input("Enter your NIM: ")
            # time.sleep(10)
            continue
        logging.info("User exists")
        break

    face_detector = detector.Detector(model="cnn")

    identify(face_detector, face_recognizer, frame_count_limit=30)

if __name__ == "__main__":
    main()