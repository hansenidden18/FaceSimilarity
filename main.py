import cv2
import logging
from recognizer import recognizer
from detector import detector
from utility import utility

def identify(face_detector, face_recognizer, frame_count_limit=1):
    """
    Start identifying faces in video

    :param face_detector: Detector object
    :param face_recognizer: Recognizer object
    :param frame_count_limit: proccess only the `frame_count_limit`th frame
    """
    video_capture = cv2.VideoCapture(0)

    frame_count = 1
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        preprocessed_frame = utility.preprocess(frame)
        
        if frame_count % frame_count_limit == 0:
            face_locations = face_detector.get_face_locations(preprocessed_frame)
            face_names = face_recognizer.get_face_names(preprocessed_frame, face_locations)
            logging.info(utility.get_log_message(face_locations, face_detector.model, face_names))
            utility.mark_faces(frame, face_locations, face_names)
            frame_count = 1

        frame_count += 1
        prev_frame_time, new_frame_time = utility.draw_fps(frame, prev_frame_time, new_frame_time)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return

def main():
    face_recognizer = recognizer.Recognizer()

    logging.basicConfig(level=logging.INFO)
    faces_subdirname = ""
    # faces_subdirname = input("Enter your NIM: ")
    
    while True:
        try:
            face_recognizer.get_known_face_encodings(faces_dirname="pictures", faces_subdirname=faces_subdirname)
        except OSError as err:
            logging.info(str(err))
            # faces_subdirname = input("Enter your NIM: ")
            # time.sleep(10)
            continue
        # logging.info("User exists")
        break

    face_detector = detector.Detector(model="cnn")

    identify(face_detector, face_recognizer, frame_count_limit=30)

if __name__ == "__main__":
    main()