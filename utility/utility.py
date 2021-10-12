import cv2
import constants

def restore_scale(top, right, bottom, left):
    top *= int(1 / constants.FY)
    right *= int(1 / constants.FX)
    bottom *= int(1 / constants.FY)
    left *= int(1 / constants.FX)
    return (top, right, bottom, left)

def mark_faces(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        (top, right, bottom, left) = restore_scale(top, right, bottom, left)

        cv2.rectangle(
            img=frame,
            pt1=(left - constants.LEFT_OFFSET, top - constants.TOP_OFFSET),
            pt2=(right + constants.RIGHT_OFFSET, bottom + constants.BOTTOM_OFFSET),
            color=(0, 0, 255),
            thickness=2
        )

        (text_width, text_height), base_line = cv2.getTextSize(text=name, fontFace=constants.DEFAULT_FONT, fontScale=1, thickness=1)

        cv2.putText(
            img=frame,
            text=name,
            org=(left, bottom + text_height),
            fontFace=constants.DEFAULT_FONT,
            fontScale=1.0,
            color=(255, 255, 255),
            thickness=1
        )
    return

def preprocess(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    preprocessed_frame = cv2.resize(
        src=rgb_frame,
        dsize=(0, 0),
        fx=constants.FX,
        fy=constants.FY
    )
    
    return preprocessed_frame