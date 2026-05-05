from config.settings import HEAD_TURN_THRESHOLD

NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454


def detect_head_turn(landmarks, frame_shape):
    h, w, _ = frame_shape

    nose_x = landmarks[NOSE_TIP].x * w
    left_x = landmarks[LEFT_FACE].x * w
    right_x = landmarks[RIGHT_FACE].x * w

    face_center = (left_x + right_x) / 2
    offset = nose_x - face_center

    if offset > HEAD_TURN_THRESHOLD:
        return "RIGHT"

    if offset < -HEAD_TURN_THRESHOLD:
        return "LEFT"

    return "CENTER"