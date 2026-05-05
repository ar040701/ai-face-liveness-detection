import numpy as np
from config.settings import EAR_THRESHOLD

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(points):
    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal == 0:
        return 0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def detect_blink(landmarks, frame_shape):
    h, w, _ = frame_shape

    def get_points(indices):
        return [
            (
                int(landmarks[i].x * w),
                int(landmarks[i].y * h)
            )
            for i in indices
        ]

    left_eye = get_points(LEFT_EYE)
    right_eye = get_points(RIGHT_EYE)

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    ear = (left_ear + right_ear) / 2.0

    return ear < EAR_THRESHOLD, ear