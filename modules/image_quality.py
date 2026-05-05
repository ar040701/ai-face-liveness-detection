import cv2
import numpy as np


def check_image_quality(frame, landmarks=None):
    """
    Checks whether the camera frame is good enough for liveness verification.

    Conditions:
    1. Image should not be too dark
    2. Image should not be too bright
    3. Image should not be blurry
    4. Face should be large enough
    5. Face should be roughly centered
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness_ok = 50 <= brightness <= 210
    blur_ok = blur_score >= 70

    face_size_ok = True
    face_center_ok = True

    face_box = None

    if landmarks is not None:
        h, w, _ = frame.shape

        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x_min = max(0, min(xs))
        x_max = min(w, max(xs))
        y_min = max(0, min(ys))
        y_max = min(h, max(ys))

        face_width = x_max - x_min
        face_height = y_max - y_min

        face_area_ratio = (face_width * face_height) / float(w * h)

        face_center_x = (x_min + x_max) / 2
        face_center_y = (y_min + y_max) / 2

        frame_center_x = w / 2
        frame_center_y = h / 2

        center_offset_x = abs(face_center_x - frame_center_x) / w
        center_offset_y = abs(face_center_y - frame_center_y) / h

        face_size_ok = face_area_ratio >= 0.12
        face_center_ok = center_offset_x <= 0.25 and center_offset_y <= 0.25

        face_box = {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
            "face_area_ratio": round(face_area_ratio, 3)
        }

    quality_ok = brightness_ok and blur_ok and face_size_ok and face_center_ok

    message = "Image quality good"

    if not brightness_ok:
        if brightness < 50:
            message = "Image too dark"
        else:
            message = "Image too bright"

    elif not blur_ok:
        message = "Image too blurry"

    elif not face_size_ok:
        message = "Move closer to camera"

    elif not face_center_ok:
        message = "Center your face"

    return {
        "quality_ok": quality_ok,
        "message": message,
        "brightness": round(brightness, 2),
        "blur_score": round(blur_score, 2),
        "brightness_ok": brightness_ok,
        "blur_ok": blur_ok,
        "face_size_ok": face_size_ok,
        "face_center_ok": face_center_ok,
        "face_box": face_box
    }