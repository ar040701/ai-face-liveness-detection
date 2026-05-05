import random
import time

from modules.blink_detection import detect_blink
from modules.head_pose import detect_head_turn
from modules.anti_spoof import AntiSpoofModel
from modules.face_landmarker import FaceLandmarkerDetector
from modules.image_quality import check_image_quality

from config.settings import RANDOM_CHALLENGE_COUNT, CHALLENGE_TIMEOUT_SECONDS


face_detector = FaceLandmarkerDetector()
anti_spoof_model = AntiSpoofModel()


class LivenessState:
    def __init__(self):
        self.blink_count = 0
        self.was_blinking = False

        self.looked_left = False
        self.looked_right = False

        self.available_challenges = [
            "BLINK",
            "TURN_LEFT",
            "TURN_RIGHT"
        ]

        self.challenges = random.sample(
            self.available_challenges,
            k=min(RANDOM_CHALLENGE_COUNT, len(self.available_challenges))
        )

        self.current_challenge_index = 0
        self.completed_challenges = []

        self.challenge_started_at = time.time()
        self.verified = False


state = LivenessState()


def reset_liveness():
    global state
    state = LivenessState()


def format_challenge(challenge):
    if challenge == "BLINK":
        return "Blink now"

    if challenge == "TURN_LEFT":
        return "Turn your head left"

    if challenge == "TURN_RIGHT":
        return "Turn your head right"

    if challenge is None:
        return "All challenges completed"

    return str(challenge)


def get_current_challenge():
    if state.current_challenge_index >= len(state.challenges):
        return None

    return state.challenges[state.current_challenge_index]


def get_base_response():
    current_challenge = get_current_challenge()

    return {
        "blink_count": state.blink_count,
        "head_left": state.looked_left,
        "head_right": state.looked_right,
        "current_challenge": current_challenge,
        "current_challenge_text": format_challenge(current_challenge),
        "completed_challenges": state.completed_challenges,
        "challenges": state.challenges,
        "challenge_ok": current_challenge is None,
    }


def challenge_timed_out():
    return time.time() - state.challenge_started_at > CHALLENGE_TIMEOUT_SECONDS


def move_to_next_challenge():
    state.current_challenge_index += 1
    state.challenge_started_at = time.time()


def check_current_challenge(is_blinking, head_direction):
    current_challenge = get_current_challenge()

    if current_challenge is None:
        return True

    challenge_passed = False

    if current_challenge == "BLINK":
        if is_blinking and not state.was_blinking:
            challenge_passed = True

    elif current_challenge == "TURN_LEFT":
        if head_direction == "LEFT":
            challenge_passed = True

    elif current_challenge == "TURN_RIGHT":
        if head_direction == "RIGHT":
            challenge_passed = True

    if challenge_passed:
        state.completed_challenges.append(current_challenge)
        move_to_next_challenge()

    return get_current_challenge() is None


def check_liveness(frame):
    base = get_base_response()

    if state.verified:
        return {
            **base,
            "status": "LIVE",
            "message": "Live person already verified",
            "head_direction": "CENTER",
            "spoof_model_available": anti_spoof_model.model_available,
            "spoof_score": 1.0,
            "spoof_ok": True,
            "quality_ok": True,
            "quality_message": "Verified",
            "brightness": None,
            "blur_score": None,
        }

    landmarks = face_detector.detect(frame)

    if landmarks is None:
        return {
            **base,
            "status": "NO_FACE",
            "message": "No face detected",
            "head_direction": "N/A",
            "spoof_score": None,
            "spoof_ok": False,
            "spoof_model_available": anti_spoof_model.model_available,
            "quality_ok": False,
            "quality_message": "No face detected",
            "brightness": None,
            "blur_score": None,
        }

    quality_result = check_image_quality(frame, landmarks)

    if not quality_result["quality_ok"]:
        return {
            **base,
            "status": "BAD_IMAGE",
            "message": quality_result["message"],
            "head_direction": "N/A",
            "spoof_score": None,
            "spoof_ok": False,
            "spoof_model_available": anti_spoof_model.model_available,
            "quality_ok": False,
            "quality_message": quality_result["message"],
            "brightness": quality_result["brightness"],
            "blur_score": quality_result["blur_score"],
        }

    if challenge_timed_out():
        reset_liveness()
        base = get_base_response()

        return {
            **base,
            "status": "TIMEOUT",
            "message": "Challenge timed out. New challenge started.",
            "head_direction": "N/A",
            "spoof_score": None,
            "spoof_ok": False,
            "spoof_model_available": anti_spoof_model.model_available,
            "quality_ok": True,
            "quality_message": "Image quality good",
            "brightness": quality_result["brightness"],
            "blur_score": quality_result["blur_score"],
        }

    is_blinking, ear = detect_blink(landmarks, frame.shape)
    head_direction = detect_head_turn(landmarks, frame.shape)

    if is_blinking and not state.was_blinking:
        state.blink_count += 1

    if head_direction == "LEFT":
        state.looked_left = True

    if head_direction == "RIGHT":
        state.looked_right = True

    challenge_ok = check_current_challenge(
        is_blinking=is_blinking,
        head_direction=head_direction
    )

    state.was_blinking = is_blinking

    spoof_result = anti_spoof_model.predict(frame)

    spoof_ok = spoof_result["available"] and spoof_result["is_real"]

    live = challenge_ok and spoof_ok

    if live:
        state.verified = True
        status = "LIVE"
        message = "Live person verified"

    elif not spoof_result["available"]:
        status = "ANTI_SPOOF_MISSING"
        message = "Anti-spoof model missing or failed"

    elif challenge_ok and not spoof_ok:
        status = "SPOOF"
        message = "Spoof detected"

    else:
        status = "CHECKING"
        message = f"Challenge: {format_challenge(get_current_challenge())}"

    base = get_base_response()

    return {
        **base,
        "status": status,
        "message": message,
        "ear": round(ear, 3),
        "head_direction": head_direction,
        "spoof_model_available": spoof_result["available"],
        "spoof_score": round(spoof_result["score"], 3),
        "spoof_ok": spoof_ok,
        "quality_ok": quality_result["quality_ok"],
        "quality_message": quality_result["message"],
        "brightness": quality_result["brightness"],
        "blur_score": quality_result["blur_score"],
        "challenge_ok": challenge_ok,
    }