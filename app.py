import streamlit as st
import av
import cv2
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from modules.liveness import check_liveness, reset_liveness


st.set_page_config(
    page_title="AI Liveness Detection",
    layout="centered"
)

st.title("AI Face Liveness Detection")
st.write("Follow the random challenge shown on screen. Anti-spoofing and image quality must pass.")

if st.button("Reset Verification"):
    reset_liveness()
    st.success("Verification reset")


class LivenessProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        result = check_liveness(img)
        print("LIVENESS RESULT KEYS:", result.keys())
        print("LIVENESS RESULT:", result)

        status = result["status"]
        message = result["message"]

        if status == "LIVE":
            color = (0, 255, 0)
        elif status in ["NO_FACE", "SPOOF", "BAD_IMAGE", "TIMEOUT", "ANTI_SPOOF_MISSING"]:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        y = 40

        def draw_line(text, line_color=(255, 255, 255), scale=0.65):
            nonlocal y

            cv2.putText(
                img,
                text,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                line_color,
                2
            )

            y += 35

        draw_line(f"Status: {status}", color, 1.0)
        draw_line(message, color, 0.75)

        current_challenge_text = result.get("current_challenge_text", "MISSING_KEY")
        challenges = result.get("challenges", [])
        completed = result.get("completed_challenges", [])

        draw_line(f"Challenge: {current_challenge_text}")
        draw_line(f"Sequence: {challenges}")
        draw_line(f"Completed: {len(completed)} / {len(challenges)}")
        draw_line(f"Blinks: {result.get('blink_count')}")
        draw_line(f"Head: {result.get('head_direction')}")

        draw_line(f"Image Quality: {result.get('quality_ok')}")
        draw_line(f"Quality Message: {result.get('quality_message')})")
        draw_line(f"Brightness: {result.get('brightness')}")
        draw_line(f"Blur: {result.get('blur_score')}")

        draw_line(f"Anti Spoof OK: {result.get('spoof_ok')}")

        spoof_score = result.get("spoof_score")

        if spoof_score is not None:
            draw_line(f"Real Score: {spoof_score}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


ice_servers = [
    {
        "urls": [
            "stun:stun.relay.metered.ca:80",
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
        ]
    }
]

turn_url = os.getenv("TURN_URL")
turn_username = os.getenv("TURN_USERNAME")
turn_credential = os.getenv("TURN_CREDENTIAL")

if turn_url and turn_username and turn_credential:
    ice_servers.append(
        {
            "urls": [turn_url],
            "username": turn_username,
            "credential": turn_credential,
        }
    )

ctx = webrtc_streamer(
    key="liveness-demo",
    video_processor_factory=LivenessProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    rtc_configuration={
        "iceServers": ice_servers
    },
    async_processing=True,
)