# AI Face Liveness Detection

A real-time face liveness detection system built with **Streamlit**, **WebRTC**, **OpenCV**, **MediaPipe**, and **ONNX Runtime**.

The app verifies whether the person in front of the camera is live by using:

- Face landmark detection
- Blink detection
- Head movement detection
- Random challenge verification
- Image quality checks
- ONNX anti-spoofing model

---

## Features

- Real-time webcam processing
- Face landmark detection using MediaPipe
- Blink detection using Eye Aspect Ratio
- Head turn detection
- Random challenges:
  - Blink
  - Turn left
  - Turn right
- Anti-spoofing using ONNX Runtime
- Image quality checks:
  - Brightness
  - Blur
  - Face size
  - Face position

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Streamlit | Web app interface |
| streamlit-webrtc | Webcam streaming |
| OpenCV | Image processing |
| MediaPipe | Face landmarks |
| ONNX Runtime | Anti-spoof model inference |
| NumPy | Numerical operations |

---

## Project Structure

```text
ai-face-liveness-detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── modules/
│   ├── __init__.py
│   ├── anti_spoof.py
│   ├── blink_detection.py
│   ├── face_landmarker.py
│   ├── head_pose.py
│   ├── image_quality.py
│   └── liveness.py
│
└── models/
    ├── face_landmarker.task
    └── best_model.onnx
```

---

## How It Works

Each webcam frame is processed through this flow:

```text
Camera Frame
    ↓
Face Landmark Detection
    ↓
Image Quality Check
    ↓
Blink / Head Movement Detection
    ↓
Random Challenge Verification
    ↓
Anti-Spoof Model Prediction
    ↓
Final Liveness Decision
```

The final verification condition is:

```python
live = challenge_ok and spoof_ok
```

The user is marked as **LIVE** only when the random challenge is completed and the anti-spoofing model confirms the face is real.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-face-liveness-detection.git
cd ai-face-liveness-detection
```

---

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Your `requirements.txt` should contain:

```txt
streamlit
streamlit-webrtc
opencv-python-headless
mediapipe
onnxruntime
numpy
av
```

---

## Model Setup

Create a folder named:

```text
models/
```

Add these two model files:

```text
models/face_landmarker.task
models/best_model.onnx
```

Expected model paths are defined in `config/settings.py`.

---

## Configuration

Update `config/settings.py`:

```python
FACE_LANDMARKER_MODEL_PATH = "models/face_landmarker.task"

ANTI_SPOOF_MODEL_PATH = "models/best_model.onnx"

ANTI_SPOOF_THRESHOLD = 0.65

EAR_THRESHOLD = 0.22

HEAD_TURN_THRESHOLD = 35

RANDOM_CHALLENGE_COUNT = 3

CHALLENGE_TIMEOUT_SECONDS = 8
```

---

## Run the App

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

Allow webcam permission in the browser.

---

## Usage

1. Start the app.
2. Allow webcam access.
3. Keep your face centered.
4. Make sure the image is clear and well-lit.
5. Follow the random challenge shown on screen.
6. Complete all required challenges.
7. If anti-spoofing passes, the status becomes:

```text
LIVE
```

---

## Possible Status Values

| Status | Meaning |
|---|---|
| `CHECKING` | Verification is in progress |
| `LIVE` | User passed liveness verification |
| `NO_FACE` | No face detected |
| `BAD_IMAGE` | Image quality is poor |
| `SPOOF` | Anti-spoofing detected a fake face |
| `TIMEOUT` | Challenge was not completed in time |
| `ANTI_SPOOF_MISSING` | Anti-spoofing model is missing or failed |

---

## Notes

- Webcam access requires browser permission.
- Good lighting improves detection accuracy.
- The anti-spoofing model must be available for verification.
- If model filenames are different, update the paths in `config/settings.py`.

---

## License

This project is for educational and demonstration purposes.
