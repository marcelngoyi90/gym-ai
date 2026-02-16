import cv2
import av
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os
import urllib.request

# --- CONFIGURATION ---
MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

# --- 1. DOWNLOADER ---
def force_download_model():
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) < 4000000:  
            os.remove(MODEL_PATH)
        else:
            return
    print(f"📥 Downloading fresh model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to download model: {e}")

# --- 2. MATH HELPER ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 3. AI LOADER ---
@st.cache_resource
def get_detector():
    force_download_model()
    if not os.path.exists(MODEL_PATH): return None
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    return vision.PoseLandmarker.create_from_options(options)

# --- 4. THE GYM PROCESSOR ---
class GymProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = get_detector()
        self.counter = 0
        self.stage = "down"
        self.mode = "curl"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Optimize resolution for cloud performance
        img = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        if self.detector is None: return av.VideoFrame.from_ndarray(img, format="bgr24")

        detection_result = self.detector.detect(mp_image)
        
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            try:
                p1, p2, p3 = None, None, None
                
                # Logic
                if self.mode == "curl":
                    p1 = [landmarks[12].x, landmarks[12].y]
                    p2 = [landmarks[14].x, landmarks[14].y]
                    p3 = [landmarks[16].x, landmarks[16].y]
                    angle = calculate_angle(p1, p2, p3)
                    
                    if angle > 160: self.stage = "down"
                    if angle < 40 and self.stage == 'down':
                        self.stage = "up"
                        self.counter += 1

                elif self.mode == "squat":
                    p1 = [landmarks[24].x, landmarks[24].y]
                    p2 = [landmarks[26].x, landmarks[26].y]
                    p3 = [landmarks[28].x, landmarks[28].y]
                    angle = calculate_angle(p1, p2, p3)
                    
                    if angle < 75: self.stage = "down" 
                    if angle > 160 and self.stage == 'down':
                        self.stage = "up"
                        self.counter += 1

                # Drawing
                h, w, _ = img.shape
                cv2.rectangle(img, (0,0), (250, 80), (245,117,16), -1)
                cv2.putText(img, self.mode.upper(), (10,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, str(self.counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(img, self.stage, (100,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                if p1 and p2 and p3:
                    start = (int(p1[0]*w), int(p1[1]*h))
                    mid = (int(p2[0]*w), int(p2[1]*h))
                    end = (int(p3[0]*w), int(p3[1]*h))
                    cv2.line(img, start, mid, (255, 255, 255), 4)
                    cv2.line(img, mid, end, (255, 255, 255), 4)

            except Exception as e:
                print(f"Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Gym AI", layout="centered")
st.title("Gym AI Coach 🏋️")

col1, col2 = st.columns([2, 1])
with col1:
    mode_selection = st.radio("Select Exercise:", ["Bicep Curl", "Squat"], horizontal=True)
with col2:
    st.write("") 
    st.write("") 
    reset_btn = st.button("Reset Counter", type="primary")

# --- NETWORK CONFIGURATION (FIXED) ---
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
}

# Start Stream
ctx = webrtc_streamer(
    key="gym-ai", 
    video_processor_factory=GymProcessor,
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            # We request a lower resolution to keep the network fast
            "width": {"min": 480, "ideal": 640, "max": 640},
            "height": {"min": 360, "ideal": 480, "max": 480},
        },
        "audio": False, # Keep audio disabled to prevent crashes
    },
    async_processing=True,
)

# --- 6. BRIDGE ---
if ctx.video_processor:
    if mode_selection == "Bicep Curl":
        ctx.video_processor.mode = "curl"
    else:
        ctx.video_processor.mode = "squat"
        
    if reset_btn:
        ctx.video_processor.counter = 0
        ctx.video_processor.stage = "down"