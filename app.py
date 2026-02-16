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
# We use the FULL model now because your laptop can handle it!
MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"

# --- 1. DOWNLOADER (Full Model) ---
def force_download_model():
    # Check if file exists. If it's small (Lite model), delete it and get the Big one.
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) < 5000000: # Full model is > 5MB
            print("⚠️ Lite model detected. Upgrading to Full Model...")
            os.remove(MODEL_PATH)
        else:
            return
    print(f"📥 Downloading Full AI Model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Download Complete!")
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
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5) # Higher confidence for stability
    return vision.PoseLandmarker.create_from_options(options)

# --- 4. THE GYM PROCESSOR (High Performance) ---
class GymProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = get_detector()
        self.counter = 0
        self.stage = "down"
        self.mode = "curl"
        self.prev_angle = 0

    def recv(self, frame):
        # High Quality Input
        img = frame.to_ndarray(format="bgr24")
        
        # 720p Resolution (Your laptop can handle this easily)
        img = cv2.resize(img, (1280, 720))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        if self.detector is None: return av.VideoFrame.from_ndarray(img, format="bgr24")

        detection_result = self.detector.detect(mp_image)
        
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            try:
                p1, p2, p3 = None, None, None
                
                if self.mode == "curl":
                    p1 = [landmarks[12].x, landmarks[12].y]
                    p2 = [landmarks[14].x, landmarks[14].y]
                    p3 = [landmarks[16].x, landmarks[16].y]
                    angle = calculate_angle(p1, p2, p3)
                    
                    # Logic
                    if angle > 160: self.stage = "down"
                    if angle < 40 and self.stage == 'down':
                        self.stage = "up"
                        self.counter += 1

                elif self.mode == "squat":
                    p1 = [landmarks[24].x, landmarks[24].y]
                    p2 = [landmarks[26].x, landmarks[26].y]
                    p3 = [landmarks[28].x, landmarks[28].y]
                    angle = calculate_angle(p1, p2, p3)
                    
                    # Logic
                    if angle < 75: self.stage = "down"
                    if angle > 160 and self.stage == 'down':
                        self.stage = "up"
                        self.counter += 1

                # Drawing (High Res)
                h, w, _ = img.shape
                cv2.rectangle(img, (0,0), (350, 100), (245,117,16), -1)
                cv2.putText(img, self.mode.upper(), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(img, str(self.counter), (20,85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv2.LINE_AA)
                cv2.putText(img, self.stage, (150,85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)

                if p1 and p2 and p3:
                    start = (int(p1[0]*w), int(p1[1]*h))
                    mid = (int(p2[0]*w), int(p2[1]*h))
                    end = (int(p3[0]*w), int(p3[1]*h))
                    cv2.line(img, start, mid, (255, 255, 255), 6)
                    cv2.line(img, mid, end, (255, 255, 255), 6)
                    cv2.circle(img, mid, 15, (0, 0, 255), -1)

            except Exception as e:
                print(f"Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. UI SETUP ---
st.set_page_config(page_title="Gym AI Pro", layout="wide")
st.title("Gym AI Coach Pro 🏋️‍♂️")

col1, col2 = st.columns([2, 1])
with col1:
    mode_selection = st.radio("Select Exercise:", ["Bicep Curl", "Squat"], horizontal=True)
with col2:
    st.write("") 
    st.write("") 
    reset_btn = st.button("Reset Counter", type="primary")

# Local Network Config (We don't need complex STUN servers for local)
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

ctx = webrtc_streamer(
    key="gym-ai", 
    video_processor_factory=GymProcessor,
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"width": 1280, "height": 720}, # HD Quality
        "audio": False,
    },
    async_processing=True,
)

if ctx.video_processor:
    if mode_selection == "Bicep Curl":
        ctx.video_processor.mode = "curl"
    else:
        ctx.video_processor.mode = "squat"
    if reset_btn:
        ctx.video_processor.counter = 0
        ctx.video_processor.stage = "down"