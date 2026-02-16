import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import urllib.request
import os

# --- CONFIGURATION ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose/pose_landmarker/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker.task"

# --- HELPER: AUTO-DOWNLOAD MODEL ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading AI Model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Download Complete!")

# --- GEOMETRY HELPER ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- OPTIMIZATION: Cache the AI Detector ---
@st.cache_resource
def get_detector():
    # 1. Ensure model exists before loading
    download_model()
    
    # 2. Load the model from the LOCAL file
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    return vision.PoseLandmarker.create_from_options(options)

# --- THE PROCESSOR ---
class GymProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = get_detector()
        
        # State variables
        self.counter = 0
        self.stage = "down"

    def transform(self, frame):
        # 1. Convert frame
        img = frame.to_ndarray(format="bgr24")
        
        # 2. Resize to 480p (Save Cloud RAM)
        img = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # 3. Inference
        detection_result = self.detector.detect(mp_image)
        
        # 4. Logic & Drawing
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            
            try:
                # Get Coordinates (Indexes: 12=Shoulder, 14=Elbow, 16=Wrist)
                p1 = [landmarks[12].x, landmarks[12].y]
                p2 = [landmarks[14].x, landmarks[14].y]
                p3 = [landmarks[16].x, landmarks[16].y]
                
                angle = calculate_angle(p1, p2, p3)
                
                # Logic
                if angle > 160: self.stage = "down"
                if angle < 40 and self.stage == 'down':
                    self.stage = "up"
                    self.counter += 1

                # Visualization
                h, w, _ = img.shape
                
                # Draw Box
                cv2.rectangle(img, (0,0), (200, 80), (245,117,16), -1)
                
                # Draw Text
                cv2.putText(img, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, str(self.counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(img, self.stage, (80,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                # Simple Line Drawing (Manual)
                start_point = (int(p1[0]*w), int(p1[1]*h))
                end_point = (int(p2[0]*w), int(p2[1]*h))
                cv2.line(img, start_point, end_point, (255, 255, 255), 4)

                start_point = (int(p2[0]*w), int(p2[1]*h))
                end_point = (int(p3[0]*w), int(p3[1]*h))
                cv2.line(img, start_point, end_point, (255, 255, 255), 4)

            except Exception as e:
                print(f"Error: {e}")

        return img

# --- UI SETUP ---
st.title("Gym AI Lite ⚡")
st.write("Processing in Cloud...")

ctx = webrtc_streamer(
    key="gym-ai", 
    video_transformer_factory=GymProcessor,
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": {"width": 640, "height": 480}},
    async_processing=True,
)