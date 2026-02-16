import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os

# --- CONFIGURATION ---
MODEL_PATH = "pose_landmarker.task"

# --- GEOMETRY HELPER ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- CACHED DETECTOR ---
@st.cache_resource
def get_detector():
    # Check if the file is actually there
    if not os.path.exists(MODEL_PATH):
        return None

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    return vision.PoseLandmarker.create_from_options(options)

# --- THE PROCESSOR ---
class GymProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = get_detector()
        self.counter = 0
        self.stage = "down"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Resize (Optimization)
        img = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # 2. Check if model loaded correctly
        if self.detector is None:
            cv2.putText(img, "Error: Model File Missing", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # 3. Inference
        detection_result = self.detector.detect(mp_image)
        
        # 4. Logic & Drawing
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            try:
                # 12=Shoulder, 14=Elbow, 16=Wrist
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
                cv2.rectangle(img, (0,0), (200, 80), (245,117,16), -1)
                cv2.putText(img, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, str(self.counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(img, self.stage, (80,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                # Manual Draw
                start_point = (int(p1[0]*w), int(p1[1]*h))
                end_point = (int(p2[0]*w), int(p2[1]*h))
                cv2.line(img, start_point, end_point, (255, 255, 255), 4)

                start_point = (int(p2[0]*w), int(p2[1]*h))
                end_point = (int(p3[0]*w), int(p3[1]*h))
                cv2.line(img, start_point, end_point, (255, 255, 255), 4)

            except Exception as e:
                print(f"Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI SETUP ---
import av # Needed for new video processor type
st.title("Gym AI Lite ⚡")

if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ FILE MISSING: Please upload '{MODEL_PATH}' to your GitHub repository.")
else:
    # START THE STREAM
    ctx = webrtc_streamer(
        key="gym-ai", 
        video_processor_factory=GymProcessor,  # <--- UPDATED NAME
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": {"width": 640, "height": 480}},
        async_processing=True,
        
    )
