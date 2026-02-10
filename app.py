import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import av

# --- 1. SETUP ---
st.set_page_config(page_title="AI Gym Coach", layout="centered")

st.title("🏋️ AI Gym Coach")

# --- 2. CONTROLS ---
# Dropdown to select exercise
mode = st.radio("Select Exercise:", ["Bicep Curl", "Squat"], horizontal=True)

# Dropdown to select Camera Type (Fixes mobile selection issue)
cam_type = st.radio("Select Camera:", ["Front Camera (User)", "Back Camera (Environment)"], horizontal=True)

# Set the constraints based on selection
if cam_type == "Front Camera (User)":
    # "user" means front-facing camera
    constraints = {"video": {"facingMode": "user"}, "audio": False}
else:
    # "environment" means back-facing camera
    constraints = {"video": {"facingMode": "environment"}, "audio": False}

st.write("---")

# --- 3. THE AI PROCESSOR ---
class GymProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.counter = 0
        self.stage = "down"
        # We default to curl, but this will be updated by the sidebar logic in a real app
        # For this simple demo, we will detect legs vs arms visibility
        
    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Flip only if using front camera
        # We can't easily check cam_type here, so we default to flipping (selfie mode)
        img = cv2.flip(img, 1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        h, w, _ = img.shape
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            p1, p2, p3 = None, None, None
            
            # AUTO-DETECT MODE based on visibility
            # If legs are visible -> Squat Mode
            if (landmarks[24].visibility > 0.6 and 
                landmarks[26].visibility > 0.6 and 
                landmarks[28].visibility > 0.6):
                
                current_mode = "Squat"
                p1 = [landmarks[24].x, landmarks[24].y]
                p2 = [landmarks[26].x, landmarks[26].y]
                p3 = [landmarks[28].x, landmarks[28].y]
            else:
                current_mode = "Curl"
                p1 = [landmarks[12].x, landmarks[12].y]
                p2 = [landmarks[14].x, landmarks[14].y]
                p3 = [landmarks[16].x, landmarks[16].y]

            if p1 and p2 and p3:
                angle = self.calculate_angle(p1, p2, p3)
                
                # Logic
                if current_mode == "Squat":
                    if angle < 90: self.stage = "deep"
                    if angle > 160 and self.stage == 'deep':
                        self.stage = "up"
                        self.counter += 1
                else: # Curl
                    if angle > 160: self.stage = "down"
                    if angle < 30 and self.stage == 'down':
                        self.stage = "up"
                        self.counter += 1

                # Draw
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Stats Box
                cv2.rectangle(img, (0,0), (200, 80), (245, 117, 16), -1)
                cv2.putText(img, current_mode.upper(), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                cv2.putText(img, str(self.counter), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
                
                # Angle
                p2_px = tuple(np.multiply(p2, [w, h]).astype(int))
                cv2.putText(img, str(int(angle)), p2_px, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. STREAMER ---
ctx = webrtc_streamer(
    key="gym-coach",
    video_processor_factory=GymProcessor,
    # This is the magic line that fixes mobile cameras:
    media_stream_constraints=constraints,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)