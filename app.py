import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import av

st.set_page_config(page_title="AI Gym Coach", layout="centered")

st.title("🏋️ AI Gym Coach")
st.write("Select your exercise and start the camera.")

# Sidebar for Mode Selection
mode = st.sidebar.radio("Select Exercise:", ["Bicep Curl", "Squat"])
st.sidebar.markdown("---")
st.sidebar.write("### Instructions")
if mode == "Bicep Curl":
    st.sidebar.write("1. Stand sideways.")
    st.sidebar.write("2. Show your right arm.")
    st.sidebar.write("3. Curl up to your shoulder.")
else:
    st.sidebar.write("1. Stand far back.")
    st.sidebar.write("2. Show your whole body.")
    st.sidebar.write("3. Squat until your thighs are parallel.")

# --- 2. THE AI PROCESSOR ---
class GymProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # State Variables
        self.counter = 0
        self.stage = "down"
        self.feedback = "Ready"
        
        # We need to access the mode from the sidebar, but we can't directly.
        # So we default to Curl 
        self.mode = "curl" 

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
        
        # 1. Flip & Convert
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        h, w, _ = img.shape
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            p1, p2, p3 = None, None, None
            
            # 2. Get Coordinates
            
            # Check for Squat (Hip/Knee/Ankle visible?)
            if (landmarks[24].visibility > 0.6 and 
                landmarks[26].visibility > 0.6 and 
                landmarks[28].visibility > 0.6):
                
                # SQUAT LOGIC
                p1 = [landmarks[24].x, landmarks[24].y]
                p2 = [landmarks[26].x, landmarks[26].y]
                p3 = [landmarks[28].x, landmarks[28].y]
                current_mode = "Squat"
            else:
                # CURL LOGIC (Default)
                p1 = [landmarks[12].x, landmarks[12].y]
                p2 = [landmarks[14].x, landmarks[14].y]
                p3 = [landmarks[16].x, landmarks[16].y]
                current_mode = "Curl"

            if p1 and p2 and p3:
                angle = self.calculate_angle(p1, p2, p3)
                
                # 3. Counting Logic
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

                # 4. Draw UI on Video
                # Draw Skeleton
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Draw Box for Stats
                cv2.rectangle(img, (0,0), (250, 80), (245, 117, 16), -1)
                
                # Mode Text
                cv2.putText(img, current_mode.upper(), (10,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                
                # Rep Counter
                cv2.putText(img, str(self.counter), (10,70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
                           
                # Angle Text at joint
                p2_px = tuple(np.multiply(p2, [w, h]).astype(int))
                cv2.putText(img, str(int(angle)), p2_px, 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. STREAMER ---
ctx = webrtc_streamer(
    key="gym-coach",
    video_processor_factory=GymProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)