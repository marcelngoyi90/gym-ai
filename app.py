import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import av

# --- 1. SETUP & LAYOUT ---
st.set_page_config(page_title="AI Gym Coach", layout="centered")

st.title("🏋️ AI Gym Coach")

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("Settings")

# Exercise Selection
# We use a session state to help pass this data to the video processor
mode = st.sidebar.radio("Select Exercise:", ["Bicep Curl", "Squat"])

# Camera Selection
cam_options = {"Front Camera (User)": "user", "Back Camera (Environment)": "environment"}
cam_label = st.sidebar.radio("Select Camera:", list(cam_options.keys()))
facing_mode = cam_options[cam_label]

# Reset Button Logic
if st.sidebar.button("Reset Counter"):
    st.session_state.reset = True
else:
    st.session_state.reset = False

st.sidebar.markdown("---")
st.sidebar.write("### Instructions")
if mode == "Bicep Curl":
    st.sidebar.write("1. Stand sideways.")
    st.sidebar.write("2. Show your arm clearly.")
    st.sidebar.write("3. Curl up to your shoulder.")
else:
    st.sidebar.write("1. Stand back (6-8 ft).")
    st.sidebar.write("2. Show your full body.")
    st.sidebar.write("3. Squat until thighs are parallel.")

# --- 3. THE AI PROCESSOR ---
class GymProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # State Variables
        self.counter = 0
        self.stage = "down"
        self.mode = "Bicep Curl"  # Default
        self.feedback = ""

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    def recv(self, frame):
        # Check for reset command from the UI
        if hasattr(self, 'reset_now') and self.reset_now:
            self.counter = 0
            self.reset_now = False

        img = frame.to_ndarray(format="bgr24")
        
        # 1. Flip & Convert
        # If using front camera, we usually want a mirror effect
        if "user" in str(self.facing_mode): 
            img = cv2.flip(img, 1)
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        h, w, _ = img.shape
        
        # 2. Draw UI Box
        cv2.rectangle(img, (0,0), (250, 90), (245, 117, 16), -1)
        
        # Display Mode & Count
        cv2.putText(img, "MODE", (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(img, self.mode.upper(), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(img, "REPS", (150,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(img, str(self.counter), (140,70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

        # 3. Pose Logic
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            p1, p2, p3 = None, None, None
            
            # --- STRICT MODE SELECTION ---
            if self.mode == "Squat":
                # Hip (24), Knee (26), Ankle (28)
                p1 = [landmarks[24].x, landmarks[24].y]
                p2 = [landmarks[26].x, landmarks[26].y]
                p3 = [landmarks[28].x, landmarks[28].y]
            else:
                # Shoulder (12), Elbow (14), Wrist (16)
                p1 = [landmarks[12].x, landmarks[12].y]
                p2 = [landmarks[14].x, landmarks[14].y]
                p3 = [landmarks[16].x, landmarks[16].y]

            # Calculate & Count
            if p1 and p2 and p3:
                angle = self.calculate_angle(p1, p2, p3)
                
                if self.mode == "Squat":
                    if angle < 90: self.stage = "deep"
                    if angle > 160 and self.stage == 'deep':
                        self.stage = "up"
                        self.counter += 1
                else: # Curl
                    if angle > 160: self.stage = "down"
                    if angle < 30 and self.stage == 'down':
                        self.stage = "up"
                        self.counter += 1

                # Visuals
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Draw Angle at the joint
                p2_px = tuple(np.multiply(p2, [w, h]).astype(int))
                cv2.putText(img, str(int(angle)), p2_px, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. STREAMER ---
ctx = webrtc_streamer(
    key="gym-coach",
    video_processor_factory=GymProcessor,
    media_stream_constraints={"video": {"facingMode": facing_mode}, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- 5. REAL-TIME DATA SYNC ---
# This block sends the sidebar selections into the video processor
if ctx.video_processor:
    # Send the selected mode
    ctx.video_processor.mode = mode
    # Send the camera mode (for flipping logic)
    ctx.video_processor.facing_mode = facing_mode
    
    # Send reset command
    if st.session_state.reset:
        ctx.video_processor.reset_now = True