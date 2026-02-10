import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import av

# --- 1. SETUP ---
st.set_page_config(page_title="AI Gym Coach", layout="centered")

st.title("🏋️ AI Gym Coach")

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("Settings")

# Exercise Selection
mode = st.sidebar.radio("Select Exercise:", ["Bicep Curl", "Squat"])

# Camera Selection
cam_options = {"Front Camera (User)": "user", "Back Camera (Environment)": "environment"}
cam_label = st.sidebar.radio("Select Camera:", list(cam_options.keys()))
facing_mode = cam_options[cam_label]

# Reset Button
if st.sidebar.button("Reset Counter"):
    if "reset_trigger" not in st.session_state:
        st.session_state.reset_trigger = True
    else:
        st.session_state.reset_trigger = True

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
        
        # Variables
        self.counter = 0
        self.stage = "down"
        self.mode = "Bicep Curl"  # Default, will be updated by UI
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
        img = frame.to_ndarray(format="bgr24")
        
        # Flip image if using front camera (Mirror effect)
        # We assume "user" mode usually needs flipping
        # You can toggle this if the back camera feels backwards
        # img = cv2.flip(img, 1) 
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        h, w, _ = img.shape
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            p1, p2, p3 = None, None, None
            
            # STRICT MODE SELECTION (No Auto-Detect)
            if self.mode == "Squat":
                # Hip, Knee, Ankle
                p1 = [landmarks[24].x, landmarks[24].y]
                p2 = [landmarks[26].x, landmarks[26].y]
                p3 = [landmarks[28].x, landmarks[28].y]
            else:
                # Shoulder, Elbow, Wrist
                p1 = [landmarks[12].x, landmarks[12].y]
                p2 = [landmarks[14].x, landmarks[14].y]
                p3 = [landmarks[16].x, landmarks[16].y]

            # Calculate Angle
            if p1 and p2 and p3:
                angle = self.calculate_angle(p1, p2, p3)
                
                # COUNTING LOGIC
                if self.mode == "Squat":
                    if angle < 90: 
                        self.stage = "deep"
                        self.feedback = "Good Depth!"
                    if angle > 160 and self.stage == 'deep':
                        self.stage = "up"
                        self.counter += 1
                        self.feedback = "Up!"
                else: # Curl
                    if angle > 160: 
                        self.stage = "down"
                        self.feedback = "Stretch"
                    if angle < 30 and self.stage == 'down':
                        self.stage = "up"
                        self.counter += 1
                        self.feedback = "Squeeze!"

                # VISUALS
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Stats Box
                cv2.rectangle(img, (0,0), (250, 80), (245, 117, 16), -1)
                
                # Reps
                cv2.putText(img, "REPS", (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, str(self.counter), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                
                # Mode
                cv2.putText(img, "MODE", (90,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, self.mode, (90,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                
                # Angle at joint
                p2_px = tuple(np.multiply(p2, [w, h]).astype(int))
                cv2.putText(img, str(int(angle)), p2_px, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. STREAMER CONFIG ---
ctx = webrtc_streamer(
    key="gym-coach",
    video_processor_factory=GymProcessor,
    media_stream_constraints={"video": {"facingMode": facing_mode}, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- 5. DATA SYNC (The Magic Part) ---
# This updates the video processor with the sidebar settings in real-time
if ctx.video_processor:
    ctx.video_processor.mode = mode
    
    # Check if reset button was pressed
    if "reset_trigger" in st.session_state and st.session_state.reset_trigger:
        ctx.video_processor.counter = 0
        st.session_state.reset_trigger = False