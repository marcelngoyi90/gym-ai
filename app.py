import cv2
import av
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- INITIALIZE MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- MATH HELPER ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- THE GYM PROCESSOR ---
class GymProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1 
        )
        self.counter = 0
        self.stage = "up"
        self.mode = "curl"
        self.smooth_angle = 0
        self.alpha = 0.8 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1280, 720))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # Get points for the left side (standardized)
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # --- CALC ANGLE ---
                if self.mode == "curl" or self.mode == "pushup":
                    raw_angle = calculate_angle(shoulder, elbow, wrist)
                else: # Squat
                    raw_angle = calculate_angle(hip, knee, ankle)
                
                self.smooth_angle = (self.alpha * raw_angle) + ((1 - self.alpha) * self.smooth_angle)

                # --- FRONT-FACING PUSHUP LOGIC ---
                if self.mode == "pushup":
                    # When facing the camera, elbows flare out. 
                    # 'Down' is usually around 70-90 degrees.
                    # 'Up' is 150+ degrees.
                    if self.smooth_angle < 90:
                        self.stage = "down"
                    if self.smooth_angle > 150 and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1

                elif self.mode == "curl":
                    if self.smooth_angle > 160: self.stage = "down"
                    if self.smooth_angle < 40 and self.stage == 'down':
                        self.stage = "up"; self.counter += 1

                elif self.mode == "squat":
                    if self.smooth_angle < 80: self.stage = "down"
                    if self.smooth_angle > 160 and self.stage == 'down':
                        self.stage = "up"; self.counter += 1

                # --- DRAWING ---
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Visual UI
                cv2.rectangle(img, (0,0), (450, 120), (245,117,16), -1)
                cv2.putText(img, f"MODE: {self.mode.upper()}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(img, f"REPS: {self.counter}", (20,95), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
                
                # Color feedback for the stage
                stage_color = (0, 255, 0) if self.stage == "up" else (0, 255, 255)
                cv2.putText(img, self.stage.upper(), (280,95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, stage_color, 2, cv2.LINE_AA)

            except Exception as e:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI SETUP ---
st.set_page_config(page_title="Gym AI Pro", layout="wide")
st.title("Gym AI Coach: Front-Facing Enabled 🏋️‍♂️")

col1, col2 = st.columns([2, 1])
with col1:
    mode_selection = st.radio("Select Exercise:", ["Bicep Curl", "Squat", "Pushup"], horizontal=True)
with col2:
    st.write(""); st.write("") 
    reset_btn = st.button("Reset Counter", type="primary")

ctx = webrtc_streamer(
    key="gym-ai-front", 
    video_processor_factory=GymProcessor,
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.mode = mode_selection.lower().replace("bicep ", "")
    if reset_btn:
        ctx.video_processor.counter = 0