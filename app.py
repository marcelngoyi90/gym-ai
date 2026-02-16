import cv2
import av
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import math

# --- INITIALIZE MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

NEON_GREEN = (20, 255, 57)
NEON_GOLD = (0, 215, 255) # For Goal Achievement

# --- MATH HELPERS ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

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
        self.goal = 10 # Default goal
        self.smooth_angle = 0
        self.alpha = 0.8 
        self.reset_cooldown = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1280, 720))
        h, w, _ = img.shape
        
        overlay = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # --- RESET LOGIC ---
                l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                if get_distance(r_wrist, l_shoulder) < 0.12 and get_distance(l_wrist, r_shoulder) < 0.12 and self.reset_cooldown == 0:
                    self.counter = 0
                    self.reset_cooldown = 50 
                
                if self.reset_cooldown > 0: self.reset_cooldown -= 1

                # --- EXERCISE LOGIC ---
                p_shoulder = [l_shoulder.x, l_shoulder.y]
                p_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                p_wrist = [l_wrist.x, l_wrist.y]
                p_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                p_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                p_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                if self.mode in ["curl", "pushup"]:
                    raw_angle = calculate_angle(p_shoulder, p_elbow, p_wrist)
                else:
                    raw_angle = calculate_angle(p_hip, p_knee, p_ankle)
                
                self.smooth_angle = (self.alpha * raw_angle) + ((1 - self.alpha) * self.smooth_angle)

                if self.mode == "pushup":
                    if self.smooth_angle < 95: self.stage = "down"
                    if self.smooth_angle > 155 and self.stage == "down":
                        self.stage = "up"; self.counter += 1
                elif self.mode == "curl":
                    if self.smooth_angle > 160: self.stage = "down"
                    if self.smooth_angle < 40 and self.stage == 'down':
                        self.stage = "up"; self.counter += 1
                elif self.mode == "squat":
                    if self.smooth_angle < 80: self.stage = "down"
                    if self.smooth_angle > 160 and self.stage == 'down':
                        self.stage = "up"; self.counter += 1

                # --- NEON DRAWING ---
                color = NEON_GOLD if self.counter >= self.goal else NEON_GREEN
                mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=4),
                    mp_drawing.DrawingSpec(color=color, thickness=3)
                )
                
                # --- UI OVERLAY ---
                cv2.rectangle(overlay, (0,0), (450, 150), (0,0,0), -1)
                
                # Progress Bar calculation
                progress = min(self.counter / self.goal, 1.0)
                bar_width = int(progress * 400)
                cv2.rectangle(overlay, (20, 120), (420, 140), (50, 50, 50), -1) # Background bar
                cv2.rectangle(overlay, (20, 120), (20 + bar_width, 140), color, -1) # Active bar

                cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
                
                # Labels
                cv2.putText(img, f"GOAL: {self.goal}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(img, f"REPS: {self.counter}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,255,255), 4)
                
                if self.counter >= self.goal:
                    cv2.putText(img, "GOAL ACHIEVED!", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, NEON_GOLD, 5)

                stage_text_color = color if self.stage == "up" else (50, 220, 255)
                cv2.putText(img, self.stage.upper(), (280,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, stage_text_color, 3)

            except Exception as e:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")

col1, col2 = st.columns([3, 1])
with col1:
    ex_col, goal_col = st.columns([2, 1])
    with ex_col:
        mode_selection = st.radio("EXERCISE", ["Bicep Curl", "Squat", "Pushup"], horizontal=True)
    with goal_col:
        goal_selection = st.number_input("SET GOAL", min_value=1, max_value=100, value=10)

with col2:
    st.markdown("### Controls")
    st.info("🙅‍♂️ Touch shoulders to reset")
    reset_btn = st.button("Manual Reset", use_container_width=True)

ctx = webrtc_streamer(
    key="gym-ai-goal", 
    video_processor_factory=GymProcessor,
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.mode = mode_selection.lower().replace("bicep ", "")
    ctx.video_processor.goal = goal_selection
    if reset_btn:
        ctx.video_processor.counter = 0