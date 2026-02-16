import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- OPTIMIZATION 1: Cache the AI Model ---
# This prevents the app from reloading the model 30 times a second
@st.cache_resource
def get_pose_model():
    mp_pose = mp.solutions.pose
    # Reduced complexity to 0 (fastest) for cloud
    return mp_pose.Pose(
        model_complexity=0, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

# --- Geometry Helper ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- The Processor Class ---
class GymProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = get_pose_model()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # State variables
        self.counter = 0
        self.stage = "down"
        self.feedback = "Start"

    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # --- OPTIMIZATION 2: Resize Frame ---
        # Force 480p to save memory bandwidth
        img = cv2.resize(img, (640, 480))

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get Coordinates (Curl Mode)
            # You can add the switch logic here later
            try:
                p1 = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                p2 = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                p3 = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                angle = calculate_angle(p1, p2, p3)
                
                # Logic
                if angle > 160: self.stage = "down"
                if angle < 40 and self.stage == 'down':
                    self.stage = "up"
                    self.counter += 1
                    self.feedback = "Good Rep!"
                
                # Visualize
                # Draw Box
                cv2.rectangle(img, (0,0), (200, 80), (245,117,16), -1)
                
                # Rep Data
                cv2.putText(img, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, str(self.counter), (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Feedback
                cv2.putText(img, self.stage, (80,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
            except:
                pass

        return img

# --- Streamlit UI ---
st.title("Gym AI Lite ⚡")
st.write("Memory-Optimized for Cloud")

# WebRTC Streamer
ctx = webrtc_streamer(
    key="gym-ai", 
    video_transformer_factory=GymProcessor,
    mode=WebRtcMode.SENDRECV,
    # --- OPTIMIZATION 3: Limit Input Resolution ---
    media_stream_constraints={"video": {"width": 640, "height": 480}},
    async_processing=True,
)