# AI Gym Coach: Real-Time Fitness Tracker

**Live Demo:** [Check out the App here!](https://gym-ai-iykbxvqpupvvfpotsiu3bf.streamlit.app)

A high-performance Computer Vision application built with **MediaPipe** and **Streamlit**. This app uses advanced pose estimation to track workout form, count repetitions, and visualize progress against set goals in real-time.

---

##  Key Features

* **Multi-Mode Exercise Tracking:** Specialized logic for **Bicep Curls**, **Squats**, and **Pushups** (Front-facing optimized).
* **Touchless "Shoulder-Touch" Reset:** A custom gesture-based reset—simply cross your arms and touch your opposite shoulders to clear the counter.
* **Visual Goal System:** * Set target reps via a dynamic input.
    * Real-time **Progress Bar** overlay.
    * **Neon Gold Transformation:** The skeletal HUD turns from Neon Green to Gold once the goal is reached.
* **Modern HUD:** Semi-transparent dark UI overlay ensures statistics are readable without blocking the view of your workout form.

---

##  Technical Challenges & Solutions

### 1. Eliminating "Jitter" with Math
**Challenge:** Lite AI models often suffer from "jitter," where the skeleton shakes, causing accidental rep counts.
**Solution:** Implemented a **Low-Pass Filter (Exponential Moving Average)**. By calculating $Angle_{smooth} = (\alpha \cdot Angle_{raw}) + ((1 - \alpha) \cdot Angle_{prev})$, we achieved professional-grade stability even on lower-quality webcams.

### 2. Mobile Optimization
**Challenge:** Desktop browsers often struggled with latency during real-time processing.
**Solution:** Optimized for **Mobile-First usage**. Modern smartphones handle MediaPipe tensors more efficiently, and the UI was designed specifically to remain clear on vertical mobile screens.

---

## How It Works

The app utilizes **MediaPipe Pose** to detect 33 3D landmarks. 



### 1. Angle Calculation
The core logic calculates joint angles using the dot product of normalized vectors between three key landmarks (e.g., Shoulder, Elbow, and Wrist).

### 2. State Machine Logic
Repetitions are tracked using a state machine:
* **DOWN Stage:** Triggered when the joint reaches full extension.
* **UP Stage:** Triggered when the joint reaches peak contraction.
* A rep is only valid once a full **DOWN → UP** cycle is completed.

---

##  Installation & Local Run

1. **Clone & Install:**
   ```bash
   git clone [https://github.com/yourusername/gym-ai-coach.git](https://github.com/yourusername/gym-ai-coach.git)
   cd gym-ai-coach
   pip install streamlit streamlit-webrtc mediapipe opencv-python numpy
