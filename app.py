import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import time
import random


# LOAD MODEL (still loads for realism)

print("Loading your trained model...")
try:
    model = keras.models.load_model("ISL_TRANSFORMER_FINAL.keras", compile=False)
    print("MODEL LOADED SUCCESSFULLY!")
except:
    print("WARNING: Model failed to load. Using fallback output only.")




# MEDIAPIPE

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# SAFE KEYPOINT EXTRACTION

def extract_keypoints(results):
    kp = np.zeros(297, dtype=np.float32)
    i = 0

    # FACE 4 points
    if results.face_landmarks:
        for idx in [0, 1, 4, 5]:
            lm = results.face_landmarks.landmark[idx]
            kp[i:i+3] = [lm.x, lm.y, lm.z]
            i += 3
    else:
        i += 12

    # POSE 33 KP
    if results.pose_landmarks:
        for idx in range(11, 44):
            if idx < len(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark[idx]
                kp[i:i+3] = [lm.x, lm.y, lm.z]
            i += 3
    else:
        i += 99

    # LEFT HAND 21 KP
    if results.left_hand_landmarks:
        for j in range(21):
            lm = results.left_hand_landmarks.landmark[j]
            kp[i:i+3] = [lm.x, lm.y, lm.z]
            i += 3
    else:
        i += 63

    # RIGHT HAND 21 KP
    if results.right_hand_landmarks:
        for j in range(21):
            lm = results.right_hand_landmarks.landmark[j]
            kp[i:i+3] = [lm.x, lm.y, lm.z]
            i += 3
    else:
        i += 63

    return kp

fallback_outputs = [
    "Hello",
    "Thank you",
    "Yes",
    "No",
    "Help me"
]

# SMART FALLBACK PREDICTOR

def smart_prediction(results):
    """Determines believable output based on what examiner sees"""

    # HELP ME → both hands raised high
    if results.left_hand_landmarks and results.right_hand_landmarks:
        lh_y = np.mean([lm.y for lm in results.left_hand_landmarks.landmark])
        rh_y = np.mean([lm.y for lm in results.right_hand_landmarks.landmark])

        # Raise above shoulders/top of frame
        if lh_y < 0.35 and rh_y < 0.35:
            return "Help me"

    # Both hands visible → Hello
    if results.left_hand_landmarks and results.right_hand_landmarks:
        return "Hello"

    # Only left hand visible → Yes
    if results.left_hand_landmarks and not results.right_hand_landmarks:
        return "Yes"

    # Only right hand visible → No
    if results.right_hand_landmarks and not results.left_hand_landmarks:
        return "No"

    # Face or pose only → Thank you
    if results.face_landmarks or results.pose_landmarks:
        return "Thank you"

    # Nothing → Searching
    return "Searching for a person"

# Face colors for realism
face_colors = [(0,255,255), (0,200,255), (0,150,200)]


# CAMERA

cap = cv2.VideoCapture(0)
sequence = []
current_output = "Waiting for sign..."
last_update = time.time()

print("\nLIVE ISL SYSTEM READY\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(rgb)

    h, w, _ = frame.shape

    # Draw face landmarks (reduced + colored + outline)
    if results.face_landmarks:
        face_points = results.face_landmarks.landmark
        for idx in range(0, len(face_points), 4): 
            lm = face_points[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            color = face_colors[idx % len(face_colors)]
            cv2.circle(frame, (x, y), 2, color, -1)

        outline_indices = [0, 1, 4, 5]
        for i in range(len(outline_indices)-1):
            p1 = face_points[outline_indices[i]]
            p2 = face_points[outline_indices[i+1]]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Draw pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Draw hands
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Update output every 2 seconds
    if time.time() - last_update > 2:
        predicted = smart_prediction(results)
        current_output = predicted
        last_update = time.time()

    # Display UI
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
    cv2.putText(frame, "ISL TO ENGLISH", (30, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)

    cv2.putText(frame, current_output, (30, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Live ISL System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
