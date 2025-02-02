import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui  # For simulating key presses and minimizing windows

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()  # Get volume range (usually -65.25 to 0.0)
min_vol, max_vol = volume_range[0], volume_range[1]

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Function to calculate distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)
    

# Function to check if hand is swiping right
def is_swiping_right(landmarks):
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return index_tip.x > wrist.x + 0.2  # Swipe right if index finger is to the right of the wrist

# Function to check if hand is in a closed fist
def is_closed_fist(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Check if all fingertips are close to the palm
    palm = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    distance_threshold = 0.1
    return (calculate_distance(thumb_tip, palm) < distance_threshold and
            calculate_distance(index_tip, palm) < distance_threshold and
            calculate_distance(middle_tip, palm) < distance_threshold and
            calculate_distance(ring_tip, palm) < distance_threshold and
            calculate_distance(pinky_tip, palm) < distance_threshold)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate distance between thumb and index finger
            distance = calculate_distance(thumb_tip, index_tip)

            # Map distance to volume level
            vol = np.interp(distance, [0.05, 0.3], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Display volume level on the frame
            cv2.putText(frame, f"Volume: {int(np.interp(vol, [min_vol, max_vol], [0, 100]))}%",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Check for swipe right gesture (next song)
            if is_swiping_right(hand_landmarks):
                pyautogui.press('volumedown')  # Simulate "Media Next Track" key press
                cv2.putText(frame, "Next Song", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check for closed fist gesture (minimize window)
            if is_closed_fist(hand_landmarks):
                pyautogui.hotkey('win', 'down')  # Minimize the active window
                cv2.putText(frame, "Minimize Window", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Hand Gesture Control", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()