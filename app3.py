import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6)
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

# Function to check if eyes are closed
def are_eyes_closed(face_landmarks, frame_shape):
    # Indices for eye landmarks in MediaPipe Face Mesh
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Right eye landmarks

    # Get the landmarks for left and right eyes
    left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE_INDICES]
    right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE_INDICES]

    # Calculate the vertical distance between upper and lower eyelids
    def eye_aspect_ratio(eye):
        vertical_dist1 = calculate_distance(eye[1], eye[5])
        vertical_dist2 = calculate_distance(eye[2], eye[4])
        horizontal_dist = calculate_distance(eye[0], eye[3])
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    # Average EAR for both eyes
    avg_ear = (left_ear + right_ear) / 2.0

    # Display EAR on the frame for debugging
    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # If the average eye aspect ratio is below a threshold, eyes are closed
    ear_threshold = 0.3   # Adjust this value based on your environment
    return avg_ear < ear_threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands and Face Mesh
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    # Eye control for volume
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            if are_eyes_closed(face_landmarks, frame.shape):
                # Decrease volume if eyes are closed
                current_vol = volume.GetMasterVolumeLevel()
                new_vol = max(min_vol, current_vol - 1.0)  # Decrease by 1.0 dB
                volume.SetMasterVolumeLevel(new_vol, None)
                cv2.putText(frame, "Eyes Closed: Volume Down", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Increase volume if eyes are open
                current_vol = volume.GetMasterVolumeLevel()
                new_vol = min(max_vol, current_vol + 1.0)  # Increase by 1.0 dB
                volume.SetMasterVolumeLevel(new_vol, None)
                cv2.putText(frame, "Eyes Open: Volume Up", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hand control for volume (existing functionality)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
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

    # Display the frame
    cv2.imshow("Hand and Eye Control", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()