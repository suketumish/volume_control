# 🎵 Hand Gesture Volume Control

## 📌 Overview
This Python application allows users to control the system volume using hand gestures. By detecting the distance between the thumb and index finger using the **MediaPipe** library, the script dynamically adjusts the volume. Perfect for controlling YouTube videos or any other media playback hands-free! 🎶🖐️

## 🚀 Features
- 🖐️ **Hand Gesture Detection** using **MediaPipe**
- 🔊 **System Volume Control** with **Pycaw**
- 🎥 **Real-time Webcam Processing** via **OpenCV**
- 📏 **Dynamic Volume Adjustment** based on thumb & index finger distance
- ✅ **User-Friendly and Intuitive** controls

## 🛠️ Technologies Used
- **Python** 🐍
- **OpenCV** 📸
- **MediaPipe** 🖐️
- **Pycaw** 🎵
- **NumPy** 🔢

## 📥 Installation
Make sure you have Python installed, then run the following command to install the required libraries:

```bash
pip install mediapipe opencv-python numpy pycaw
```

## 🎯 How It Works
1. **Hand Detection**: The script detects hand landmarks using **MediaPipe**.
2. **Volume Control**: It calculates the distance between the **thumb tip** and **index finger tip**.
3. **Gesture Mapping**:
   - ✊ **Thumb & index close** ➝ Decrease volume 🔉
   - 🖐️ **Thumb & index apart** ➝ Increase volume 🔊
4. **YouTube Integration**: Since it adjusts **system volume**, it affects any media playing, including YouTube videos.

## 📌 Usage
Run the script:
```bash
python hand_volume_control.py
```
- Ensure your webcam is working 📷
- Move your hand in front of the camera to adjust volume 📶
- Press **'q'** to quit ❌

## 🔮 Future Enhancements
🔹 Add more gestures (e.g., **fist for mute**, **open palm for max volume**).
🔹 Integrate with specific applications instead of **system-wide control**.
🔹 Develop a **GUI interface** for enhanced user experience.

---

🌟 Enjoy hands-free volume control and make your media experience more interactive! 🚀🎶
