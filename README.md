# Gesture-Controlled Android System 🖐️➡️📱

This project allows you to remotely control an Android device using hand gestures captured by a webcam. It uses a Python script for computer vision and a Firebase Realtime Database to send commands to an Android accessibility service.

## Key Features

* **Real-time Hand Tracking**: Utilizes MediaPipe for accurate and fast hand landmark detection.
* **Gesture Recognition**: Recognizes distinct gestures like taps and directional swipes.
* **Wireless Control**: Sends commands over the internet via Firebase for real-time control.
* **On-screen Cursor**: Displays a visual cursor on the Android device for feedback.

## Tech Stack

* **Python Script**:
    * OpenCV
    * MediaPipe
    * Firebase Admin SDK
* **Android App**:
    * Java/Kotlin
    * Android Accessibility Service
    * Firebase Realtime Database SDK

## Setup and Installation

### Python Script

1.  Clone the repository: `git clone https://github.com/your-username/your-repo-name.git`
2.  Navigate to the `python_script` directory.
3.  Install the required packages: `pip install opencv-python mediapipe firebase-admin`
4.  **IMPORTANT**: Download your Firebase Admin SDK private key. Rename it to `firebase_key.json` and place it inside the `python_script` folder.

### Android App

1.  Open the `android_app` folder in Android Studio.
2.  Let Gradle sync the project.
3.  Run the app on a physical device.
4.  Enable the Accessibility Service for the app when prompted in your device's settings.

## Usage

1.  Run the Python script on your computer. A webcam feed should appear.
2.  Open the Android app on your phone.
3.  Perform the recognized gestures in front of the webcam to control your device.


P.S. This README was AI generated with a summary I gave.
