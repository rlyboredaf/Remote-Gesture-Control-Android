import cv2 as reader
import mediapipe as mp
import numpy as np
import os

camera = reader.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

# Initialising stuff to make life easier
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_hands = mp.solutions.hands
# Create a hand landmarker instance with the image mode:
model_path = r"C:\Coding\Python stuff\Python 3.10.11\Hand project 2\hand_landmarker.task"
options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path = model_path),    running_mode=VisionRunningMode.IMAGE)


# Image operations
_, snapshot = camera.read()                                                 # Taking a frame
snapshot_RGB = reader.cvtColor(snapshot, reader.COLOR_BGR2RGB)              # Converting the frame into RGB

# Creating a namespace landmarker
with HandLandmarker.create_from_options(options) as landmarker:
    mediapipe_img = mp.Image(mp.ImageFormat.SRGB, snapshot_RGB)
    hand_landmarker_result = landmarker.detect(mediapipe_img)

    if hand_landmarker_result:
        # print(hand_landmarker_result)

        for landmark in hand_landmarker_result.hand_landmarks[0]:           # Green Bubbles
            x = int(landmark.x * snapshot_RGB.shape[1])                     # shape[1] is columns in the numpy array => width
            y = int(landmark.y * snapshot_RGB.shape[0])                     # shape[0] is rows => height
            reader.circle(snapshot_RGB, (x, y), 5, (0, 255, 0), -1)         # circle() makes a circle with arguments: Destination, Co-ords, Colour, Setting[-1 => Fill]

        for s_idx, e_idx in mp_hands.HAND_CONNECTIONS:                      # Red Lines
            start = hand_landmarker_result.hand_landmarks[0][s_idx]         # hand_landmarks[0][] gives the tuple array for co-ords of landmarks for hand indexed 0
            end = hand_landmarker_result.hand_landmarks[0][e_idx]           # hand_landmarks[0][s_idx and e_idx] are used to mark the start and end points for the current line

            x_start = int(start.x * snapshot_RGB.shape[1])                  
            y_start = int(start.y * snapshot_RGB.shape[0])
            x_end = int(end.x * snapshot_RGB.shape[1])
            y_end = int(end.y * snapshot_RGB.shape[0])

            reader.line(snapshot_RGB, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)   # line() makes a line with arguments: Destination, start Co-ords, end Co-ords, colour, Pixel width

        bgr_img = reader.cvtColor(snapshot_RGB, reader.COLOR_RGB2BGR)
        reader.imshow("Final output", bgr_img)
        reader.waitKey(0)