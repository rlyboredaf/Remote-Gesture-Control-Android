import cv2 as reader
import mediapipe as mp
import numpy as np
import os

camera = reader.VideoCapture(0)
camera.set(3, 1080)
camera.set(4, 720)

# Initialising stuff to make life easier
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_hands = mp.solutions.hands
# Create a hand landmarker instance with the image mode:
model_path = r"C:\Coding\Python stuff\Hand_project\hand_landmarker.task"
options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path = model_path),    
                                running_mode=VisionRunningMode.IMAGE,
                                num_hands = 2)

# Video with Image Operations
            # Creating a namespace landmarker
with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        flag, snapshot = camera.read()                                                 # Taking a frame
        if flag:
            # Image Operations
            snapshot_RGB = reader.cvtColor(snapshot, reader.COLOR_BGR2RGB)              # Converting the frame into RGB

            mediapipe_img = mp.Image(mp.ImageFormat.SRGB, snapshot_RGB)
            hand_landmarker_result = landmarker.detect(mediapipe_img)

            if hand_landmarker_result:
                # print(hand_landmarker_result)

                if len(hand_landmarker_result.hand_landmarks) > 0:                      # To check that there exists atleast one hand
                    # Hand[0]
                    first_hand = hand_landmarker_result.hand_landmarks[0]
                    for landmark in first_hand:                                         # Green Bubbles
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
                    
                    if len(hand_landmarker_result.hand_landmarks) > 1:
                        # Hand[1]
                        second_hand = hand_landmarker_result.hand_landmarks[1]
                        for landmark in second_hand:
                            x = int(landmark.x * snapshot_RGB.shape[1])
                            y = int(landmark.y * snapshot_RGB.shape[0])
                            reader.circle(snapshot_RGB, (x, y), 5, (0, 255, 0), -1)

                        for s_idx, e_idx in mp_hands.HAND_CONNECTIONS:
                            start = hand_landmarker_result.hand_landmarks[1][s_idx]
                            end = hand_landmarker_result.hand_landmarks[1][e_idx]

                            x_start = int(start.x * snapshot_RGB.shape[1])                  
                            y_start = int(start.y * snapshot_RGB.shape[0])
                            x_end = int(end.x * snapshot_RGB.shape[1])
                            y_end = int(end.y * snapshot_RGB.shape[0])

                            reader.line(snapshot_RGB, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

                bgr_img = reader.cvtColor(snapshot_RGB, reader.COLOR_RGB2BGR)
                reader.imshow("Video", bgr_img)
                if reader.waitKey(1) == ord('q'):
                    break  


# # Firebase imports
# import firebase_admin
# from firebase_admin import credentials, db

# cred = credentials.Certificate("hand-tracking-gesture-control-firebase-adminsdk-fbsvc-0fda341d63.json")  # 🔁 Your downloaded file
# firebase_admin.initialize_app(cred, {'databaseURL': 'https://hand-tracking-gesture-control-default-rtdb.asia-southeast1.firebasedatabase.app/'})

# # Reference to the "gesture" key
# gesture_ref = db.reference('gesture')

# # Send a gesture
# gesture_ref.set('10')  # Change this to 'tap', 'swipe_left', etc.

# print("Gesture sent!")