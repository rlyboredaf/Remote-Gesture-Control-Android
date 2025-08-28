import cv2 as reader
import mediapipe
import numpy
import time

# Firebase integration
import firebase_admin
from firebase_admin import credentials, db

# Initialising the firebase app using credentials
cred = credentials.Certificate(r"C:\Coding\Python stuff\Hand_project\firebase_key.json")
firebase_admin.initialize_app(cred, {'databaseURL': "https://hand-tracking-gesture-control-default-rtdb.asia-southeast1.firebasedatabase.app/"})

# Creating a gesture node
sms = db.reference("Pookie")


# Live_stream mode stuff
latest_frame = None
latest_result = None

def result_callback (result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


# Setting up the camera
camera = reader.VideoCapture(0)
camera.set(3, 480)
camera.set(4, 360)


# Initialising stuff to make life easier
BaseOptions = mediapipe.tasks.BaseOptions
HandLandmarker = mediapipe.tasks.vision.HandLandmarker
HandLandmarkerOptions = mediapipe.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mediapipe.tasks.vision.RunningMode
mp_hands = mediapipe.solutions.hands

# Creating a handlandmarker instance with LIVE_STREAM mode
model_path = r"C:\Coding\Python stuff\Hand_project\hand_landmarker.task"
options = HandLandmarkerOptions(base_options = BaseOptions(model_asset_path = model_path),
                                running_mode = VisionRunningMode.LIVE_STREAM,
                                result_callback = result_callback,
                                num_hands = 1)
landmarker = HandLandmarker.create_from_options(options)


while True:
    flag, snapshot = camera.read()
    if not flag:
        continue

    # Converting snapshot from BGR to RGB
    snapshot_RGB = reader.cvtColor(snapshot, reader.COLOR_BGR2RGB)
    mediapipe_img = mediapipe.Image(image_format = mediapipe.ImageFormat.SRGB,
                                    data = snapshot_RGB)
    
    timestamp = (int)(time.time() * 1000)
    latest_frame = snapshot.copy()

    try:
        landmarker.detect_async(mediapipe_img, timestamp)
    except Exception as e:
        print(f"Detection error: {e}")

    # To make sure that there is no segmentation fault because of accessing latest_result when it is None
    if not latest_result or not latest_result.hand_landmarks:
        reader.imshow("LIVE_FEED", latest_frame)
        if reader.waitKey(1) == ord('q'):
            break
        continue

    # Gesture sending
    image_height, image_width = latest_frame.shape[:2]
    for hand_idx, hand in enumerate(latest_result.hand_landmarks):
        thumb = hand[4]
        index = hand[8]
        thumb_x = int(thumb.x * image_width)
        thumb_y = int(thumb.y * image_height)
        index_x = int(index.x * image_width)
        index_y = int(index.y * image_height)

        pinch_distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5

        if pinch_distance < 40:
            sms.set("pinched")
        else:
            sms.set("unpinched")

    
    # Doing the window things
    reader.imshow("LIVE_FEED", latest_frame)
    if reader.waitKey(1) == ord('q'):
        break  

camera.release()
reader.destroyAllWindows()