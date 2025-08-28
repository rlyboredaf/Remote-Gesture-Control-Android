import cv2 as reader
import mediapipe
import numpy
import time
import threading
from concurrent.futures import ThreadPoolExecutor

#--------------------------------------------------------------------------------------------------------------------


# Firebase integration
import firebase_admin
from firebase_admin import credentials, db

# Initialising the firebase app using credentials
cred = credentials.Certificate(r"C:\Coding\Python stuff\Hand_project\firebase_key.json")
firebase_admin.initialize_app(cred, {'databaseURL': "https://hand-tracking-gesture-control-default-rtdb.asia-southeast1.firebasedatabase.app/"})

# Creating a gesture node
sms = db.reference("Pookie")
anchor = db.reference("Pookie_anchor")
#--------------------------------------------------------------------------------------------------------------------


# Live_stream mode stuff
latest_frame = None
latest_result = None

def result_callback (result, output_image, timestamp_ms):
    global latest_result
    latest_result = result
#--------------------------------------------------------------------------------------------------------------------


# Setting up the camera
camera = reader.VideoCapture(0)
camera.set(3, 480)
camera.set(4, 360)
#--------------------------------------------------------------------------------------------------------------------


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
#--------------------------------------------------------------------------------------------------------------------


# FPS counter initialisation
fps_start_time = time.time()
frame_count = 0
fps = 0
#--------------------------------------------------------------------------------------------------------------------


# Kalman filter stuff
def create_kalman_filter():
    """Creates a standard Kalman filter for 2D point tracking."""
    kf = reader.KalmanFilter(4, 2)
    kf.measurementMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0]], numpy.float32)
    kf.transitionMatrix = numpy.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], numpy.float32)
    kf.processNoiseCov = numpy.eye(4, dtype=numpy.float32) * 0.05  # Q: Trust in prediction
    kf.measurementNoiseCov = numpy.eye(2, dtype=numpy.float32) * 1 # R: Trust in measurement
    return kf

kalman_index = create_kalman_filter()
index_initialized = False
#--------------------------------------------------------------------------------------------------------------------


# CLAHE filter creating
clahe = reader.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#-------------------------------------------------------------------------------------------------------------------

# State Management for Gestures
last_sent_status_swipe = "no_gesture"
current_status_swipe = "no_gesture"

# Swipe Detection variables
swipe_start_pos = None
swipe_start_time = None

def set_firebase_status(status_to_send):
    """Function to be run in a separate thread to avoid blocking."""
    try:
        sms.set(status_to_send)
        # print(f"Sent status to Firebase: {status_to_send}")
        time.sleep(0.1)
        sms.set("Default")
    except Exception as e:
        print(f"Firebase update to gesture failed: {e}")
#-------------------------------------------------------------------------------------------------------------------


# State management for anchor
last_sent_anchor_position = None


def set_firebase_anchor(anchor_to_send):
    """Function to be run in a separate thread to avoid blocking."""
    try:
        anchor.set(anchor_to_send)
        # print(f"Anchor coords sent to firebase: {anchor_to_send}")
    except Exception as e:
        print(f"Firebase update to anchor failed: {e}")
#-------------------------------------------------------------------------------------------------------------------

# Creating the ThreadPoolExecutor thing
handymen = ThreadPoolExecutor(max_workers = 2)


# Main loop
while True:
    flag, snapshot = camera.read()
    if not flag:
        continue
    #-------------------------------------------------------------------------------------------------------------------


    # Converting image from BGR to YCrCb for CLAHE
    snapshot_YCrCb = reader.cvtColor(snapshot, reader.COLOR_BGR2YCrCb)
    snapshot_YCrCb[:, :, 0] = clahe.apply(snapshot_YCrCb[:, :, 0])
    enhanced_frame = reader.cvtColor(snapshot_YCrCb, reader.COLOR_YCrCb2BGR)
    snapshot_RGB = reader.cvtColor(enhanced_frame, reader.COLOR_BGR2RGB)
    mediapipe_img = mediapipe.Image(image_format = mediapipe.ImageFormat.SRGB, data = snapshot_RGB)
    #-------------------------------------------------------------------------------------------------------------------


    timestamp = (int)(time.time() * 1000)
    latest_frame = snapshot.copy()
    #-------------------------------------------------------------------------------------------------------------------


    try:
        landmarker.detect_async(mediapipe_img, timestamp)
    except Exception as e:
        print(f"Detection error: {e}")
    #-------------------------------------------------------------------------------------------------------------------

    current_status_swipe = "no_gesture" # Default status for each frame

    # To make sure that there is no segmentation fault because of accessing latest_result when it is None
    # If no Hand then {if:} or if hand then {else:}
    if not latest_result or not latest_result.hand_landmarks:
        index_initialized = False
        # Reset swipe tracking if hand is lost
        swipe_start_pos = None 
        # Reset anchor position
        last_sent_anchor_position = None
    else:                                                                                              # If you find hand
        # Gesture sending
        image_height, image_width = latest_frame.shape[:2]
        for hand_idx, hand in enumerate(latest_result.hand_landmarks):
            index = hand[8]
            wrist = hand[0]
            index_x = int(index.x * image_width)
            index_y = int(index.y * image_height)
            wrist_x = int(wrist.x * image_width)
            wrist_y = int(wrist.y * image_height)
            #-------------------------------------------------------------------------------------------------------------------


            # Kalmanising the index's position
            if not index_initialized:
                kalman_index.statePost = numpy.array([[index_x], [index_y], [0], [0]], dtype = numpy.float32)
                index_initialized = True
            else:
                kalman_index.predict()
                measurement = numpy.array([[index_x], [index_y]], dtype = numpy.float32)
                kalman_index.correct(measurement)
            
            smooth_index_x = int(kalman_index.statePost[0, 0])
            smooth_index_y = int(kalman_index.statePost[1, 0])
            #-------------------------------------------------------------------------------------------------------------------
            
            
            # Sending the wrist coords
            current_anchor_position = {'x' : wrist.x, 'y' : wrist.y}
            try:
                if last_sent_anchor_position is None or ((current_anchor_position['x'] - last_sent_anchor_position['x'])**2 + (current_anchor_position['y'] - last_sent_anchor_position['y'])**2)**0.5 > 0.02:
                    last_sent_anchor_position = current_anchor_position
                    handymen.submit(set_firebase_anchor, current_anchor_position)
            except Exception as e:
                print(f"Exception : {e}")
            #-------------------------------------------------------------------------------------------------------------------


            # Swipe checking logic
            if swipe_start_pos is None:
                swipe_start_pos = (smooth_index_x, smooth_index_y)
                swipe_start_wrist_pos = (wrist_x, wrist_y)
                swipe_start_time = time.time()
            else:
                time_elapsed = time.time() - swipe_start_time
                if time_elapsed > 0.2:
                    start_x, start_y = swipe_start_pos
                    start_wrist_x, start_wrist_y = swipe_start_wrist_pos
                    
                    delta_x = smooth_index_x - start_x
                    delta_y = smooth_index_y - start_y
                    
                    wrist_movement_distance = ((wrist_x - start_wrist_x)**2 + (wrist_y - start_wrist_y)**2)**0.5

                    # Check for vertical swipes, only if the wrist is relatively still
                    if abs(delta_y) > 50 and abs(delta_y) > abs(delta_x) and wrist_movement_distance <= 12:
                        if delta_y < 0:
                            current_status_swipe = "swipe_up"
                        else:
                            current_status_swipe = "swipe_down"
                    
                    # Check for horizontal swipes, only if the wrist is relatively still
                    elif abs(delta_x) > 50 and abs(delta_x) > abs(delta_y) and wrist_movement_distance <= 12:
                        if delta_x > 0:
                            current_status_swipe = "swipe_left"
                        else:
                            current_status_swipe = "swipe_right"
                    
                    # Reset for the next swipe detection window
                    swipe_start_pos = (smooth_index_x, smooth_index_y)
                    swipe_start_wrist_pos = (wrist_x, wrist_y)
                    swipe_start_time = time.time()
            
            reader.circle(latest_frame, (smooth_index_x, smooth_index_y), 4, (0, 255, 0), -1)
            reader.circle(latest_frame, (wrist_x, wrist_y), 4, (0, 255, 0), -1)
    #-------------------------------------------------------------------------------------------------------------------


    # Threading: Send to Firebase only if status has changed
    if current_status_swipe != last_sent_status_swipe:
        last_sent_status_swipe = current_status_swipe
        if current_status_swipe != "no_gesture": # Only send actual gestures
            handymen.submit(set_firebase_status, current_status_swipe)
    #-------------------------------------------------------------------------------------------------------------------


    # FPS calculation
    frame_count += 1
    if (time.time() - fps_start_time) > 1: # Update FPS every second
        fps = frame_count / (time.time() - fps_start_time)
        frame_count = 0
        fps_start_time = time.time()
    reader.putText(latest_frame, f'FPS: {int(fps)}', (10, 30), reader.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    #-------------------------------------------------------------------------------------------------------------------


    # Doing the window things
    reader.imshow("LIVE_FEED", latest_frame)
    if reader.waitKey(1) == ord('q'):
        break
    #-------------------------------------------------------------------------------------------------------------------


camera.release()
reader.destroyAllWindows()