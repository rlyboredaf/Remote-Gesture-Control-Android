import cv2 as reader
import mediapipe
import numpy
import time
import threading

#--------------------------------------------------------------------------------------------------------------------


# Firebase integration
import firebase_admin
from firebase_admin import credentials, db

# Initialising the firebase app using credentials
cred = credentials.Certificate(r"C:\Coding\Python stuff\Hand_project\firebase_key.json")
firebase_admin.initialize_app(cred, {'databaseURL': "https://hand-tracking-gesture-control-default-rtdb.asia-southeast1.firebasedatabase.app/"})

# Creating a gesture node
sms = db.reference("Pookie")
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
start_time = time.time()
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

kalman_thumb = create_kalman_filter()
kalman_index = create_kalman_filter()

# First frame flag initialisation
thumb_initialized = False
index_initialized = False
#--------------------------------------------------------------------------------------------------------------------


# CLAHE filter creating
clahe = reader.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#-------------------------------------------------------------------------------------------------------------------

# Threading: ---- State Management for Firebase ----
last_sent_status_pinch = None
current_status_pinch = "unpinched"

def set_firebase_status(status_to_send):
    """Function to be run in a separate thread to avoid blocking."""
    try:
        sms.set(status_to_send)
    except Exception as e:
        print(f"Firebase update failed: {e}")


# Main loop
while True:
    # FPS: Start timer for FPS calculation
    loop_start_time = time.time()
    
    flag, snapshot = camera.read()
    if not flag:
        continue

    # Converting image from BGR to YCrCb for CLAHE
    snapshot_YCrCb = reader.cvtColor(snapshot, reader.COLOR_BGR2YCrCb)

    #Applying CLAHE to Y (Luma) channel
    snapshot_YCrCb[:, :, 0] = clahe.apply(snapshot_YCrCb[:, :, 0])

    # Converting enhanced image back to BGR
    enhanced_frame = reader.cvtColor(snapshot_YCrCb, reader.COLOR_YCrCb2BGR)

    # Use the enhanced frame for mediapipe and display
    snapshot_RGB = reader.cvtColor(enhanced_frame, reader.COLOR_BGR2RGB)
    mediapipe_img = mediapipe.Image(image_format = mediapipe.ImageFormat.SRGB,
                                      data = snapshot_RGB)

    timestamp = (int)(time.time() * 1000)
    latest_frame = enhanced_frame.copy()

    try:
        landmarker.detect_async(mediapipe_img, timestamp)
    except Exception as e:
        print(f"Detection error: {e}")

    # To make sure that there is no segmentation fault because of accessing latest_result when it is None
    # In short when there's no hand
    if not latest_result or not latest_result.hand_landmarks:
        index_initialized = False
        thumb_initialized = False
        current_status_pinch = "unpinched"
    else:
        # Gesture sending
        image_height, image_width = latest_frame.shape[:2]
        for hand_idx, hand in enumerate(latest_result.hand_landmarks):
            thumb = hand[4]
            index = hand[8]
            thumb_x = int(thumb.x * image_width)
            thumb_y = int(thumb.y * image_height)
            index_x = int(index.x * image_width)
            index_y = int(index.y * image_height)

            # Safety for first frame or when thumb(Hand) isnt detected.
            if not thumb_initialized:
                kalman_thumb.statePost = numpy.array([[thumb_x], [thumb_y], [0], [0]], dtype=numpy.float32)
                thumb_initialized = True
            else:
                kalman_thumb.predict()
                measurement = numpy.array([[thumb_x], [thumb_y]], dtype = numpy.float32)
                kalman_thumb.correct(measurement)
            
            # Ctrl C + Ctrl V for index
            if not index_initialized:
                kalman_index.statePost = numpy.array([[index_x], [index_y], [0], [0]], dtype = numpy.float32)
                index_initialized = True
            else:
                kalman_index.predict()
                measurement = numpy.array([[index_x], [index_y]], dtype = numpy.float32)
                kalman_index.correct(measurement)
            
            smooth_thumb_x = int(kalman_thumb.statePost[0, 0])
            smooth_thumb_y = int(kalman_thumb.statePost[1, 0])
            smooth_index_x = int(kalman_index.statePost[0, 0])
            smooth_index_y = int(kalman_index.statePost[1, 0])

            pinch_distance = ((smooth_index_x - smooth_thumb_x) ** 2 + (smooth_index_y - smooth_thumb_y) ** 2) ** 0.5

            if pinch_distance < 10:
                current_status_pinch = "pinched"
            else:
                current_status_pinch = "unpinched"

    # Threading: ---- Send to Firebase only if status has changed ----
    if current_status_pinch != last_sent_status_pinch:
        last_sent_status_pinch = current_status_pinch
        # Create and start a new thread for the network call
        firebase_thread = threading.Thread(target=set_firebase_status, args=(current_status_pinch,))
        firebase_thread.daemon = True # Allows main program to exit even if thread is running
        firebase_thread.start()

    # FPS: ---- FPS Calculation and Display ----
    frame_count += 1
    if (time.time() - start_time) > 1: # Update FPS every second
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()

    # Display FPS on the frame
    reader.putText(latest_frame, f'FPS: {int(fps)}', (5, 15), reader.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Doing the window things
    reader.imshow("LIVE_FEED", latest_frame)
    if reader.waitKey(1) == ord('q'):
        break  

camera.release()
reader.destroyAllWindows()