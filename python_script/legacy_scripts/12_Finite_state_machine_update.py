import cv2 as reader
import mediapipe
import time
import numpy

# Initialising the camera object
camera = reader.VideoCapture(0)
camera.set(3, 720)
camera.set(4, 405)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Creating the CLAHE filter object
clahe = reader.createCLAHE(clipLimit = 2, tileGridSize = (10, 10))
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Defining the callback function for the LIVE_STREAM mode
Hand_landmarks_containing_object = None

def resultcallback(result, output_img, timestamp_ms):
    global Hand_landmarks_containing_object
    Hand_landmarks_containing_object = result

# Initialising the mediapipe handlandmarker stuff
BaseOptions = mediapipe.tasks.BaseOptions
HandLandmarker = mediapipe.tasks.vision.HandLandmarker
HandLandmarkerOptions = mediapipe.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mediapipe.tasks.vision.RunningMode

modelpath = r"C:\Coding\Python stuff\Hand_project\hand_landmarker.task"
options = HandLandmarkerOptions(base_options = BaseOptions(model_asset_path = modelpath),
                                running_mode = VisionRunningMode.LIVE_STREAM,
                                result_callback = resultcallback,
                                num_hands = 1)
landmarker = HandLandmarker.create_from_options(options)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Function to create a Kalman Filter
def create_kalman_filter():
    """Creates a standard Kalman filter for 2D point tracking."""
    kf = reader.KalmanFilter(4, 2)
    kf.measurementMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0]], numpy.float32)
    kf.transitionMatrix = numpy.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], numpy.float32)
    kf.processNoiseCov = numpy.eye(4, dtype=numpy.float32) * 0.05  # Q: Trust in prediction
    kf.measurementNoiseCov = numpy.eye(2, dtype=numpy.float32) * 0.07 # R: Trust in measurement
    return kf

# Defining a Kalman object for the index
kalman_index_flag = False
kalman_index = create_kalman_filter()
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Firebase setup and functions
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate(r"C:\Coding\Python stuff\Hand_project\firebase_key.json")
firebase_admin.initialize_app(cred, {'databaseURL': "https://hand-tracking-gesture-control-default-rtdb.asia-southeast1.firebasedatabase.app/"})

gesture = db.reference("Pookie")
anchor = db.reference("Pookie_anchor")

def set_firebase_gesture(gesture_to_set):
    try:
        gesture.set(gesture_to_set)
        time.sleep(0.3)
        gesture.set("Default")
    except Exception as e:
        print(f"Firebase sending failed. Exception: {e}")
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Velocity calculation variables
last = None
velocities = None
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Threshold variables
TAP_VELOCITY_THRESHOLD = 0.0113 # A sharp Z-movement
SWIPE_VELOCITY_THRESHOLD = 14.3784   # A sharp XY-movement
STILLNESS_THRESHOLD_XY = 3.3541    # Max v_xy to be considered "still"
STILLNESS_THRESHOLD_Z = 0.0034   # Max v_z to be considered "still"
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# FSM variables
listt = []
FSM_curr_pos = None
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Think and call
def think(listt):   #listt[0] is the start postion tuple and list[1] is the end position tuple
    start = listt[0]
    end = listt[1]
    movement_x = end[0] - start[0]
    movement_y = end[1] - start[1]
    movement_xy = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
    movement_z = end[2] - start[2]

    if movement_z > TAP_VELOCITY_THRESHOLD and movement_xy < SWIPE_VELOCITY_THRESHOLD:
        handymen.submit(set_firebase_gesture, "tap")
    elif movement_xy > 30:
        if abs(movement_y) > abs(movement_x):
            handymen.submit(set_firebase_gesture, "swipe_up" if movement_y < 0 else "swipe_down")
        else:
            handymen.submit(set_firebase_gesture, "swipe_right" if movement_x < 0 else "swipe_left")
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Creating the ThreadPoolExecutor thing
from concurrent.futures import ThreadPoolExecutor
handymen = ThreadPoolExecutor(max_workers = 3)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Main loop
while True:
    flag, frame = camera.read()

    if not flag:
        continue

    # Converting the frames (and applying the CLAHE filter on it) 
    frame_YCrCb = reader.cvtColor(frame, reader.COLOR_BGR2YCR_CB)
    frame_YCrCb[:, :, 0] = clahe.apply(frame_YCrCb[:, :, 0])
    frame_RGB = reader.cvtColor(frame_YCrCb, reader.COLOR_YCR_CB2RGB)
    mediapipe_img = mediapipe.Image(image_format = mediapipe.ImageFormat.SRGB, data = frame_RGB)
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # Calling the detection function using the landmarker object.
    # If the detect_async() detects a hand it updates the Hand_landmarks_containing_object with the latest value of all those landmarks else throws an 
    # Exception.
    timestamp = (int)(time.time() * 1000)
    try:
        landmarker.detect_async(mediapipe_img, timestamp)
    except Exception:
        print(f"Detection failed, {Exception}")
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # Proceeding onto hand evaluation
    if not Hand_landmarks_containing_object or not Hand_landmarks_containing_object.hand_landmarks:
        # This is the case when no hand is detected by the detect_async() method above
        last = None
    else:
        # This is the case where a hand is detectedd by the detect_async() method above
        image_height, image_width = frame.shape[:2]

        # for looping over all the landmarks in the Hand_landmarks_containing_object with hand_idx being the 0 based index in case of mulitple
        # hands and hand being the object containing the landmarks.
        for hand_idx, hand in enumerate(Hand_landmarks_containing_object.hand_landmarks):
            # Finger position calculation
            index = hand[8]
            index_x = (int)(index.x * image_width)
            index_y = (int)(index.y * image_height)
            index_z = index.z
            wrist = hand[0]
            wrist_x = (int)(wrist.x * image_width)
            wrist_y = (int)(wrist.y * image_height)

            # Applying the kalman filter on index and storing kalmanised coords in new variables
            if not kalman_index_flag:
                kalman_index.statePost = numpy.array([[index_x], [index_y], [0], [0]], dtype = numpy.float32)
                kalman_index_flag = True
            else:
                kalman_index.predict()
                measurement = numpy.array([[index_x], [index_y]], dtype = numpy.float32)
                kalman_index.correct(measurement)
            
            kalman_index_x = (int)(kalman_index.statePost[0, 0])    # Kalmanised index_x
            kalman_index_y = (int)(kalman_index.statePost[1, 0])    # Kalmanised index_y
            
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


            # Velocity calculation
            if last is None:
                last = ((kalman_index_x, kalman_index_y, index_z), (wrist_x, wrist_y))   #last is a tuple of tuples, last.first is the index's positions and last.second is the wrist's
                continue

            vx = index_x - last[0][0]
            vy = index_y - last[0][1]
            vz = index_z - last[0][2]
            vw = ((wrist_x - last[1][0])**2 + (wrist_y - last[1][1])**2)**0.5
            vxy = (vx**2 + vy**2)**0.5
            velocities = (vx, vy, vz, vxy, vw)
            last = ((kalman_index_x, kalman_index_y, index_z), (wrist_x, wrist_y))
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


            # FSM
            if velocities[4] > 3.7:
                reader.putText(frame, "Hand_too_fast", (50, 20), reader.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, reader.LINE_AA)
                continue
            if velocities[3] > SWIPE_VELOCITY_THRESHOLD:
                if len(listt) == 0: 
                    listt.append((kalman_index_x, kalman_index_y, index_z))
                FSM_curr_pos = (kalman_index_x, kalman_index_y, index_z)
                reader.putText(frame, "State: Gesture", (50, 50), reader.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, reader.LINE_AA)
                continue
            elif abs(velocities[2]) > TAP_VELOCITY_THRESHOLD:
                if len(listt) == 0: 
                    listt.append((kalman_index_x, kalman_index_y, index_z))
                FSM_curr_pos = (kalman_index_x, kalman_index_y, index_z)
                reader.putText(frame, "State: Gesture, Reason: Tap", (50, 50), reader.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, reader.LINE_AA)
                continue
            else:
                if len(listt) > 0:
                    listt.append(FSM_curr_pos)
                    think(listt)
                    listt.clear()
                reader.putText(frame, "State: Listening", (50, 50), reader.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, reader.LINE_AA)
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


            reader.circle(frame, (kalman_index_x, kalman_index_y), 4, (255, 255, 255), -1)
            reader.circle(frame, (wrist_x, wrist_y), 4, (255, 255, 255), -1)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # Doing the window things
    reader.imshow("Feed", frame)
    if reader.waitKey(1) == ord('q'):
        break
    #-------------------------------------------------------------------------------------------------------------------


camera.release()
reader.destroyAllWindows()
handymen.shutdown()