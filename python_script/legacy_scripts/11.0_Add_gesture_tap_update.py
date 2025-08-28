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
        time.sleep(0.1)
        gesture.set("Default")
    except Exception as e:
        print(f"Firebase sending failed. Exception: {e}")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Creating the ThreadPoolExecutor thing
from concurrent.futures import ThreadPoolExecutor
handymen = ThreadPoolExecutor(max_workers = 3)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Threshold variables
TAP_VELOCITY_THRESHOLD = 0.004 # A sharp Z-movement
SWIPE_VELOCITY_THRESHOLD = 14.3784   # A sharp XY-movement
STILLNESS_THRESHOLD_XY = 3.3541    # Max v_xy to be considered "still"
STILLNESS_THRESHOLD_Z = 0.0034   # Max v_z to be considered "still"
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Gesture state variables
v_z = None
v_xy = None
last_index_z = None
last_index_x = None
last_index_y = None
tap_stage_1_flag = False
swipe_start_position = None
swipe_start_time = None
swipe_start_wrist_position = None
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Check tap function
def check_tap():
    global tap_stage_1_flag
    if not tap_stage_1_flag:
        tap_stage_1_flag = True
        return
    handymen.submit(set_firebase_gesture, "tap")
    tap_stage_1_flag = False
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Check swipe function
# def check_swipe(wrist_x, wrist_y, kalman_index_x, kalman_index_y):
#     global swipe_start_position, swipe_start_wrist_position, swipe_start_time
#     if swipe_start_position == None:
#         swipe_start_position = (kalman_index_x, kalman_index_y)
#         swipe_start_wrist_position = (wrist_x, wrist_y)
#         swipe_start_time = time.time()
#     else:
#         elapsed_time = time.time() - swipe_start_time
#         if elapsed_time > 0.2:
#             start_x, start_y = swipe_start_position
#             start_wrist_x, start_wrist_y = swipe_start_wrist_position

#             movement_x = kalman_index_x - start_x
#             movement_y = kalman_index_y - start_y
#             movement_wrist = ((wrist_x - start_wrist_x)**2 + (wrist_y - start_wrist_y)**2)**0.5

#             if (movement_wrist < 12):
#                 if (abs(movement_y) > 50) and (abs(movement_y) > abs(movement_x)):
#                     handymen.submit(set_firebase_gesture, "swipe_up" if movement_y < 0 else "swipe_down")
#                 elif (abs(movement_x) > 50) and (abs(movement_x) > abs(movement_y)):
#                     handymen.submit(set_firebase_gesture, "swipe_left" if movement_x > 0 else "swipe_right")
            
#             swipe_start_position = (kalman_index_x, kalman_index_y)
#             swipe_start_time = time.time()
#             swipe_start_wrist_position = (wrist_x, wrist_y)
def check_swipe():
    v_x = kalman_index.statePost[2, 0]
    v_y = kalman_index.statePost[3, 0]

    if abs(v_y) > abs(v_x):
        handymen.submit(set_firebase_gesture, "swipe_up" if v_y < 0 else "swipe_down")
    elif abs(v_x) > abs(v_y):
        handymen.submit(set_firebase_gesture, "swipe_left" if v_x > 0 else "swipe_right")
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
        kalman_index_flag = False   # Reset
        v_z = v_xy = None      # Reset velocities when no hand is found, ofc
        last_index_z = last_index_x = last_index_y = None
    else:
        # This is the case where a hand is detectedd by the detect_async() method above
        image_height, image_width = frame.shape[:2]

        # for looping over all the landmarks in the Hand_landmarks_containing_object with hand_idx being the 0 based index in case of mulitple
        # hands and hand being the object containing the landmarks.
        for hand_idx, hand in enumerate(Hand_landmarks_containing_object.hand_landmarks):
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
            if last_index_z is None and last_index_x is None and last_index_y is None:
                # If last values of the coords are None.
                last_index_x = kalman_index_x
                last_index_y = kalman_index_y
                last_index_z = index_z

                # For the first frame that the hands exist, there is no calculation to be done and hence the loop must go on.
                reader.circle(frame, (kalman_index_x, kalman_index_y), 4, (255, 255, 255), -1)
                reader.circle(frame, (wrist_x, wrist_y), 4, (255, 255, 255), -1)
                continue
            else:
                v_z = index_z - last_index_z
                v_xy = ((kalman_index_x - last_index_x)**2 + (kalman_index_y - last_index_y)**2)**0.5
                last_index_x = kalman_index_x
                last_index_y = kalman_index_y
                last_index_z = index_z
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


            # Velocity comparison and consequent gesture category distinction
            v_z = abs(v_z)
            if v_z >= TAP_VELOCITY_THRESHOLD and v_xy <= STILLNESS_THRESHOLD_XY:
                check_tap()
            elif v_xy >= SWIPE_VELOCITY_THRESHOLD and v_z <= STILLNESS_THRESHOLD_Z:
                check_swipe()
            else:
                pass
            
            reader.circle(frame, (kalman_index_x, kalman_index_y), 4, (255, 255, 255), -1)
            reader.circle(frame, (wrist_x, wrist_y), 4, (255, 255, 255), -1)
    
    
    # Doing the window things
    reader.imshow("Feed", frame)
    if reader.waitKey(1) == ord('q'):
        break
    #-------------------------------------------------------------------------------------------------------------------


camera.release()
reader.destroyAllWindows()
handymen.shutdown()