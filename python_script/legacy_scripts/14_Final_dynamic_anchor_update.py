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

# Defining a Kalman object for the index tip
kalman_index_flag = False
kalman_index = create_kalman_filter()
# Defining a Kalman object for the middle tip
kalman_middle_flag = False
kalman_middle = create_kalman_filter()
# -------------------------------------------   ---------------------------------------------------------------------------------------------------------------------------------------


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
        print(f"Firebase gesture sending failed. Exception: {e}")

# State variable for anchor
last_anchor_position = None

def set_anchor_position(anchor_position):
    try:
        anchor.set(anchor_position)
    except Exception as e:
        print(f"Firebase anchor sending failed. Exception: {e}")    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Velocity calculation variables
last_frame_coordinates = None
velocities = None
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Check pose function
def check_pose(hand):
    is_index_up = hand[8].y < hand[6].y
    is_middle_up = hand[12].y < hand[10].y

    if is_index_up and is_middle_up:
        return "TAP_MODE"
    elif not is_middle_up:
        return "SWIPE_MODE"
    return "NO_MODE"
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Threshold Values
TAP_VELOCITY_THRESHOLD = 0.01 # A sharp Z-movement
SWIPE_VELOCITY_THRESHOLD = 14.3784   # A sharp XY-movement
SWIPE_EVALUATION_DELAY = 0.1    # 100ms
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Creating the ThreadPoolExecutor thing
from concurrent.futures import ThreadPoolExecutor
handymen = ThreadPoolExecutor(max_workers = 3)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# FSM state variables
swipe_start_data = None
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
    # If the detect_async() detects a hand it updates the Hand_landmarks_containing_object with the latest value of all those landmarks else throws an Exception.
    timestamp = (int)(time.time() * 1000)
    try:
        landmarker.detect_async(mediapipe_img, timestamp)
    except Exception as e:
        print(f"Detection failed, {e}")
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    if not Hand_landmarks_containing_object or not Hand_landmarks_containing_object.hand_landmarks:
        # No hand was detected.
        # The velocity calculation variables need to be reset
        last_frame_coordinates = None
        velocities = None
        # The swipe data needs to be reset
        swipe_start_data = None
        # ALl the kalman flags need to be reset
        kalman_index_flag = False
        kalman_middle_flag = False
        # The anchor coordinates need to be reset
        last_anchor_position = None
    else:
        # A hand wass detectedd by the detect_async() method above
        image_height, image_width = frame.shape[:2]

        # for looping over all the landmarks in the Hand_landmarks_containing_object with hand_idx being the 0 based index in case of mulitple hands and hand being the object containing the landmarks.
        for hand_idx, hand in enumerate(Hand_landmarks_containing_object.hand_landmarks):
            # Finger position calculation
            index = hand[8]
            index_x = (int)(index.x * image_width)
            index_y = (int)(index.y * image_height)
            index_z = index.z
            middle = hand[12]
            middle_x = (int)(middle.x * image_width)
            middle_y = (int)(middle.y * image_height)
            middle_z = middle.z
            wrist = hand[0]
            wrist_x = (int)(wrist.x * image_width)
            wrist_y = (int)(wrist.y * image_height)
            index_MCP = hand[5]
            index_MCP_x = (int)(index_MCP.x * image_width)
            index_MCP_y = (int)(index_MCP.y * image_height) 
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


            # Applying the kalman filter on index and storing kalmanised coords
            if not kalman_index_flag:
                kalman_index.statePost = numpy.array([[index_x], [index_y], [0], [0]], dtype = numpy.float32)
                kalman_index_flag = True
            else:
                kalman_index.predict()
                measurement = numpy.array([[index_x], [index_y]], dtype = numpy.float32)
                kalman_index.correct(measurement)
            
            index_x = (int)(kalman_index.statePost[0, 0])    # Kalmanised index_x
            index_y = (int)(kalman_index.statePost[1, 0])    # Kalmanised index_y

            # Applying the kalman filter on middle and storing kalmanised coords
            if not kalman_middle_flag:
                kalman_middle.statePost = numpy.array([[middle_x], [middle_y], [0], [0]], dtype = numpy.float32)
                kalman_middle_flag = True
            else:
                kalman_middle.predict()
                measurement = numpy.array([[middle_x], [middle_y]], dtype = numpy.float32)
                kalman_middle.correct(measurement)
            
            middle_x = (int)(kalman_middle.statePost[0, 0])
            middle_y = (int)(kalman_middle.statePost[1, 0])
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

            
            # Sending anchor coords if the anchor moves by 2%
            anchor_position = {'x' : index_MCP.x, 'y' : index_MCP.y, 'timestamp': int(time.time() * 1000)}
            if last_anchor_position is None or (((anchor_position['x'] - last_anchor_position['x'])**2 + (anchor_position['y'] - last_anchor_position['y'])**2 )**0.5 > 0.02):
                last_anchor_position = anchor_position
                try:
                    handymen.submit(set_anchor_position, anchor_position)
                except Exception as e:
                    print(f"Handymen submit error for anchor. Exception: {e}")
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

            
            # Velocities calculation
            if last_frame_coordinates is None:
                last_frame_coordinates = ((index_x, index_y, index_z), (middle_x, middle_y, middle_z), (wrist_x, wrist_y))
                continue

            v_index_x = index_x - last_frame_coordinates[0][0]
            v_index_y = index_y - last_frame_coordinates[0][1]
            v_index_z = index_z - last_frame_coordinates[0][2]

            v_middle_x = middle_x - last_frame_coordinates[1][0]
            v_middle_y = middle_y - last_frame_coordinates[1][1]
            v_middle_z = middle_z - last_frame_coordinates[1][2]

            v_wrist = ((wrist_x - last_frame_coordinates[2][0])**2 + (wrist_y - last_frame_coordinates[2][1])**2)**0.5

            v_xy = (v_index_x**2 + v_index_y**2)**0.5

            velocities = ((v_index_x, v_index_y, v_index_z), (v_middle_x, v_middle_y, v_middle_z), (v_wrist,))
            last_frame_coordinates = ((index_x, index_y, index_z), (middle_x, middle_y, middle_z), (wrist_x, wrist_y))
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


            # Gesture logic
            if velocities[2][0] > 3.7:
                reader.putText(frame, "Hand too fast.", (20, image_height - 20), reader.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                continue
            else:
                gesture_mode = check_pose(hand)
                if gesture_mode == "TAP_MODE":
                    reader.putText(frame, "MODE: Tap", (50, 15), reader.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                    if velocities[0][2] > TAP_VELOCITY_THRESHOLD:
                        handymen.submit(set_firebase_gesture, "tap")
                    
                elif gesture_mode == "SWIPE_MODE":
                    reader.putText(frame, "MODE: Swipe", (50, 25), reader.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                    if v_xy > SWIPE_VELOCITY_THRESHOLD and swipe_start_data is None:
                        swipe_start_data = {'time': time.time(), 'pos': (index_x, index_y)}
                    pass
                else:
                    reader.putText(frame, "MODE: None", (50, 35), reader.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                    pass
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


            # Check if a swipe needs to be evaluated
            if swipe_start_data is not None and (time.time() - swipe_start_data['time']) > SWIPE_EVALUATION_DELAY:
                start_x, start_y = swipe_start_data['pos']
                delta_x = index_x - start_x
                delta_y = index_y - start_y

                if abs(delta_x) > abs(delta_y):
                    handymen.submit(set_firebase_gesture, "swipe_right" if delta_x > 0 else "swipe_left")
                else:
                    handymen.submit(set_firebase_gesture, "swipe_down" if delta_y > 0 else "swipe_up")
                
                swipe_start_data = None
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


            reader.circle(frame, (index_x, index_y), 2, (255, 255, 255), -1)
            reader.circle(frame, (middle_x, middle_y), 2, (255, 255, 255), -1)
            reader.circle(frame, (wrist_x, wrist_y), 2, (255, 255, 255), -1)
            reader.circle(frame, (index_MCP_x, index_MCP_y), 2, (255, 255, 255), -1)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # Doing the window things
    reader.imshow("Feed", frame)
    if reader.waitKey(1) == ord('q'):
        break
    #-------------------------------------------------------------------------------------------------------------------


camera.release()
reader.destroyAllWindows()