import cv2 as reader
import mediapipe
import time
import numpy
import os


# Initialising the camera object
camera = reader.VideoCapture(0)
camera.set(3, 720)
camera.set(4, 405)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Function to preprocess the frame for mediapipe
def preprocess_frame_for_mediapipe(frame):
    """
    This function converts the BGR frame input from OpenCV and converts it to YCrCb to apply the CLAHE filter before converting it to RGB send to the Mediapipe detect_async()

    Args:
        frame
    
    Returns:
        MediaPipe ready image for the detect_async() in SRGB format.
    """
    frame_YCrCb = reader.cvtColor(frame, reader.COLOR_BGR2YCR_CB)

    # Initialising the clahe filter to apply on the frame
    clahe = reader.createCLAHE(clipLimit = 2, tileGridSize = (10, 10))

    frame_YCrCb[:, :, 0] = clahe.apply(frame_YCrCb[:, :, 0])
    frame_RGB = reader.cvtColor(frame_YCrCb, reader.COLOR_YCR_CB2RGB)
    return mediapipe.Image(image_format = mediapipe.ImageFormat.SRGB, data = frame_RGB)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Variable to store the landmarks
landmarks = None

# Defining the callback function for LIVE_STREAM mode
def resultcallback(result, output_img, timestamp):
    """
    This is the function detect_async() calls when it detects a hand in a frame. It updates the landmarks objects with the latest data.

    Args:
        result: The object containing the landmarks for the detected hand
        output_img: No idea
        timestamp: The timestamp for LIVE_STREAM mode
    """
    global landmarks
    landmarks = result

# Initialising the mediapipe hand_land_marker stuff
BaseOptions = mediapipe.tasks.BaseOptions
HandLandmarker = mediapipe.tasks.vision.HandLandmarker
HandLandmarkerOptions = mediapipe.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mediapipe.tasks.vision.RunningMode

# Defining the options for the landmarker object
script_dir = os.path.dirname(os.path.abspath(__file__))
modelpath = os.path.join(script_dir, "hand_landmarker.task")
options = HandLandmarkerOptions(base_options = BaseOptions(model_asset_path = modelpath),
                                running_mode = VisionRunningMode.LIVE_STREAM,
                                result_callback = resultcallback,
                                num_hands = 1)

# Initialising the Landmarker object
landmarker = HandLandmarker.create_from_options(options)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Extract, smooth and return coordinates
def create_kalman_filter():
    """
    Creates a standard Kalman filter for 2D point tracking.
    """
    kf = reader.KalmanFilter(4, 2)
    kf.measurementMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0]], numpy.float32)
    kf.transitionMatrix = numpy.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], numpy.float32)
    kf.processNoiseCov = numpy.eye(4, dtype=numpy.float32) * 0.05  # Q: Trust in prediction
    kf.measurementNoiseCov = numpy.eye(2, dtype=numpy.float32) * 0.07 # R: Trust in measurement
    return kf

# Function to extract and smooth out the necessary coordinates
def get_coordinates(hand, frame):
    """
    Extracts and smoothens the necessary coordinates using the Kalman filter.

    Args:
        hand: The hand landmarks containing object
        frame: Just to extract the frame height and width.

    Results:
        Returns a tuple of tuples with [0] : index tip coords (kalmanised x and y)
                                       [1] : middle tip coords (kalmanised x and y)
                                       [2] : wrist coords
                                       [3] : index MCP coords (Mediapipe format)
    """
    frame_height, frame_width = frame.shape[:2]

    # Finger position calculation
    index = hand[8]
    index_x = (int)(index.x * frame_width)
    index_y = (int)(index.y * frame_height)
    index_z = index.z

    middle = hand[12]
    middle_x = (int)(middle.x * frame_width)
    middle_y = (int)(middle.y * frame_height)
    middle_z = middle.z

    wrist = hand[0]
    wrist_x = (int)(wrist.x * frame_width)
    wrist_y = (int)(wrist.y * frame_height)
    
    index_MCP = hand[5]
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # Defining a Kalman object for the index tip
    kalman_index_flag = False
    kalman_index = create_kalman_filter()
    # Defining a Kalman object for the middle tip
    kalman_middle_flag = False
    kalman_middle = create_kalman_filter()
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


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
    
    middle_x = (int)(kalman_middle.statePost[0, 0])    # Kalmanised middle_x
    middle_y = (int)(kalman_middle.statePost[1, 0])    # Kalmanised middle_y
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


    return ((index_x, index_y, index_z), (middle_x, middle_y, middle_z), (wrist_x, wrist_y), (index_MCP.x, index_MCP.y))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Firebase setup and functions
import firebase_admin
from firebase_admin import credentials, db

firebase_key_path = os.path.join(script_dir, "firebase_key.json")
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
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Initialising the ThreadPoolExecutor thing
from concurrent.futures import ThreadPoolExecutor
handymen = ThreadPoolExecutor(max_workers = 3)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Sending anchor coordinates
last_anchor_position = None

def set_anchor_position(anchor_position):
    try:
        anchor.set(anchor_position)
    except Exception as e:
        print(f"Firebase anchor sending failed. Exception: {e}")

def update_anchor_position(anchor):
    global last_anchor_position
    index_MCP_x, index_MCP_y = anchor
    anchor_position = {'x' : index_MCP_x, 'y' : index_MCP_y}

    # Sending the anchor coordinates only if it's the first time hand appeared or if the anchor has moved by 2%.
    if last_anchor_position is None or (((anchor_position['x'] - last_anchor_position['x'])**2 + (anchor_position['y'] - last_anchor_position['y'])**2 )**0.5 > 0.02):
        last_anchor_position = anchor_position
        try:
            handymen.submit(set_anchor_position, anchor_position)
        except Exception as e:
            print(f"Handymen submit error for anchor. Exception: {e}")
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Velocities calculation
last_frame_coordinates = None

def calculate_velocities(coords):
    global last_frame_coordinates
    if last_frame_coordinates is None:
        last_frame_coordinates = coords

    v_index_x = coords[0][0] - last_frame_coordinates[0][0]
    v_index_y = coords[0][1] - last_frame_coordinates[0][1]
    v_index_z = coords[0][2] - last_frame_coordinates[0][2]

    v_middle_x = coords[1][0] - last_frame_coordinates[1][0]
    v_middle_y = coords[1][1] - last_frame_coordinates[1][1]
    v_middle_z = coords[1][2] - last_frame_coordinates[1][2]

    v_wrist = ((coords[2][0] - last_frame_coordinates[2][0])**2 + (coords[2][1] - last_frame_coordinates[2][1]))

    v_index_xy = ((v_index_x)**2 + (v_index_y)**2)**0.5

    last_frame_coordinates = coords
    return ((v_index_x, v_index_y, v_index_z), (v_middle_x, v_middle_y, v_middle_z), (v_wrist,), (v_index_xy,))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Gesture logic
# Threshold variables
TAP_VELOCITY_THRESHOLD = 0.01 # A sharp Z-movement
SWIPE_VELOCITY_THRESHOLD = 14.3784   # A sharp XY-movement
SWIPE_EVALUATION_DELAY = 0.1    # 100ms

# Swipe detection variables
swipe_start_data = None

# Pose detection function for Mode identification
def check_pose(hand):
    is_index_up = hand[8].y < hand[5].y
    is_middle_up = hand[12].y < hand[9].y

    if is_index_up and is_middle_up:
        return "TAP_MODE"
    elif not is_middle_up:
        return "SWIPE_MODE"
    return "NO_MODE"


def process_gestures(frame, velocities, hand, coords):
    frame_height, frame_width = frame.shape[:2]
    global swipe_start_data

    if velocities[2][0] > 8:
        reader.putText(frame, "Hand unstable.", (20, frame_height - 20), reader.FONT_ITALIC, 0.5, (0, 0, 0), 1)
        return
    else:
        gesture_mode = check_pose(hand)

        if gesture_mode == "TAP_MODE":
            reader.putText(frame, "Mode: TAP", (50, 15), reader.FONT_ITALIC, 0.5, (0, 0, 0), 1)

            avg_velocity = (velocities[0][2] + velocities[1][2]) / 2
            if avg_velocity > TAP_VELOCITY_THRESHOLD:
                reader.putText(frame, "Gesture: tap", (20, 40), reader.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                handymen.submit(set_firebase_gesture, "tap")

        elif gesture_mode == "SWIPE_MODE":
            reader.putText(frame, "Mode: SWIPE", (20, 45), reader.FONT_ITALIC, 0.5, (0, 0, 0), 1)

            if velocities[3][0] > SWIPE_VELOCITY_THRESHOLD and swipe_start_data is None:
                swipe_start_data = {'time' : time.time(), 'pos' : (coords[0][0], coords[0][1])}
        
        else:
            reader.putText(frame, "Mode: None", (20, 45), reader.FONT_ITALIC, 0.5, (0, 0, 0), 1)
        

        # Check if a swipe needs to be evaluated
        if swipe_start_data is not None and (time.time() - swipe_start_data['time']) > SWIPE_EVALUATION_DELAY:
            start_x, start_y = swipe_start_data['pos']
            delta_x = coords[0][0] - start_x
            delta_y = coords[0][1] - start_y

            if abs(delta_x) > abs(delta_y):
                handymen.submit(set_firebase_gesture, "swipe_right" if delta_x < 0 else "swipe_left")
                reader.putText(frame, "Gesture: swipe_right" if delta_x < 0 else "Gesture: swipe_left", (20, 45), reader.FONT_ITALIC, 0.5, (0, 0, 0), 1)
            else:
                handymen.submit(set_firebase_gesture, "swipe_down" if delta_y > 0 else "swipe_up")
                reader.putText(frame, "Gesture: swipe_down" if delta_y > 0 else "Gesture: swipe_up", (20, 45), reader.FONT_ITALIC, 0.5, (0, 0, 0), 1)
            
            swipe_start_data = None # Reset swipe data after performing a swipe
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Drawing visual feedback
def draw_feedback(frame, coords):
    frame_height, frame_width = frame.shape[:2]
    index_MCP_x = (int)(coords[3][0] * frame_width)
    index_MCP_y = (int)(coords[3][1] * frame_height)

    reader.circle(frame, (coords[0][0], coords[0][1]), 2, (255, 255, 255), -1)
    reader.circle(frame, (coords[1][0], coords[1][1]), 2, (255, 255, 255), -1)
    reader.circle(frame, (coords[2][0], coords[2][1]), 2, (255, 255, 255), -1)
    reader.circle(frame, (index_MCP_x, index_MCP_y), 2, (255, 255, 255), -1)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Main Loop
while True:
    # Get the frame and confirmation_boolean from .read()
    flag, frame = camera.read()
    if not flag:    # If the frame reading wasn't successful, skip this iteration.
        continue
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # Calling the detection function using the landmarker object
    timestamp = (int)(time.time() * 1000)
    try:
        # If detect_async() detects a hand it calls resultcallback() which updates "landmarks" with new result
        landmarker.detect_async(preprocess_frame_for_mediapipe(frame), timestamp)
    except Exception as e:
        print (f"Detection error in detect_async(). Exception: {e}")
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


    if not landmarks or not landmarks.hand_landmarks:
        pass
    else:

        for hand_idx, hand in enumerate(landmarks.hand_landmarks):
            # 1. Get the smoothened coordinates and the index_MCP coordinates to send as anchor position
            coords = get_coordinates(hand, frame)       # coords is a tuple of tuples with coords[0] = index_tip, coords[1] = middle_tip, coords[2] = wrist, coords[3] = index_MCP

            # 2. Sending the anchor position
            update_anchor_position(coords[3])

            # 3. Velocities calculation
            velocities = calculate_velocities(coords)

            # 4. Gesture logic
            process_gestures(frame, velocities, hand, coords)

            # 5. Visual feedback
            draw_feedback(frame, coords)
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

    # Doing the window things
    reader.imshow("Feed", frame)
    if reader.waitKey(1) == ord('q'):
        break
    #-------------------------------------------------------------------------------------------------------------------


camera.release()
reader.destroyAllWindows()
handymen.shutdown()