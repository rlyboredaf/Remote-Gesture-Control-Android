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
    kf.measurementNoiseCov = numpy.eye(2, dtype=numpy.float32) * 1 # R: Trust in measurement
    return kf

# Defining a Kalman object for the index
kalman_index_flag = False
kalman_index = create_kalman_filter()
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# --- CALIBRATION VARIABLES ---
calibration_mode = 'none' # Can be 'none', 'still', 'tap', 'swipe'
frame_recorder_count = 0
RECORDING_FRAMES = 30

# Lists to store the recorded velocities
still_v_xy_readings = []
still_v_z_readings = []
still_v_wrist_readings = [] # New list for wrist stillness
tap_v_z_readings = []
swipe_v_xy_readings = []

# Gesture state variables for velocity calculation
last_index_z = None
last_index_x = None
last_index_y = None
last_wrist_x = None # New variable for wrist velocity
last_wrist_y = None # New variable for wrist velocity
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Main loop
while True:
    flag, frame = camera.read()

    if not flag:
        continue

    # Create a clean copy for display before drawing on it
    display_frame = frame.copy()

    # Converting the frames (and applying the CLAHE filter on it)
    frame_YCrCb = reader.cvtColor(frame, reader.COLOR_BGR2YCR_CB)
    frame_YCrCb[:, :, 0] = clahe.apply(frame_YCrCb[:, :, 0])
    frame_RGB = reader.cvtColor(frame_YCrCb, reader.COLOR_YCR_CB2RGB)
    mediapipe_img = mediapipe.Image(image_format = mediapipe.ImageFormat.SRGB, data = frame_RGB)
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # Calling the detection function using the landmarker object.
    timestamp = (int)(time.time() * 1000)
    try:
        landmarker.detect_async(mediapipe_img, timestamp)
    except Exception as e:
        print(f"Detection failed, {e}")
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # Proceeding onto hand evaluation
    if not Hand_landmarks_containing_object or not Hand_landmarks_containing_object.hand_landmarks:
        # This is the case when no hand is detected by the detect_async() method above
        kalman_index_flag = False   # Reset
        last_index_z = last_index_x = last_index_y = None
        last_wrist_x = last_wrist_y = None
    else:
        # This is the case where a hand is detectedd by the detect_async() method above
        image_height, image_width = frame.shape[:2]

        for hand_idx, hand in enumerate(Hand_landmarks_containing_object.hand_landmarks):
            index = hand[8]
            wrist = hand[0]
            index_x = (int)(index.x * image_width)
            index_y = (int)(index.y * image_height)
            index_z = index.z
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
            
            kalman_index_x = (int)(kalman_index.statePost[0, 0])
            kalman_index_y = (int)(kalman_index.statePost[1, 0])
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


            # Velocity calculation
            if last_index_x is None: # Simplified check for first frame
                last_index_x = kalman_index_x
                last_index_y = kalman_index_y
                last_index_z = index_z
                last_wrist_x = wrist_x
                last_wrist_y = wrist_y
                continue # Skip the first frame to get a valid velocity reading
            else:
                v_z = index_z - last_index_z
                v_xy = ((kalman_index_x - last_index_x)**2 + (kalman_index_y - last_index_y)**2)**0.5
                v_wrist = ((wrist_x - last_wrist_x)**2 + (wrist_y - last_wrist_y)**2)**0.5
                
                last_index_x = kalman_index_x
                last_index_y = kalman_index_y
                last_index_z = index_z
                last_wrist_x = wrist_x
                last_wrist_y = wrist_y
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


            # --- CALIBRATION RECORDING LOGIC ---
            if frame_recorder_count < RECORDING_FRAMES:
                if calibration_mode == 'still':
                    still_v_xy_readings.append(v_xy)
                    still_v_z_readings.append(abs(v_z))
                    still_v_wrist_readings.append(v_wrist) # Record wrist velocity
                    frame_recorder_count += 1
                elif calibration_mode == 'tap':
                    # Only record the "plunge" part of the tap (moving away from camera)
                    if v_z > 0:
                        tap_v_z_readings.append(v_z)
                    frame_recorder_count += 1
                elif calibration_mode == 'swipe':
                    swipe_v_xy_readings.append(v_xy)
                    frame_recorder_count += 1
            else:
                if calibration_mode != 'none':
                    print(f"--- Finished recording for '{calibration_mode}' ---")
                    calibration_mode = 'none' # Reset mode after recording is done

            reader.circle(display_frame, (kalman_index_x, kalman_index_y), 4, (0, 255, 0), -1)
            reader.circle(display_frame, (wrist_x, wrist_y), 4, (0, 255, 0), -1)
    
    # --- VISUAL FEEDBACK FOR CALIBRATION ---
    if calibration_mode != 'none':
        text = f"RECORDING {calibration_mode.upper()}: {frame_recorder_count}/{RECORDING_FRAMES}"
        reader.putText(display_frame, text, (10, 30), reader.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        text = "Press 'c' (still), 't' (tap), 's' (swipe), or 'q' (quit)"
        reader.putText(display_frame, text, (10, 30), reader.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    reader.imshow("Calibration Feed", display_frame)
    key = reader.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        print("\n--- Starting STILLNESS calibration ---")
        calibration_mode = 'still'
        frame_recorder_count = 0
        still_v_xy_readings.clear()
        still_v_z_readings.clear()
        still_v_wrist_readings.clear()
    elif key == ord('t'):
        print("\n--- Starting TAP calibration ---")
        calibration_mode = 'tap'
        frame_recorder_count = 0
        tap_v_z_readings.clear()
    elif key == ord('s'):
        print("\n--- Starting SWIPE calibration ---")
        calibration_mode = 'swipe'
        frame_recorder_count = 0
        swipe_v_xy_readings.clear()


# --- FINAL CALCULATION AND RESULTS ---
print("\n\n--- CALIBRATION COMPLETE ---")

if still_v_wrist_readings:
    wrist_still_thresh = max(still_v_wrist_readings) * 1.5 # Max noise + 50% buffer
    print(f"\nRECOMMENDED WRIST STILLNESS THRESHOLD:")
    print(f"WRIST_STILLNESS_THRESHOLD = {wrist_still_thresh:.4f}")
else:
    print("\nWrist stillness calibration was not performed. Using default value.")
    wrist_still_thresh = 12.0

if still_v_xy_readings and still_v_z_readings:
    still_thresh_xy = max(still_v_xy_readings) * 1.5 # Max noise + 50% buffer
    still_thresh_z = max(still_v_z_readings) * 1.5   # Max noise + 50% buffer
    print(f"\nRECOMMENDED FINGER STILLNESS THRESHOLDS (Max noise detected + 50% buffer):")
    print(f"STILLNESS_THRESHOLD_XY = {still_thresh_xy:.4f}")
    print(f"STILLNESS_THRESHOLD_Z = {still_thresh_z:.4f}")
else:
    print("\nFinger stillness calibration was not performed. Using default values.")
    still_thresh_xy = 5.0
    still_thresh_z = 0.008

if tap_v_z_readings:
    # Use the average of the top 50% of tap velocities to get a robust threshold
    tap_v_z_readings.sort(reverse=True)
    top_half_taps = tap_v_z_readings[:len(tap_v_z_readings)//2]
    tap_thresh = sum(top_half_taps) / len(top_half_taps) if top_half_taps else 0
    print(f"\nRECOMMENDED TAP THRESHOLD (Average of the fastest 50% of Z-velocities):")
    print(f"TAP_VELOCITY_THRESHOLD = {tap_thresh:.4f}")
else:
    print("\nTap calibration was not performed.")

if swipe_v_xy_readings:
    # Use the average of the bottom 50% of swipe velocities to find the slowest intentional swipe
    swipe_v_xy_readings.sort()
    bottom_half_swipes = [v for v in swipe_v_xy_readings if v > still_thresh_xy][:len(swipe_v_xy_readings)//2]
    swipe_thresh = sum(bottom_half_swipes) / len(bottom_half_swipes) if bottom_half_swipes else 0
    print(f"\nRECOMMENDED SWIPE THRESHOLD (Average of the slowest 50% of XY-velocities):")
    print(f"SWIPE_VELOCITY_THRESHOLD = {swipe_thresh:.4f}")
else:
    print("\nSwipe calibration was not performed.")

print("\n-------------------------------------------")
print("Use these values in your main application's if/elif logic.")


camera.release()
reader.destroyAllWindows()
