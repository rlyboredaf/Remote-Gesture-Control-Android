import cv2 as reader
import mediapipe
import numpy
import time

latest_frame = None
latest_result = None

def result_callback (result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


# Setting up the camera
camera = reader.VideoCapture(0)
camera.set(3, 1080)
camera.set(4, 720)


# Initialising stuff to make life easier
BaseOptions = mediapipe.tasks.BaseOptions
HandLandmarker = mediapipe.tasks.vision.HandLandmarker
HandLandmarkerOptions = mediapipe.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mediapipe.tasks.vision.RunningMode
mp_hands = mediapipe.solutions.hands


# Create a hand landmarker instance with the LIVE_STREAM mode:
model_path = r"C:\Coding\Python stuff\Hand_project\hand_landmarker.task"
options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path = model_path),    
                                running_mode=VisionRunningMode.LIVE_STREAM,
                                result_callback = result_callback,
                                num_hands = 2)
landmarker = HandLandmarker.create_from_options(options)


while True:
    flag, snapshot = camera.read()
    if not flag:
        continue

    # Converting the frame for mediapipe to work on
    snapshot_RGB = reader.cvtColor(snapshot, reader.COLOR_BGR2RGB)
    mediapipe_img = mediapipe.Image(image_format = mediapipe.ImageFormat.SRGB,
                                    data = snapshot_RGB)
    
    timestamp = int(time.time() * 1000)
    latest_frame = snapshot.copy()

    # Try catching an Exception
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


    # Drawing
    frame_to_draw = latest_frame.copy()
    if latest_result and latest_result.hand_landmarks:
        for hand in latest_result.hand_landmarks:
            for landmark in hand:
                x = int(landmark.x * frame_to_draw.shape[1])
                y = int(landmark.y * frame_to_draw.shape[0])
                reader.circle(frame_to_draw, (x, y), 5, (0, 255, 0), -1)


            for s_idx, e_idx in mp_hands.HAND_CONNECTIONS:
                start = hand[s_idx]
                end = hand[e_idx]
                x1 = int(start.x * frame_to_draw.shape[1])
                y1 = int(start.y * frame_to_draw.shape[0])
                x2 = int(end.x * frame_to_draw.shape[1])
                y2 = int(end.y * frame_to_draw.shape[0])
                reader.line(frame_to_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

    
    image_height, image_width = frame_to_draw.shape[:2]

    for hand_idx, hand in enumerate(latest_result.hand_landmarks):
        x = [int(lm.x * image_width) for lm in hand]
        y = [int(lm.y * image_height) for lm in hand]
        x_min = max(min(x) - 20, 0)
        y_min = max(min(y) - 20, 0)
        x_max = min(max(x) + 20, image_width)
        y_max = min(max(y) + 20, image_height)

        reader.rectangle(frame_to_draw, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

        label = latest_result.handedness[hand_idx][0].category_name
        score = latest_result.handedness[hand_idx][0].score
        text = f"{label} ({score:2f})"
        reader.putText(frame_to_draw, text, (x_min, y_min - 5), reader.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

        # Thumb index line
        thumb = hand[4]
        index = hand[8]

        thumb_x = int(thumb.x * image_width)
        thumb_y = int(thumb.y * image_height)
        index_x = int(index.x * image_width)
        index_y = int(index.y * image_height)

        reader.line(frame_to_draw, (thumb_x, thumb_y), (index_x, index_y), (255, 255, 255), 2)

        pinch_distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5
        if pinch_distance < 20:
            reader.putText(frame_to_draw, "Pinched", (x_min, y_max + 25), reader.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        else:
            reader.putText(frame_to_draw, "Unpinched", (x_min, y_max + 25), reader.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

    
    # Doing the window things
    reader.imshow("LIVE_FEED", frame_to_draw)
    if reader.waitKey(1) == ord('q'):
        break  

camera.release()
reader.destroyAllWindows()