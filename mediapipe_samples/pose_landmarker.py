import math
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    RunningMode,
)

import numpy as np
from mediapipe.python.solutions import drawing_utils, drawing_styles, pose
from mediapipe.framework.formats import landmark_pb2

dir_path = os.path.dirname(__file__)
model_path = f"{dir_path}/../models/pose_landmarker_lite.task"
model_path = f"{dir_path}/../models/pose_landmarker_full.task"
# model_path = f"{dir_path}/../models/pose_landmarker_heavy.task"


def draw_landmarks_on_image(rgb_image, detection_result: PoseLandmarkerResult):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for pose_landmarks in pose_landmarks_list:
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            pose.POSE_CONNECTIONS,
            drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


annotated_image = None


def print_result(
    detection_result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    global annotated_image

    print(f"{timestamp_ms}: {detection_result}")

    annotated_image = draw_landmarks_on_image(
        output_image.numpy_view(), detection_result
    )


# Create a pose landmarker instance with the live stream mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=print_result,
)

with PoseLandmarker.create_from_options(options) as landmarker:
    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, cv2_image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Process and draw landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_image)
        timestamp_ms = math.floor(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # Quit on 'q' press
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

        # Display frame
        if annotated_image is not None:
            cv2.imshow("MediaPipe Pose", annotated_image)

    cap.release()
    cv2.destroyAllWindows()
