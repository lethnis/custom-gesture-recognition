import pickle
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision

import utils

options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO,
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)


model = pickle.load(open("data/model.pickle", "rb"))


while True:
    ret, frame = cap.read()
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    detections = hand_landmarker.detect_for_video(mp_img, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    pred, frame = utils.classify_img(mp_img, detections, model)

    cv2.imshow("", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
