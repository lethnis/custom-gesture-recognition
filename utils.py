import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.python import solutions
from mediapipe.framework.formats import landmark_pb2
import sklearn


classes = {0: "Dog", 1: "Rock", 2: "Phone"}


def draw_landmarks(img: mp.Image, detection_results: vision.HandLandmarkerResult) -> np.array:
    annotated_img = np.copy(img.numpy_view())
    hand_landmarks_list = detection_results.hand_landmarks

    for hand_landmarks in hand_landmarks_list:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_img,
            landmark_list=hand_landmarks_proto,
            connections=solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=solutions.drawing_styles.get_default_hand_connections_style(),
        )
    return annotated_img


def extract_hand_landmarks(detection_results: vision.HandLandmarkerResult) -> np.array:
    hands_landmarks = detection_results.hand_landmarks
    data_xy = []
    for hand in hands_landmarks:
        hand_xy = []
        for landmark in hand:
            hand_xy.append((landmark.x, landmark.y))
        data_xy.append(hand_xy)
    return data_xy


def classify_img(
    img: mp.Image, detection_results: vision.HandLandmarkerResult, model: sklearn.base.BaseEstimator
) -> np.array:
    img = draw_landmarks(img, detection_results)
    data = extract_hand_landmarks(detection_results)
    pred = "Unknown"
    if data:
        for hand in data:
            x_min = int(min(i[0] for i in hand) * img.shape[1])
            y_min = int(min(i[1] for i in hand) * img.shape[0])
            hand = np.reshape(hand, (1, -1))
            pred = model.predict_proba(hand)
            pred = classes[pred.argmax()] if any(pred[0] > 0.7) else "Unknown"

            cv2.putText(img, pred, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

    return pred, img
