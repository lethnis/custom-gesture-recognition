import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.python import solutions
from mediapipe.framework.formats import landmark_pb2
import sklearn


classes = {0: "Dog", 1: "Rock", 2: "Phone"}

MIN_CONFIDENCE = 0.7


def draw_landmarks(img: mp.Image, detection_results: vision.HandLandmarkerResult) -> np.array:
    """Function takes an image and detection results and draws landmarks
    if found any hand. Returns an annotated image."""

    # img.numpy_view() is readable only
    # so we need to make a copy of it for use in drawing_utils
    annotated_img = np.copy(img.numpy_view())
    # get only hand landmarks from detection results
    hand_landmarks_list = detection_results.hand_landmarks

    # for every hand found
    for hand_landmarks in hand_landmarks_list:
        # we convert every landmark to pb2 format
        # so we can use in in drawing utils
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks]
        )

        # main drawing function
        solutions.drawing_utils.draw_landmarks(
            image=annotated_img,
            landmark_list=hand_landmarks_proto,
            connections=solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=solutions.drawing_styles.get_default_hand_connections_style(),
        )
    return annotated_img


def extract_hand_landmarks(detection_results: vision.HandLandmarkerResult) -> np.array:
    """Function takes detection results and
    returns a list of 21 (x, y) pair for every hand detected"""

    # get only hand landmarks from detection results
    hand_landmarks_list = detection_results.hand_landmarks

    # initialize a resulting list
    data_xy = []
    # for every hand found
    for hand in hand_landmarks_list:
        # list for current hand landmarks
        hand_xy = []
        # for every landmark we extract x and y
        for landmark in hand:
            hand_xy.append((landmark.x, landmark.y))
        # append current hand 21 (x, y) pairs
        data_xy.append(hand_xy)
    return data_xy


def classify_and_draw(
    img: mp.Image, detection_results: vision.HandLandmarkerResult, model: sklearn.base.BaseEstimator
) -> np.array:
    """Function takes an image, detection results and a model
    and returns image with drawn landmarks and predicted class"""

    # first we draw landmarks
    img = draw_landmarks(img, detection_results)
    # extract 21 pair of (x, y) for every hand in the image
    data = extract_hand_landmarks(detection_results)
    # if at least 1 hand found
    if data:
        for hand in data:
            # get top left corner coordinates for text
            x_min = int(min(i[0] for i in hand) * img.shape[1])
            y_min = int(min(i[1] for i in hand) * img.shape[0])
            # reshape 21 pair of (x, y) to 1 vector with 42 features
            hand = np.reshape(hand, (1, -1))  # (21, 2) -> (1, 42)

            # prediction will look like [[0.1, 0.2, 0.7]] for 3 classes
            pred = model.predict_proba(hand)

            # if there is class with probability score greater than MAX_CONFIDENCE
            if any(pred[0] > MIN_CONFIDENCE):
                # we get the index of the biggest value in pred
                index = pred.argmax()
                # get the name of the class from dict at the top of this file
                class_name = classes[index]
            else:
                # if confidence is less than MAX_CONFIDENCE
                class_name = "Unknown"

            cv2.putText(img, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

    return img
