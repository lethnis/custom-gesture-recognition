import pickle
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision

import utils


def main():
    """Program runs model on a webcam. It takes each frame and detect landmarks
    then classifies those landmarks.
    Landmarker model is here https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models.
    Press 'q' to quit"""

    MAX_NUM_HANDS = 2

    # create options for hand landmarker model
    # specify path to the model, max number of hands and video mode
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=MAX_NUM_HANDS,
        running_mode=vision.RunningMode.VIDEO,
    )

    # create an instance of our model from options above
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    # start capturing video from a webcam with index 0
    cap = cv2.VideoCapture(0)

    # load random forest classifier model
    with open("data/model.pickle", "rb") as f:
        model = pickle.load(f)

    # until user don't press 'q' we reading and showing webcam stream
    print("Press 'q' to quit.")
    while True:

        # read next frame and convert it to the mp.Image
        ret, frame = cap.read()
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # get our detections
        # in video mode you need to specify frame and timestamp in ms
        detections = hand_landmarker.detect_for_video(frame, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        frame = utils.classify_and_draw(frame, detections, model)

        # show frame with drawn landmarks and predicted class
        cv2.imshow("", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
