import os
import pickle
from pathlib import Path

from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks.python import vision

import utils


def main():
    """Creates a dataset from hands landmarks. Program uses mediapipe's hand
    landmarks detection model https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models.
    Dataset contains labels and 21 pair of (x, y) for each hand detected.
    """

    MAX_NUM_HANDS = 2
    DATA_DIR = "data/dataset"

    classes = Path("classes.txt").read_text(encoding="utf-8")
    dict_classes = {idx: val.strip() for idx, val in enumerate(classes.split(","))}

    # specify path to the hand landmarker model and num_hands
    # model should be downloaded manually from the link above
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=MAX_NUM_HANDS,
    )

    # create our model from options
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    # create empty lists for (x, y) pairs and labels
    data = []
    labels = []

    # iterating through every folder and every image
    for idx, val in dict_classes.items():

        print(f"Extracting class '{val}'")
        images_path = os.listdir(os.path.join(DATA_DIR, str(idx)))

        for img in tqdm(images_path):
            img_path = os.path.join(DATA_DIR, str(idx), img)
            # create image as mp.Image file
            img = mp.Image.create_from_file(img_path)
            # detect hands
            results = hand_landmarker.detect(img)

            # if we detected at least 1 hand
            if results.hand_landmarks:
                # extract 21 (x, y) pairs for every hand
                hands = utils.extract_hand_landmarks(results)
                # add 1 or more hands
                data.extend(hands)
                # add class num depending on how much hands we detected
                labels.extend([idx] * len(hands))

    # save data and labels as dict
    with open("data/data.pickle", "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)


if __name__ == "__main__":
    main()
