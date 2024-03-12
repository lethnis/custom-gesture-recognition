import os
import pickle
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks.python import vision

import utils


options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2,
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

DATA_DIR = "data/dataset"

data = []
labels = []

for cls in os.listdir(DATA_DIR):
    print(f"Extracting class {cls}")
    images_path = os.listdir(os.path.join(DATA_DIR, cls))

    for img in tqdm(images_path):
        img_path = os.path.join(DATA_DIR, cls, img)
        img = mp.Image.create_from_file(img_path)
        results = hand_landmarker.detect(img)

        data.extend(utils.extract_hand_landmarks(results))
        labels.append(cls)


with open("data/data.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)
