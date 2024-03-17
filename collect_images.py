import os
import cv2
from pathlib import Path


def main():
    """Program collects images from a webcam and saves them to specified folder.
    When ready press 'q' to start recording and saving.
    Edit classes.txt to change gestures. Default amount of images per each class is 500."""

    classes = Path("classes.txt").read_text(encoding="utf-8")
    classes = [val.strip() for val in classes.split(",")]
    print(classes)
    IMAGES_PER_CLASS = 500
    DATA_DIR = "data/dataset"

    # start capturing video from a webcam with index 0
    cap = cv2.VideoCapture(0)

    # main loop for recording images for each class
    for i, val in enumerate(classes):
        # make class directory if it does not exist
        os.makedirs(os.path.join(DATA_DIR, str(i)), exist_ok=True)

        print(f"Collecting data for class {val}")

        # waiting loop, if 'q' pressed goes to next loop
        while True:
            ret, frame = cap.read()
            cv2.putText(
                frame,
                f"Press 'q' to collect images for '{val}' class",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.imshow("recording", frame)
            if cv2.waitKey(25) == ord("q"):
                break

        # records and saves images
        for j in range(IMAGES_PER_CLASS):
            ret, frame = cap.read()
            cv2.imshow("recording", frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, str(i), str(j) + ".jpg"), frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
