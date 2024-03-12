import os
import cv2


def main():
    """Program collects images from a webcam and saves them to specified folder.
    When ready press 'q' to start recording and saving.
    Default number of classes is 3, and images per each class is 200."""

    NUM_CLASSES = 3
    IMAGES_PER_CLASS = 200
    DATA_DIR = "data/dataset"

    # start capturing video from a webcam with index 0
    cap = cv2.VideoCapture(0)

    # main loop for recording images for each class (default is 3)
    for i in range(NUM_CLASSES):
        # make class directory if it does not exist
        os.makedirs(os.path.join(DATA_DIR, str(i)), exist_ok=True)

        print(f"Collecting data for class {i}")

        # waiting loop, if 'q' pressed goes to next loop
        while True:
            ret, frame = cap.read()
            cv2.putText(
                frame, "Press 'q' to start", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2
            )
            cv2.imshow("recording", frame)
            if cv2.waitKey(25) == ord("q"):
                break

        # records and saves images (default is 200)
        for j in range(IMAGES_PER_CLASS):
            ret, frame = cap.read()
            cv2.imshow("recording", frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, str(i), str(j) + ".jpg"), frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
