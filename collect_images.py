import os
import cv2

num_classes = 3
images_per_class = 200

data_dir = "data/dataset"

cap = cv2.VideoCapture(0)

for i in range(num_classes):
    os.makedirs(os.path.join(data_dir, str(i)), exist_ok=True)

    print(f"Collecting data for class {i}")

    while True:
        ret, frame = cap.read()
        cv2.putText(
            frame,
            "Press 'q' to start",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.imshow("recording", frame)
        if cv2.waitKey(25) == ord("q"):
            break

    for j in range(images_per_class):
        ret, frame = cap.read()
        cv2.imshow("recording", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir, str(i), str(j) + "_new.jpg"), frame)

cap.release()
cv2.destroyAllWindows()
