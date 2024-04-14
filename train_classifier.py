import pickle

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    """Program trains random forest classifier to classify hand landmarks.
    After training model is saved to data/model.pickle"""

    # load our (x, y) coordinates and labels
    with open("data/data.pickle", "rb") as f:
        data_dict = pickle.load(f)

    # shape of data: (n_samples, 21, 2) where 21 is landmarks and 2 is (x, y)
    data = np.array(data_dict["data"])
    # shape of labels: (n_samples,)
    labels = np.array(data_dict["labels"])

    # reshape data from (n_samples, 21, 2) to (n_samples, 42)
    data = np.reshape(data, (data.shape[0], -1))

    # split and shuffle data and labels in 80%/20% proportion
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, stratify=labels, shuffle=True
    )

    # initialize random forest model with default parameters
    model = RandomForestClassifier(n_estimators=100, max_depth=7)

    # train model
    model.fit(train_data, train_labels)

    # check how model performs of unseen data
    pred_labels = model.predict(test_data)

    # get the accuracy
    score = accuracy_score(test_labels, pred_labels)

    # Correctly classified 94.87% of samples in my case
    print(f"Correctly classified {score*100:.2f}% of samples")

    with open("data/model.pickle", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
