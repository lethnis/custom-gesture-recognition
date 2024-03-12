import pickle

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open("data/data.pickle", "rb"))

data = np.array(data_dict["data"])
labels = np.array(data_dict["labels"])

data = np.reshape(data, (data.shape[0], -1))

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, stratify=labels, shuffle=True
)

model = RandomForestClassifier()

model.fit(train_data, train_labels)

pred_labels = model.predict(test_data)

score = accuracy_score(test_labels, pred_labels)

print(f"Correctly classified {score*100:.2f}% of samples")
# Correctly classified 100.00% of samples

pickle.dump(model, open("data/model.pickle", "wb"))
