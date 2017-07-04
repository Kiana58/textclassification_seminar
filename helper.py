import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
from datetime import datetime


def max_to_one(arr):
    copy_ = np.copy(arr)
    copy_[copy_ == np.max(copy_)] = 1
    copy_[copy_ < np.max(copy_)] = 0
    return copy_

def score_prediction(y_true, yhat, binary=True):
    if binary:
        acc = accuracy_score(y_true, np.round(yhat))
        print(classification_report(y_true, np.round(yhat)))
        print("Accuracy: {:.2%}".format(acc))
        print("AUC: {:.2}".format(roc_auc_score(y_true, yhat)))
        pd.Series(yhat.flatten()).hist()
    else:
        yhat_rounded = np.apply_along_axis(max_to_one, arr=yhat, axis=1)
        acc = accuracy_score(y_true, yhat_rounded)
        print(classification_report(y_true, yhat_rounded))
        print("Accuracy: {:.2%}".format(acc))

    return acc


def load_binary_data(path="data/train_binary.csv"):
    df = pd.read_csv(path)

    if "train" in path:
        target = pd.get_dummies(df["Category"])["cancer"]
        documents = df["Abstract"]
        return documents, target
    else:
        documents = df["Abstract"]
        ids = df["Id"]
        return documents, ids


def load_multiclass_data(path="data/train_multiclass.csv"):
    df = pd.read_csv(path)
    target = pd.get_dummies(df["Category"])
    documents = df["Text"]

    return documents, target, df["Category"]


def create_binary_submission(yhat, ids):
    preds = np.round(yhat.flatten())
    sub = pd.DataFrame(data={"Id": ids, "Prediction": preds})

    sub["Category"] = "nonCancer"
    sub.loc[sub["Prediction"] == 1, "Category"] = "cancer"

    return sub[["Id", "Category"]]


def cv_train_model(model, X_train, y_train):
    results = []
    model.compile(optimizer="adagrad",
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    print(model.summary())

    for i in range(11):
        print(f"Iteration {i + 1}")
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)
        model.compile(optimizer="adagrad",
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        model.fit(X_train, y_train, epochs=15, batch_size=256, verbose=0)
        yhat = model.predict(X_test)
        acc = score_prediction(y_test, yhat, binary=False)
        results.append(acc)
        gc.collect()

    print(f"Average accuracy:{np.mean(results)}")
    print(f"Min accuracy:{np.min(results)}")
    print(f"Max accuracy:{np.max(results)}")