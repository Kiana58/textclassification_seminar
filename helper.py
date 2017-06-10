from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
from datetime import datetime


def score_prediction(y_true, yhat, binary=True):

    def max_to_one(arr):
        arr[arr == np.max(arr)] = 1
        arr[arr < np.max(arr)] = 0
        return arr

    if binary:
        acc = accuracy_score(y_true, np.round(yhat))
        print(classification_report(y_true, np.round(yhat)))
        print("Accuracy: {:.2%}".format(acc))
        print("AUC: {:.2}".format(roc_auc_score(y_true, yhat)))
        pd.Series(yhat.flatten()).hist()
    else:
        yhat = np.apply_along_axis(max_to_one, arr=yhat, axis=1)
        acc = accuracy_score(y_true, np.round(yhat))
        print(classification_report(y_true, np.round(yhat)))
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

    return documents, target


def create_binary_submission(yhat, ids, save_path):
    if not save_path:
        return print("Need a save path!")
    preds = np.round(yhat.flatten())
    sub = pd.DataFrame(data={"Id": ids, "Prediction": preds})

    sub["Category"] = "nonCancer"
    sub.loc[sub["Prediction"] == 1, "Category"] = "cancer"
    path = save_path + "binary_submission_" + str(datetime.now()) + ".csv"
    sub[["Id", "Category"]].to_csv(path, index=False)