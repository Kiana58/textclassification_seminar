from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np


def score_prediction(y_true, yhat, acc_only=False):
    acc = accuracy_score(y_true, np.round(yhat))
    if not acc_only:
        print(classification_report(y_true, np.round(yhat)))
        print("Accuracy: {:.2%}".format(acc))
        print("AUC: {:.2}".format(roc_auc_score(y_true, yhat)))
        pd.Series(yhat.flatten()).hist()
    
    return acc