from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from mhcpred.config import settings

output_path = Path(settings.output_path)


def mhcflurry_metrics():
    df = pd.read_csv(str(output_path / "mhcflurry_predictions.csv"))
    y_pred = df.prediction_percentile.values <= 2
    acc = accuracy_score(y_true=df.hit.values, y_pred=y_pred)
    confusion_mat = confusion_matrix(y_true=df.hit.values, y_pred=y_pred)
    balanced_acc = balanced_accuracy_score(y_true=df.hit.values, y_pred=y_pred)
    return acc, balanced_acc, confusion_mat


def mhcpred_metrics():
    df = pd.read_csv(str(output_path / "mhcpred_predictions.csv"))
    y_pred = df.predictions.values >= 0.5
    acc = accuracy_score(y_true=df.hit.values, y_pred=y_pred)
    confusion_mat = confusion_matrix(y_true=df.hit.values, y_pred=y_pred)
    balanced_acc = balanced_accuracy_score(y_true=df.hit.values, y_pred=y_pred)
    return acc, balanced_acc, confusion_mat


mhcflurry = mhcflurry_metrics()
mhcpred = mhcpred_metrics()
