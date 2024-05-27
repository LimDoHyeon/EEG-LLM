"""
metrics : Kappa's coefficient
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def kappa_score(y_true, y_pred):
    """
    Calculate the Kappa's coefficient between the true and predicted labels.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Kappa's coefficient
    """
    return cohen_kappa_score(y_true, y_pred)


# Example usage
# TODO: y_true, y_pred 받아오는 로직 구현
