import numpy as np
from sklearn.metrics import roc_auc_score

def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    return 2 * roc_auc_score(actual, pred) - 1


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)