from lib.data import load_file
from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.config import get_config
from lib.scoring.gini import gini_normalized
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_predict, GridSearchCV
from scipy.sparse import csc_matrix
from xgboost import XGBClassifier
import numpy as np
from datetime import datetime

start = datetime.now()
logger = get_logger()

logger.info("Loading training data into X and y...")
train = load_file()
X = train.drop('target', axis = 1)
y = train.target
n = X.shape[1]

logger.info("Making GridSearchCV Pipeline...")
# kbest
pipe = Pipeline([('features', RFE(estimator = SVC(kernel="linear"))),
                 ('model', XGBClassifier())])
param_grid = {
    'features__n_features_to_select': [int(p * n) for p in [.5, .7]],
    'model__learning_rate': [0.095],
    'model__reg_alpha': [0.35],
    'model__reg_lambda': [0.76],
    'model__max_depth': [5]
}

model = GridSearchCV(pipe, param_grid, scoring = 'roc_auc')

logger.info("Fitting model on X...")
model.fit(X, y)
logger.info("Best Params: {}".format(model.best_params_))

logger.info("Predicting score (w/Cross-Val) on X...")
results = cross_val_predict(model.best_estimator_, X, y, cv = 3, method = 'predict_proba')[:, 1]
score = gini_normalized(y, results)
logger.info("normalized gini score on training set is {}".format(score))

model.best_estimator_.fit(X, y)
logger.info("Loading and predicting on Test set...")
test = load_file("test")
test['target'] = model.best_estimator_.predict_proba(test)[:, 1]
write_submission_file(test, columns = ['target'], name = 'xgb-rfe')

logger.info("Finished with time {}".format(datetime.now() - start))
