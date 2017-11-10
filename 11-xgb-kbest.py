from lib.data import load_file
from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.config import get_config
from lib.scoring.gini import gini_normalized
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, RFECV, SelectKBest
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

model = Pipeline([('features', SelectKBest(k = 40)),
                 ('model', XGBClassifier(learning_rate = .095, reg_alpha = .35, reg_lambda = .76, max_depth = 5))])

logger.info("Predicting score (w/Cross-Val) on X...")
results = cross_val_predict(model, X, y, cv = 2, method = 'predict_proba')[:, 1]
score = gini_normalized(y, results)
logger.info("normalized gini score on training set is {}".format(score))

logger.info("Fitting model on X...")
model.fit(X, y)

logger.info("Loading and predicting on Test set...")
test = load_file("test")
test['target'] = model.predict_proba(test)[:, 1]
write_submission_file(test, columns = ['target'], name = 'xgb-rfe')

logger.info("Finished with time {}".format(datetime.now() - start))
