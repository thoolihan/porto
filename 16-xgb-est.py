from lib.data import load_file
from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.config import get_config
from lib.scoring.gini import gini_normalized
from lib.porto.features import drop_cols
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Imputer
from sklearn.model_selection import cross_val_predict, GridSearchCV
from scipy.sparse import csc_matrix
from xgboost import XGBClassifier
import numpy as np
import time

start = time.time()
logger = get_logger()
cfg = get_config()

logger.info("Loading training data into X and y...")
train = load_file()
X = train.drop('target', axis = 1)
drop_idx = drop_cols(X)
y = train.target

logger.info("Making GridSearchCV Pipeline...")
pipe = Pipeline([('drops', FunctionTransformer(lambda mat: np.delete(mat, drop_idx, axis = 1))),
                 ('model', XGBClassifier())])
param_grid = {
    'model__n_estimators': [550],
    'model__learning_rate': [0.07],
    'model__reg_alpha': [8],
    'model__reg_lambda': [0.75],
    'model__gamma': [3, 4, 5],
    'model__max_depth': [4]
}

model = GridSearchCV(pipe, param_grid, cv = cfg["folds"], scoring = 'roc_auc')

logger.info("Fitting model on X...")
model.fit(X, y)
logger.info("Best Params: {}".format(model.best_params_))

logger.info("Predicting score (w/Cross-Val) on X...")
results = cross_val_predict(model.best_estimator_, X, y, cv = cfg["folds"], method = 'predict_proba')[:, 1]
score = gini_normalized(y, results)
logger.info("normalized gini score on training set is {}".format(score))

logger.info("Loading and predicting on Test set...")
test = load_file("test")
test['target'] = model.best_estimator_.predict_proba(test)[:, 1]
write_submission_file(test, columns = ['target'], name = 'xgb-est')

logger.info("Finished with time {}".format(time.time() - start))
