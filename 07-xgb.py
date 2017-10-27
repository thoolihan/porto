from lib.data import load_file
from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.config import get_config
from lib.scoring.gini import gini_normalized
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, GridSearchCV
from xgboost import XGBClassifier
from datetime import datetime

start = datetime.now()
logger = get_logger()

logger.info("Loading training data into X and y...")
train = load_file()
X = train.drop(['target'], axis = 1)
y = train.target

logger.info("Making GridSearchCV Pipeline...")
pipe = Pipeline([('model', XGBClassifier())])
param_grid = {}

model = GridSearchCV(pipe, param_grid, scoring = 'roc_auc')

logger.info("Fitting model on X...")
model.fit(X, y)
logger.info("Best Params: {}".format(model.best_params_))

logger.info("Predicting score (w/Cross-Val) on X...")
results = cross_val_predict(model.best_estimator_, X, y, cv = 3, method = 'predict_proba')[:, 1]
score = gini_normalized(y, results)
logger.info("normalized gini score on training set is {}".format(score))

logger.info("Loading and predicting on Test set...")
test = load_file("test")
test['target'] = model.predict_proba(test)[:, 1]
write_submission_file(test, columns = ['target'], name = 'xgb-v1')

logger.info("Finished with time {}".format(datetime.now() - start))
