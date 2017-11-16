from lib.data import load_file
from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.porto.features import drop_cols
from lib.scoring.gini import gini_normalized
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from xgboost import XGBClassifier
from datetime import datetime

start = datetime.now()
logger = get_logger()

logger.info("Loading training data into X and y...")
train = load_file()
X = drop_cols(train)
y = train.target

logger.info("Making Ensemble...")
classifiers = [('xgb', XGBClassifier(learning_rate=0.095, reg_alpha=0.35, reg_lambda=0.75, max_depth=5)),
               ('lgbm', LGBMClassifier()),
               ('rf', RandomForestClassifier())]

model = VotingClassifier(estimators = classifiers, voting = 'soft')

logger.info("Fitting model on X...")
model.fit(X, y)

logger.info("Predicting score (w/Cross-Val) on X...")
results = cross_val_predict(model, X, y, cv = 3, method = 'predict_proba')[:, 1]
score = gini_normalized(y, results)
logger.info("normalized gini score on training set is {}".format(score))

logger.info("Loading and predicting on Test set...")
test = load_file("test")
test['target'] = model.predict_proba(test)[:, 1]
write_submission_file(test, columns = ['target'], name = 'xgb-v2')

logger.info("Finished with time {}".format(datetime.now() - start))
