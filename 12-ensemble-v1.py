from lib.data import load_file
from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.porto.features import drop_cols
from lib.porto.feature_type import get_bin_cat_features, get_cat_features_idx
from lib.scoring.gini import gini_normalized
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Imputer, OneHotEncoder
from lightgbm.sklearn import LGBMClassifier
from xgboost import XGBClassifier
from datetime import datetime
import numpy as np

start = datetime.now()
logger = get_logger()

logger.info("Loading training data into X and y...")
train = load_file()
X = train.drop('target', axis = 1)
drop_cols = drop_cols(X, names = True)
X.drop(drop_cols, axis = 1, inplace = True)
y = train.target
cat_columns = get_cat_features_idx(X)

logger.info("Making Ensemble...")
classifiers = [('xgb', XGBClassifier(learning_rate=0.07, reg_alpha=8, reg_lambda=0.75, max_depth=4, n_estimators = 800, gamma = 3)),
               ('lgbm', LGBMClassifier(learning_rate = 0.018, max_depth = 6, num_leaves = 11, col_sample_by_tree=0.85)),
               ('rf', RandomForestClassifier(n_estimators = 200, criterion = 'gini'))]

model = Pipeline([('impute', Imputer(missing_values = -1, strategy = "most_frequent")),
                  ('encode', OneHotEncoder(categorical_features=cat_columns, handle_unknown = 'ignore')),
                  ('ensemble', VotingClassifier(estimators = classifiers, voting = 'soft'))])

logger.info("Fitting model on X...")
model.fit(X, y)

logger.info("Predicting score (w/Cross-Val) on X...")
results = cross_val_predict(model, X, y, cv = 3, method = 'predict_proba')[:, 1]
score = gini_normalized(y, results)
logger.info("normalized gini score on training set is {}".format(score))

logger.info("Loading and predicting on Test set...")
test = load_file("test")
test.drop(drop_cols, axis = 1, inplace = True)
test['target'] = model.predict_proba(test)[:, 1]
write_submission_file(test, columns = ['target'], name = 'ensemble-v1')

logger.info("Finished with time {}".format(datetime.now() - start))
