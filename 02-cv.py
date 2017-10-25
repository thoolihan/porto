from lib.data import load_file, convert_columns_to_int
from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.porto.feature_type import get_bin_cat_features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, GridSearchCV
from lib.scoring.gini import gini_normalized

logger = get_logger()

# training data
train = load_file()
bit_columns = get_bin_cat_features(train)
bit_columns.append('target')
train = convert_columns_to_int(train, bit_columns)
X = train.drop(['target'], axis = 1)
y = train.target

# make a pipeline
pipe = Pipeline([('transform', StandardScaler()),
                 ('model', GaussianNB())])

param_grid = {}

model = GridSearchCV(pipe, param_grid, scoring = 'roc_auc')
model.fit(X, y)
logger.info("Best Params: {}".format(model.best_params_))

results = cross_val_predict(model, X, y, method = 'predict_proba')[:, 1]
score = gini_normalized(y, results)
logger.info("Cross-val normalized gini score on training set is {}".format(score))

# test data
X_test = convert_columns_to_int(load_file("test"), bit_columns)

# predict
X_test['target'] = model.predict_proba(X_test)[:, 1]
write_submission_file(X_test, columns = ['target'], name = 'cv-mvp')