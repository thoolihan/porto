from lib.data import load_file, convert_columns_to_int
from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.porto.feature_type import get_bin_cat_features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV
from scoring.gini import gini_normalized

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
                 ('model', LogisticRegression())])

param_grid = {}
    # 'model': [
    #     LogisticRegression(),
    #     KNeighborsClassifier(n_neighbors=5)
    # ]
#}

model = GridSearchCV(pipe, param_grid, scoring = 'f1')
model.fit(X, y)
logger.info(model.best_params_)

results = cross_val_predict(model, X, y, method = 'predict_proba')[:, 1]
score = gini_normalized(y, results)
logger.info("Cross-val normalized gini score on training set is {}".format(score))

# test data
X_test = convert_columns_to_int(load_file("test"), bit_columns)

# predict
y_test_pred = model.predict_proba(X_test)
X_test['target'] = y_test_pred[:,1]
write_submission_file(X_test, columns = ['target'], name = 'cv-mvp')