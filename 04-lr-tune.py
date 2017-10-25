from lib.data import load_file
from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.porto.feature_type import get_cat_features_idx
from lib.scoring.gini import gini_normalized
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, Imputer
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from datetime import datetime

start = datetime.now()
logger = get_logger()

logger.info("Loading training data into X and y...")
train = load_file()
X = train.drop(['target'], axis = 1)
y = train.target
cat_columns = get_cat_features_idx(X)

logger.info("Making pipeline...")
pipe = Pipeline([('impute', Imputer(missing_values = -1)),
                 ('encode', OneHotEncoder(categorical_features=cat_columns, handle_unknown = 'ignore')),
                 ('to_dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
                 ('model', LogisticRegression())])
param_grid = {
    'impute__strategy': ["most_frequent"],
    'model': [LogisticRegression()],
    'model__C': [1.],
    'model__n_jobs': [1]
}

logger.info("Finding best parameters with GridSearchCV...")
model = GridSearchCV(pipe, param_grid, scoring = 'roc_auc')

logger.info("Fitting model on X...")
model.fit(X.as_matrix(), y)
logger.info("Best Params: {}".format(model.best_params_))

logger.info("Predicting score (w/Cross-Val) on X...")
results = cross_val_predict(model, X, y, method = 'predict_proba')[:, 1]
score = gini_normalized(y, results)
logger.info("normalized gini score on training set is {}".format(score))

logger.info("Loading and predicting on Test set...")
test = load_file("test")
test['target'] = model.predict_proba(test.as_matrix())[:, 1]
write_submission_file(test, columns = ['target'], name = 'ohe-cv-pipe')

logger.info("Finished with time {}".format(datetime.now() - start))
