from lib.data import load_file, convert_columns_to_int, make_missing_zero
from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.porto.feature_type import get_bin_cat_features, get_cat_features_idx
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from lib.scoring.gini import gini_normalized

logger = get_logger()

logger.info("Loading training data into X and y...")
train = load_file()
X = train.drop(['target'], axis = 1)
y = train.target

logger.info("Bumping all values up 1, so missing is now zero...")
cat_columns = get_cat_features_idx(X)
X = make_missing_zero(X, cat_columns)

logger.info("Making pipeline...")
pipe = Pipeline([('encode', OneHotEncoder(categorical_features=cat_columns, handle_unknown = 'ignore')),
                 ('to_dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
                 ('model', LogisticRegression())])
param_grid = {
    'model': [LogisticRegression()],
    'model__C': [.25, .5, 1., 1.5],
    'model__n_jobs': 4,
    'model__solver': ['sag', 'saga'],
}

logger.info("Finding best parameters...")
model = GridSearchCV(pipe, param_grid, scoring = 'roc_auc')

logger.info("Fitting on X...")
model.fit(X.as_matrix(), y)
logger.info("Best Params: {}".format(model.best_params_))

logger.info("Cross-val predict scoring on X...")
results = cross_val_predict(model, X, y, method = 'predict_proba')[:, 1]
score = gini_normalized(y, results)
logger.info("normalized gini score on training set is {}".format(score))

logger.info("Loading and predicting on Test set...")
test = make_missing_zero(load_file("test"), cat_columns)
test['target'] = model.predict_proba(test.as_matrix())[:, 1]
write_submission_file(test, columns = ['target'], name = 'ohe-cv-pipe')