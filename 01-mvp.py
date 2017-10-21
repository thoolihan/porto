from lib.data import load_file, convert_columns_to_int
from lib.submit import write_submission_file
from lib.logger import get_logger
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logger = get_logger()

# target columns / features
chi2_df = load_file("chi2")

def n_best(chdf, n = 15):
    sorted = chdf.sort_values('chi2', axis = 0, ascending = False)
    return sorted['feature'][:n]

columns = n_best(chi2_df)

# training data
bit_columns = ['target']
train = convert_columns_to_int(load_file(), bit_columns)
X = train[columns]
y = train.target

# make a pipeline
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X, y)

# test data
test = load_file("test")
X_test = test[columns]

# predict
y_test_pred = pipe.predict_proba(X_test)
test['target'] = y_test_pred[:,1]
write_submission_file(test, columns = ['target'], name = 'mvp')