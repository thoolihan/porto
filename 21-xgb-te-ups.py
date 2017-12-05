from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.data import load_file
from lib.config import get_config
from lib.scoring.gini import gini_normalized
from lib.target_encode import TargetEncoder
from lib.porto.features import drop_cols
from lib.porto.feature_type import get_cat_features
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier
import time

start = time.time()
logger = get_logger()
cfg = get_config()

logger.info("Loading training data into X and y...")
train = load_file()
X = train.drop('target', axis = 1)
drop_cols = drop_cols(X, names = True)
X.drop(drop_cols, axis = 1, inplace = True)
y = train.target
cat_columns = get_cat_features(X)

logger.info("Making Target Encoder...")
tenc = TargetEncoder(columns = cat_columns)
X = tenc.fit_transform(X, y)

# Create upsample set
pos_idx = (y == 1)
pos_count = len(y[pos_idx])
# capped at first param, 2nd param would bring to roughly 50/50
upsample_magnitude = min(1, int((len(y) - pos_count) / pos_count))

X_up = X
y_up = y
for _ in range(upsample_magnitude):
    X_up = X_up.append(X[pos_idx])
    y_up = y_up.append(y[pos_idx])
logger.debug("Value counts for upsample set y_up: {}".format(y_up.value_counts()))

logger.info("Making Pipeline...")
model = Pipeline([('model', XGBClassifier(n_estimators = 800,
                                         learning_rate = 0.07,
                                         reg_alpha = 8,
                                         reg_lambda = 0.75,
                                         gamma = 3,
                                         max_depth = 4))])

logger.info("Predicting score (w/Cross-Val) on X...")
results = cross_val_predict(model, X, y, cv = cfg["folds"], method = 'predict_proba')[:, 1]
score = gini_normalized(y, results)
logger.info("normalized gini score on training set is {}".format(score))

logger.info("Fitting model on upscaled X...")
model.fit(X_up, y_up)

logger.info("Loading and predicting on Test set...")
test = load_file("test")
test.drop(drop_cols, axis = 1, inplace = True)
test = tenc.transform(test)
test['target'] = model.predict_proba(test)[:, 1]
write_submission_file(test, columns = ['target'], name = 'xgb-te-ups')

logger.info("Finished with time {:.3f} minutes".format((time.time() - start)/60.0))
