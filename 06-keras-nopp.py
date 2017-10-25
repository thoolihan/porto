from lib.data import load_file
from lib.submit import write_submission_file
from lib.logger import get_logger
from lib.config import get_config
from lib.porto.feature_type import get_cat_features_idx
from lib.scoring.gini import gini_normalized
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, Imputer, FunctionTransformer
from sklearn.model_selection import train_test_split
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation

start = datetime.now()
cfg = get_config()
logger = get_logger()

logger.info("Loading training data into X and y...")
train = load_file()
X = train.drop(['target'], axis = 1)
y = train.target
n = X.shape[1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .3)

logger.info("Creating Keras Model...")
model = Sequential()
model.add(Dense(units = n, input_dim = n))
model.add(Activation('relu'))
model.add(Dense(units = 64, input_dim = n))
model.add(Activation('relu'))
model.add(Dense(units = 64, input_dim = n))
model.add(Activation('relu'))
model.add(Dense(units = 64, input_dim = n))
model.add(Activation('relu'))
model.add(Dense(units = 64, input_dim = n))
model.add(Activation('relu'))
model.add(Dense(units = 1))
model.add(Activation('sigmoid'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

logger.info("Fitting model on X_train...")
history = model.fit(X_train.as_matrix(), y_train, epochs = cfg["epochs"], batch_size = cfg["batch_size"])

logger.info("Predicting on X_val...")
results_val = model.predict(X_val.as_matrix())
score = gini_normalized(y_val, results_val)
logger.info("normalized gini score on validation set is {}".format(score))

logger.info("Loading and predicting on Test set...")
X_test = load_file("test")
X_test['target'] = model.predict(X_test.as_matrix())
write_submission_file(X_test, columns = ['target'], name = 'keras-v1')

logger.info("Finished with time {}".format(datetime.now() - start))
