import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set seed
np.random.seed(123)

# Import training data
trn = pd.read_csv('CW1_train_cleaned.csv')
X_tst = pd.read_csv('CW1_test_cleaned.csv') 

# Train your model (using Random Forest here)
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']

# make sure test set has the same columns as training set (after encoding)
X_tst = X_tst[X_trn.columns]

# tree number: n_estimators=200
model = RandomForestRegressor(n_estimators=200, random_state=123)
model.fit(X_trn, y_trn)

# Test set predictions
yhat_rf = model.predict(X_tst)

# Format submission:
# This is a single-column CSV with nothing but your predictions
out = pd.DataFrame({'yhat': yhat_rf})
out.to_csv('CW1_submission_23172173.csv', index=False)