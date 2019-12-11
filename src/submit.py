import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

train_data = pd.read_csv('../processed/train_data.csv')
train_ids = train_data['animal_id_outcome']
y = train_data['Label']
X = train_data.drop(['Label','animal_id_outcome'], axis=1)

test_data = pd.read_csv('../processed/test_data.csv')
test_ids = test_data['animal_id_outcome']

X_test = test_data.drop('animal_id_outcome', axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

## best classifier found from train_default.py + grid_search.py
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

model.fit(X, y)
y_pred = model.predict(X_test)

with open('../logs/encoding.pkl', 'rb') as f:
	enc = pickle.load(f)
enc = {v: k for k, v in enc.items()}

final = [enc[x] for x in y_pred]

submit = pd.DataFrame(
	{'animal_id_outcome': test_ids,
	'outcome_type': final}
	)

submit.to_csv('../processed/final_submission.csv', index=False)
