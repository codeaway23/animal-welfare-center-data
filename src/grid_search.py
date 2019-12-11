import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from xgboost.sklearn import XGBClassifier

def grid_search(X, y, models, params, cv=3, n_jobs=3):
	gsdict = {}
	keys = list(models.keys())
	for k in keys:
		print("running grid search for {}".format(k))
		gs = GridSearchCV(models[k], params[k], cv=cv, 
						scoring='f1_micro', n_jobs=n_jobs, verbose=1, 
						return_train_score=True, refit=False)
		gs.fit(X,y)
		gsdict[k] = gs
	return gsdict

def score_df(gsdict):
	scores = {}
	params = {}
	df = pd.DataFrame(columns=['key', 'params', 'f1_score'])
	keys = list(gsdict.keys())
	for k in keys:
		params[k] = gsdict[k].cv_results_['params']
		scores[k] = []
		for i in range(gsdict[k].cv):
			scores[k].append(gsdict[k].cv_results_['split{}_test_score'.format(i)])
		scores[k] = np.mean(np.array(scores[k]), axis=0)
		for i,p in enumerate(params[k]):
			df = df.append({'key':k, 'params':p, 'f1_score':scores[k][i]}, ignore_index=True)
	return df


train_data = pd.read_csv('../processed/train_data.csv')
ids = train_data['animal_id_outcome']
y = train_data['Label']
X = train_data.drop(['Label','animal_id_outcome'], axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)


# using top 5 models as found from train_default.py

models = {
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'XGBClassifier': XGBClassifier(),
    'MLPClassifier': MLPClassifier(),
    'SVC': SVC()
}

# hyper-parameters to tune for all models

estim = [25, 50, 100, 200, 1000]
lr = [0.01, 0.1, 1.0]
max_depth = [2, 3, 4]
hidden_layers = [(8,), (16,), (32,), (64,), (128,)]
act = ['relu', 'tanh']
crit = ['gini', 'entropy']
C = [1, 5, 10]
gamma = [0.01, 0.001, 0.0001]

params = {
    'GradientBoostingClassifier': { 'n_estimators': estim, 'learning_rate': lr, 'max_depth': max_depth  },
    'RandomForestClassifier': {'criterion': crit, 'n_estimators': estim , 'max_depth': max_depth },
    'XGBClassifier': { 'n_estimators': estim, 'learning_rate': lr , 'max_depth': max_depth },
    'MLPClassifier': { 'activation': act, 'hidden_layer_sizes': hidden_layers },
    'SVC': {'C': C, 'gamma': gamma}
}

# running grid search

gsdict = grid_search(X, y, models, params)
scores = score_df(gsdict)
scores.sort_values(by='f1_score', ascending=False, inplace=True)

print(scores)

with open('../logs/grid_search.pkl','wb') as f:
	pickle.dump(gsdict, f)

scores.to_csv('../logs/grid_search_scores.csv', index=False)
