import pandas as pd
import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV

from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# lsit of classifiers to test in default settings
classifiers = {
	'BernoulliNB': BernoulliNB(),
	'DecisionTreeClassifier': DecisionTreeClassifier(),
	'ExtraTreeClassifier': ExtraTreeClassifier(),
	'ExtraTreesClassifier': ExtraTreesClassifier(),
	'GaussianNB': GaussianNB(),
	'KNeighborsClassifier': KNeighborsClassifier(3),
	'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
	'SVC': SVC(),
	'LogisticRegression': LogisticRegression(solver='newton-cg', multi_class='multinomial'),
	'MLPClassifier': MLPClassifier(),
	'NearestCentroid': NearestCentroid(),
	'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
	'RandomForestClassifier': RandomForestClassifier(),
	'GradientBoostingClassifier': GradientBoostingClassifier(),
	'AdaBoostClassifier': AdaBoostClassifier(),
	'RidgeClassifier': RidgeClassifier(),
	'RidgeClassifierCV': RidgeClassifierCV(),
	'XGBClassifier': XGBClassifier()
	}


train_data = pd.read_csv('../processed/train_data.csv')
ids = train_data['animal_id_outcome']
y = train_data['Label']
X = train_data.drop(['Label','animal_id_outcome'], axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# testing each classifier, storing their f1 scores (micro)
f1_scores = []
for clf in classifiers.values():
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	acc = f1_score(y_test, y_pred, average='micro')
	f1_scores.append(acc)

# getting top 5 models
idx = np.argsort(f1_scores)[::-1]
best_model_keys = [list(classifiers.keys())[x] for x in idx]
best_model_values = [classifiers[x] for x in best_model_keys]
best_classifiers = dict(zip(best_model_keys, best_model_values))
sorted_f1 = [f1_scores[x] for x in idx]

print("Top 5 classifiers in default settings...")
for i in range(5):
	print(list(best_classifiers.keys())[i], "F1 SCORE: ", sorted_f1[i])

