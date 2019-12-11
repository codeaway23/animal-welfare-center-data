import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.preprocessing import LabelEncoder

def preprocessing(df, name):

	## setting index
	df.set_index('animal_id_outcome', inplace=True)

	## filling NaN values in outcomme_datetime
	in_datetime = df['intake_datetime']
	days_spent = df['time_in_shelter_days']

	if name=='train':
		dt_format = '%Y-%m-%d %H:%M:%S'
	elif name=='test':
		dt_format = '%d-%m-%Y %H:%M'

	in_datetime = in_datetime.apply(lambda x: datetime.strptime(x, dt_format))
	days_spent = days_spent.apply(lambda x: timedelta(x))

	processed_out_datetime = in_datetime + days_spent
	processed_out_datetime = processed_out_datetime.apply(lambda x: x.strftime(dt_format))

	df['outcome_datetime'] = processed_out_datetime

	## getting dates and min values from datetime data
	int_datetime = df['intake_datetime'].values
	out_datetime = df['outcome_datetime'].values
	int_date = [int(x[8:10]) for x in int_datetime]
	out_date = [int(x[8:10]) for x in out_datetime]

	if name=='train':
		int_min = [int(x[-5:-3]) for x in int_datetime]
		out_min = [int(x[-5:-3]) for x in out_datetime]
	elif name=='test':
		int_min = [int(x[-2:]) for x in int_datetime]
		out_min = [int(x[-2:]) for x in out_datetime]

	df['intake_date'] = int_date
	df['outcome_date'] = out_date
	df['intake_min'] = int_min
	df['outcome_min'] = out_min

	## getting dates from DOB values
	dob = df['date_of_birth'].values
	dob_date = [int(x[8:10]) for x in int_datetime]

	df['dob_date'] = dob_date

	## drop redundant columns
	drop_cols = ['intake_datetime',
		'outcome_datetime',
		'date_of_birth',
		'intake_monthyear',
		'outcome_monthyear',
		'time_in_shelter',
		'count',
		'age_upon_intake',
		'age_upon_outcome', 
		'outcome_number']

	df.drop(drop_cols, axis=1, inplace=True)
	df.dropna(inplace=True)

	return df

def encoding(train_df, test_df):

	# encoding target values
	le = LabelEncoder()
	train_df['outcome_type'] = le.fit_transform(train_df['outcome_type'])

	train_df.rename(columns={'outcome_type': 'Label'}, inplace=True)
	train_y = train_df['Label']

	train_df.drop('Label', axis=1, inplace=True)

	# get encoding dictionary
	keys = le.classes_
	vals = le.transform(le.classes_)
	encoding = dict(zip(keys, vals))

	# combine train test dataframes to avoid labelling discrepancies
	train_df['name'] = 'train'
	test_df['name'] = 'test'
	combined_df = pd.concat([train_df, test_df])

	to_encode_cols = ['animal_type',
		'breed',
		'color',
		'intake_condition',
		'intake_type',
		'sex_upon_intake',
		'age_upon_intake_age_group',
		'intake_weekday',
		'sex_upon_outcome',
		'age_upon_outcome_age_group',
		'outcome_weekday']
	
	# encode columns using label emcoder
	for x in to_encode_cols:
		combined_df[x] = le.fit_transform(combined_df[x])

	final_train = combined_df[combined_df.name == 'train']
	final_test = combined_df[combined_df.name == 'test']

	final_train.drop('name', axis=1, inplace=True)
	final_test.drop('name', axis=1, inplace=True)

	final_train['Label'] = train_y

	return final_train, final_test, encoding


train = pd.read_csv('../dataset/train.csv')
test = pd.read_csv('../dataset/test.csv')

train_df = preprocessing(train, name='train')
test_df = preprocessing(test, name='test')

train_data, test_data, encoding = encoding(train_df, test_df)

train_data.to_csv('../processed/train_data.csv')
test_data.to_csv('../processed/test_data.csv')

import pickle

with open('../logs/encoding.pkl', 'wb') as f:
	pickle.dump(encoding, f)
