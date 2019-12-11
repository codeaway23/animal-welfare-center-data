# Analysing data from an animal welfare center

---
## Describing the data and the problem

Data from a US animal shelter from 2013 to 2018 is provided. The data includes information about the animals taken in the shelter. A total of 36 features which include age, sex, species, breed, condition, etc upon intake as well outcome. You can find the data in the ```dataset``` directory. the ground truth data for the test set is not available. 

The task is to process the training data and make predictions about the outcome of each animal in the test data. There are 9 target labels elaborating the outcome of each animal. Died? Adopted? Euthanized? Returned to owner?

## Methodology

**Feature Engineering:** modifying the dataset to make it fit for training our models over. Changing variables to a more usable format.

**Modelling:** training and finding out which models work for our dataset.

**Hyperparameter Tuning:** using the best models found and tune their hyperparameters to obtain the best results.

**Submission:** producing results for the test data in the format required.

### Feature Engineering

1. **Treating NaN values**
- NaN values only found in one column: ‘outcome_datetime’
- Rather than getting rid of the missing values, it was possible to fill them in by doing some basic operations on some other features available. 

2. **Processing date-time data** 
- Date-time data for train and test sets were found to be in a different format. 
- Splitting these data-time values into numerical features like date, day, month, hour, min. 

3. **Processing DoB data**
- Getting date values

4. **Removing redundant features**
- Age of the animals was repeated.
- Time spent in shelter was repeated.
- Count was always equal to 1.

5. **Encoding categorical variables**
- Total of 11 features in string format.
- Encoding using a label encoder
- Dealing with unseen values in test data.

### Modelling

18 classifiers tested in default settings

**Top 5 classifiers:**
- GradientBoostingClassifier 
- XGBClassifier
- MLPClassifier
- RandomForestClassifier 
- SVC

For this phase of modelling 
- train_test_split : 0.2
- Best classifier: GradientBoostingClassifier
- Accuracy: 62.36%

### Hyperparameter tuning

Hyperparameter tuning involved fitting the data on 139 models. (GridSearchCV)

**Best model: XGBClassifier**

**Best hyperparameters found:**
- n_estimators: 50
- learning_rate: 0.1
- max_depth: 4
- cross_validation: 3

**Best model results**
Accuracy: 60.66%

**Results on test set**
Accuracy: 54.05%

### Submission

The model mentioned above wasn't th best performing on the test set. The model mentioned below performed a little better. 

**Best model: XGBClassifier**

**Best hyperparameters found:**
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 3
- cross_validation: 3

**Best model results**
Accuracy: 60.44%

**Results on test set**
Accuracy: 54.20%

---
## Suggestions and Improvements

1. **Feature Engineering**
- Treating outliers
- Dealing with class imbalances
- Processing the breeds of animals better
- Dropping more features (the highly correlated ones) for a better, more concise dataset

2. **Feature selection**
- Faced a problem when trained models gave different results when compared to chi2 tests

3. **Better hyperparameter tuning**
- Hyperopt
- RandomizedSearchCV
- Varying more hyperparameters 
- Increasing granularity of the search process

## Plots

1. **Correlation matrix heatmap**
![correlation map](https://github.com/codeaway23/animal-welfare-center-data/blob/master/plots/corr_matrix_heatmap.png)

2. **Grid search results**
![grid search results](https://github.com/codeaway23/animal-welfare-center-data/blob/master/plots/grid_search_results.png)

3. **XGBoost top 10 features**
![alt text](https://github.com/codeaway23/animal-welfare-center-data/blob/master/plots/xgb_top_10_features.png)

