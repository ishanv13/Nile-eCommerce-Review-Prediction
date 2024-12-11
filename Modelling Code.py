# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 02:39:15 2024

@author: Group 23
"""


import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import uniform, randint
# we get a load of warnings running the code so will supress them
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import ConfusionMatrixDisplay as CM

####################################################################################################################################

# Importing the dataset with all 18 features
df = pd.read_csv('df.csv')
df = df.drop(columns = 'review_score')

# One-hot encode and drop original columns
df = pd.get_dummies(df, columns=['customer_state', 'product_category_name'], drop_first=True)

# Defining the decision variable
y_value = df['class'] 
y_values = np.ravel(y_value) 

# The rest of the features are our independent variables
x_values = df.drop('class', axis=1)

# splitting data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(x_values, y_value, test_size = 0.2, random_state=4567, stratify=y_value)

# Undersampling the majority class
from imblearn.under_sampling import RandomUnderSampler

# how many of each class in training dataset
print(Y_train.value_counts()) # 1:0 = 5:1

rus = RandomUnderSampler(random_state=42, sampling_strategy='auto')
X_train, Y_train = rus.fit_resample(X_train, Y_train)
pd.Series(Y_train).value_counts()


# displaying the shape of our test and training data
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
  
GBDT_algo = GBDT()
GBDT_model = GBDT_algo.fit(X_train, Y_train)

########################################################################################################################################

# Model 1: 18 Feature Model with Macro Recall Scoring Criteria

def random_search(algo, hyperparameters, X_train, Y_train):
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall_macro', n_iter=12, refit=True)
  
  clf.fit(X_train, Y_train)
  return clf.best_params_

# Creating the HyperParameter Dictionary
GBDT_tuned_parameters = {
    'n_estimators': randint(25, 250), # Draw from a uniform distribution between 50 and 500
    'learning_rate': uniform(loc=0.01, scale=4.99),  # Draw from a uniform distribution between 0.01 and 5
    'criterion': ['friedman_mse', 'squared_error'],
    'max_depth': randint(2, 7)  # Draw from a uniform distribution between 2 and 7
}

GBDT_1_best_params = random_search(GBDT_algo, GBDT_tuned_parameters, X_train, Y_train)

GBDT_1_algo = GBDT(**GBDT_1_best_params)
GBDT_1_model = GBDT_1_algo.fit(X_train, Y_train)


# Calculate the Training Scores
predict_1 = GBDT_1_model.predict(X_train)
precision_training_1_macro, recall_training_1_macro, f1_score_training_1_macro, _ = precision_recall_fscore_support(Y_train, predict_1, average='macro')
precision_training_1, recall_training_1, f1_score_training_1, _ = precision_recall_fscore_support(Y_train, predict_1)
names = ['GBDT']  

# Calculate the Testing Scores
predict_testing_1 = GBDT_1_model.predict(X_test)
precision_testing_1_macro, recall_testing_1_macro, f1_score_testing_1_macro, _ = precision_recall_fscore_support(Y_test, predict_testing_1, average='macro')
precision_testing_1, recall_testing_1, f1_score_testing_1, _ = precision_recall_fscore_support(Y_test, predict_testing_1)
  
# Disaply the confusion matrix
print("GBDT Confusion Matrix")
print(CM.from_predictions(Y_test, predict_testing_1))

#######################################################################################################################################################

#  Model 2: 18 Feature Model with Recall Scoring Criteria

def random_search(algo, hyperparameters, X_train, Y_train):
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall', n_iter=12, refit=True)

  clf.fit(X_train, Y_train)
  return clf.best_params_

GBDT_2_best_params = random_search(GBDT_algo, GBDT_tuned_parameters, X_train, Y_train)

GBDT_2_algo = GBDT(**GBDT_2_best_params)
GBDT_2_model = GBDT_2_algo.fit(X_train, Y_train)

# Calculate the Training Scores
predict_2 = GBDT_2_model.predict(X_train)
precision_training_2_macro, recall_training_2_macro, f1_score_training_2_macro, _ = precision_recall_fscore_support(Y_train, predict_2, average='macro')
precision_training_2, recall_training_2, f1_score_training_2, _ = precision_recall_fscore_support(Y_train, predict_2)

# Calculate the Testing Scores
predict_testing_2 =GBDT_2_model.predict(X_test)
precision_testing_2_macro, recall_testing_2_macro, f1_score_testing_2_macro, _ = precision_recall_fscore_support(Y_test, predict_testing_2, average='macro')
precision_testing_2, recall_testing_2, f1_score_testing_2, _ = precision_recall_fscore_support(Y_test, predict_testing_2)

# Disaply the confusion matrix
print("GBDT Confusion Matrix")
print(CM.from_predictions(Y_test, predict_testing_2))

##################################################################################################################################
# Data Preparation for 5 Feature Models

# Checking the importances of the features used in the model
GBDT_macro_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': GBDT_1_model.feature_importances_})
GBDT_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': GBDT_2_model.feature_importances_})

# Selecting the top 5 features to keep in the model, removing the rest of the features
df = pd.read_csv('df.csv')
df = df.drop(columns = ['review_score','multiple_orders','product_vol','product_category_name','unit_price','payment_type_count','payment_type_voucher','payment_type_debit_card','payment_type_credit_card','payment_type_boleto','payment_installments','payment_value','customer_state','delivery_carrier_customer'])

y_value = df['class'] 
y_values = np.ravel(y_value) 
x_values = df.drop('class', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(x_values, y_value, test_size = 0.2, random_state=4567, stratify=y_value)

rus = RandomUnderSampler(random_state=42, sampling_strategy='auto')
X_train, Y_train = rus.fit_resample(X_train, Y_train)
pd.Series(Y_train).value_counts()

# displaying the shape of our test and training data
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
  
GBDT_algo = GBDT()
GBDT_model = GBDT_algo.fit(X_train, Y_train)
################################################################################################################################

#  Model 3: 5 Feature Model with Macro Recall Scoring Criteria

def random_search(algo, hyperparameters, X_train, Y_train):
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall_macro', n_iter=12, refit=True)
  
  clf.fit(X_train, Y_train)
  return clf.best_params_

GBDT_3_best_params = random_search(GBDT_algo, GBDT_tuned_parameters, X_train, Y_train)

GBDT_3_algo = GBDT(**GBDT_3_best_params)
GBDT_3_model = GBDT_3_algo.fit(X_train, Y_train)

# Calculate the Training Scores
predict_3 = GBDT_3_model.predict(X_train)
precision_training_3_macro, recall_training_3_macro, f1_score_training_3_macro, _ = precision_recall_fscore_support(Y_train, predict_3, average='macro')
precision_training_3, recall_training_3, f1_score_training_3, _ = precision_recall_fscore_support(Y_train, predict_3)

# Calculate the Testing Scores
predict_testing_3 =GBDT_3_model.predict(X_test)
precision_testing_3_macro, recall_testing_3_macro, f1_score_testing_3_macro, _ = precision_recall_fscore_support(Y_test, predict_testing_3, average='macro')
precision_testing_3, recall_testing_3, f1_score_testing_3, _ = precision_recall_fscore_support(Y_test, predict_testing_3)

# Disaply the confusion matrix
print("GBDT Confusion Matrix")
print(CM.from_predictions(Y_test, predict_testing_3))

#################################################################################################################################

#  Model 4: 18 Feature Model with Recall Scoring Criteria

def random_search(algo, hyperparameters, X_train, Y_train):
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall', n_iter=12, refit=True)
  
  clf.fit(X_train, Y_train)
  return clf.best_params_

GBDT_4_best_params = random_search(GBDT_algo, GBDT_tuned_parameters, X_train, Y_train)

GBDT_4_algo = GBDT(**GBDT_4_best_params)
GBDT_4_model = GBDT_4_algo.fit(X_train, Y_train)

# Calculate the Training Scores
predict_4 = GBDT_4_model.predict(X_train)
precision_training_4_macro, recall_training_4_macro, f1_score_training_4_macro, _ = precision_recall_fscore_support(Y_train, predict_4, average='macro')
precision_training_4, recall_training_4, f1_score_training_4, _ = precision_recall_fscore_support(Y_train, predict_4)

# Calculate the Testing Scores
predict_testing_4 =GBDT_4_model.predict(X_test)
precision_testing_4_macro, recall_testing_4_macro, f1_score_testing_4_macro, _ = precision_recall_fscore_support(Y_test, predict_testing_4, average='macro')
precision_testing_4, recall_testing_4, f1_score_testing_4, _ = precision_recall_fscore_support(Y_test, predict_testing_4)

# Disaply the confusion matrix
print("GBDT Confusion Matrix")
print(CM.from_predictions(Y_test, predict_testing_4))

######################################################################################################################################

# Data Preparation for Multiclass Models

df = pd.read_csv('df.csv')
df = df.drop(columns = ['class'])
df['review_score'] = df['review_score'].astype('category')
df.info()

df = pd.get_dummies(df, columns=['customer_state', 'product_category_name'], drop_first=True)

y_value = df['review_score'] 
y_values = np.ravel(y_value) 
x_values = df.drop('review_score', axis=1)

# splitting data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(x_values, y_value, test_size = 0.2, random_state=4567, stratify=y_value)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
  
GBDT_algo = GBDT()
GBDT_model = GBDT_algo.fit(X_train, Y_train)

#########################################################################################################################################

# Model 5: 18 Feature Multiclass Model with Macro Recall Scoring Criteria

def random_search(algo, hyperparameters, X_train, Y_train):
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall_macro', n_iter=12, refit=True)

  clf.fit(X_train, Y_train)
  return clf.best_params_

GBDT_5_best_params = random_search(GBDT_algo, GBDT_tuned_parameters, X_train, Y_train)

GBDT_5_algo = GBDT(**GBDT_5_best_params)
GBDT_5_model = GBDT_5_algo.fit(X_train, Y_train)


# Calculate the Training Scores
predict_5 = GBDT_5_model.predict(X_train)
precision_training_5_macro, recall_training_5_macro, f1_score_training_5_macro, _ = precision_recall_fscore_support(Y_train, predict_5, average='macro')
precision_training_5, recall_training_5, f1_score_training_5, _ = precision_recall_fscore_support(Y_train, predict_5)
names = ['GBDT']  

# Calculate the Testing Scores
predict_testing_5 =GBDT_5_model.predict(X_test)
precision_testing_5_macro, recall_testing_5_macro, f1_score_testing_5_macro, _ = precision_recall_fscore_support(Y_test, predict_testing_5, average='macro')
precision_testing_5, recall_testing_5, f1_score_testing_5, _ = precision_recall_fscore_support(Y_test, predict_testing_5)
  
# Disaply the confusion matrix
print("GBDT Confusion Matrix")
print(CM.from_predictions(Y_test, predict_testing_5))

#########################################################################################################################################################

# Model 6: 18 Feature Multiclass Model with Recall Scoring Criteria

def random_search(algo, hyperparameters, X_train, Y_train):
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall', n_iter=12, refit=True)

  clf.fit(X_train, Y_train)
  return clf.best_params_

GBDT_6_best_params = random_search(GBDT_algo, GBDT_tuned_parameters, X_train, Y_train)

GBDT_6_algo = GBDT(**GBDT_6_best_params)
GBDT_6_model = GBDT_6_algo.fit(X_train, Y_train)


# Calculate the Training Scores
predict_6 = GBDT_6_model.predict(X_train)
precision_training_6_macro, recall_training_6_macro, f1_score_training_6_macro, _ = precision_recall_fscore_support(Y_train, predict_6, average='macro')
precision_training_6, recall_training_6, f1_score_training_6, _ = precision_recall_fscore_support(Y_train, predict_6)
names = ['GBDT']  

# Calculate the Testing Scores
predict_testing_6 =GBDT_6_model.predict(X_test)
precision_testing_6_macro, recall_testing_6_macro, f1_score_testing_6_macro, _ = precision_recall_fscore_support(Y_test, predict_testing_6, average='macro')
precision_testing_6, recall_testing_6, f1_score_testing_6, _ = precision_recall_fscore_support(Y_test, predict_testing_6)
  
# Disaply the confusion matrix
print("GBDT Confusion Matrix")
print(CM.from_predictions(Y_test, predict_testing_6))



###############################################################################################################################
'''Random Forest model'''
###############################################################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
# we get a load of warnings running the code so will supress them
import warnings
from sklearn.metrics import ConfusionMatrixDisplay as CM

# join data to translate category names
df = pd.read_csv('df.csv')
df = df.drop(columns = 'review_score')

trans = pd.read_csv('product_category_name_translation.csv')
trans_df = df.merge(trans, how = 'left', left_on = 'product_category_name', right_on = 'product_category_name')

#drop the original name
trans_df = trans_df.drop(columns=['product_category_name'])

# rename the english version variable to shorter form for later encoding
trans_df.rename(columns={'product_category_name_english': 'p_category_'}, inplace=True)

'''implement ONE-HOT encoding'''

df_encoded = pd.get_dummies(trans_df, columns=['customer_state', 'p_category_'])

# Convert only dummy columns to integers 0 and 1s
dummy_columns = [col for col in df_encoded if col.startswith('customer_state_') or col.startswith('p_category_')]
df_encoded[dummy_columns] = df_encoded[dummy_columns].astype(int)

#df_encoded.to_csv('onehot_df.csv', index=False)

'''splitting train-test dataset''' 

y_value = df_encoded['class'] # set the y
y_values = np.ravel(y_value)
# drop the y from the dataframe
x_values = df_encoded.drop('class', axis=1)
# split data
X_train, X_test, Y_train, Y_test = train_test_split(x_values, y_value, test_size = 0.2, random_state=4567, stratify=y_value)

'''undersampling mjority class'''

# how many of each class in training dataset
print(Y_train.value_counts()) # 1:0 = 5:1

rus = RandomUnderSampler(random_state=42, sampling_strategy='auto')
X_train, Y_train = rus.fit_resample(X_train, Y_train)
pd.Series(Y_train).value_counts()

'''RF training and testing'''
RF_algo = RF()
RF_model = RF_algo.fit(X_train, Y_train)


############################################################################################################################
''' Hyperparameter Random searched '''

warnings.filterwarnings("ignore")

RF_tuned_parameters = {
    'n_estimators': randint(50, 500), # Draw from a uniform distribution between 50 and 500
    'max_depth': randint(2, 7),  # Draw from a uniform distribution between 2 and 7
    'min_samples_split': randint(2, 7),  # Draw from a uniform distribution between 2 and 7
    'max_features': ['sqrt', 'log2', None]
}

'''============================================================================================================='''
# Model 1: 18 Feature Model with Macro Recall Scoring Criteria

# create a hyperparameter search function for re-usability
def random_search(algo, hyperparameters, X_train, Y_train):
  # do the search using 5 folds/chunks
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall_macro', n_iter=15, refit=True)

  # pass the data to fit/train
  clf.fit(X_train, Y_train)

  return clf.best_params_

# search and save best_parameters
RF_1_best_params = random_search(RF_algo, RF_tuned_parameters, X_train, Y_train)

# Train the models with random searched parameters 
RF_1_algo = RF(**RF_1_best_params)
RF_1_model = RF_1_algo.fit(X_train, Y_train)

'''implement Recall_Macro random searched RF model on Training data'''
# predict train data using random searched RF model
predict_train_1 = RF_1_model.predict(X_train)

# check the score of random searched model
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_1, average='macro')
print(f"Macro Precision: {precision}") #0.662
print(f"Macro Recall: {recall}") #0.6478
print(f"Macro F1-score: {f1_score}") # 0.640

precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_1)
print(f"Macro Precision: {precision}") # [0.70993935 0.61414894]
print(f"Macro Recall: {recall}") # [0.50010337 0.79567225]
print(f"Macro F1-score: {f1_score}") # [0.58682732 0.69322446]

'''implement Recall_Macro random searched RF model on Test data'''
# predict based on test data
predict_test_1 = RF_1_model.predict(X_test)

# Calculate precision, recall, and F1-score for Test data
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_1, average='macro')
print(f"Macro Precision: {precision}") #0.623
print(f"Macro Recall: {recall}") #0.6459
print(f"Macro F1-score: {f1_score}") #0.6304

# Calculate precision, recall, and F1-score for Test data
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_1)
print(f"Macro Precision: {precision}") # [0.39231265 0.85411098]
print(f"Macro Recall: {recall}") # [0.50358324 0.78839539]
print(f"Macro F1-score: {f1_score}") # [0.44103802 0.81993857]

# Disaply the confusion matrix
print("RF Confusion Matrix")
print(CM.from_predictions(Y_test, predict_test_1))

'''========================================================================================================================'''
# Model 2: 18 Feature Model with Recall Scoring Criteria

# create a hyperparameter search function for re-usability
def random_search_recall(algo, hyperparameters, X_train, Y_train):
  # do the search using 5 folds/chunks
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall', n_iter=15, refit=True)

  # pass the data to fit/train
  clf.fit(X_train, Y_train)

  return clf.best_params_

# search and save best_parameters
RF_2_best_params_recall = random_search_recall(RF_algo, RF_tuned_parameters, X_train, Y_train)

# Train the models with random searched parameters 
RF_2_algo_recall = RF(**RF_2_best_params_recall)
RF_2_model = RF_2_algo_recall.fit(X_train, Y_train)

# predict train data using random searched RF model
predict_train_recall_2 = RF_2_model.predict(X_train)

'''implement Recall random searched RF on Training data'''
# check the score of random searched model, with avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_recall_2, average='macro')
print(f"Macro Precision: {precision}") #0.683
print(f"Macro Recall: {recall}") #0.6286 
print(f"Macro F1-score: {f1_score}") # 0.5987

# check the score of random searched model, without avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_recall_2)
print(f"Macro Precision: {precision}") # [0.78310329 0.58321816]
print(f"Macro Recall: {recall}") # [0.35579905 0.90145407]
print(f"Macro F1-score: {f1_score}") # [0.48929113 0.70822956]


'''implement Recall random searched RF model on Test Data'''
# predict based on test data
predict_test_recall_2 = RF_2_model.predict(X_test)

# Calculate score for Test data with avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_recall_2, average='macro')
print(f"Macro Precision: {precision}") #0.679
print(f"Macro Recall: {recall}") #0.640
print(f"Macro F1-score: {f1_score}") #0.654

# Calculate score for Test Data without avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_recall_2)
print(f"Macro Precision: {precision}") # [0.51561912 0.84259324]
print(f"Macro Recall: {recall}") # [0.37761852 0.90376851]
print(f"Macro F1-score: {f1_score}") # [0.43595863 0.87210938]

# Disaply the confusion matrix
print("RF Confusion Matrix")
print(CM.from_predictions(Y_test, predict_test_recall_2))
############################################################################################################################

# Data Preparation for RF 5 Feature Models

# Checking the importances of the features used in the model
RF_macro_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': RF_1_model.feature_importances_})
RF_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': RF_2_model.feature_importances_})

# Selecting the top 5 features to keep in the model, removing the rest of the features
df = pd.read_csv('df.csv')
df = df.drop(columns = ['review_score',
                        'delivery_carrier_customer',
                        'multiple_orders',
                        'product_vol',
                        'product_weight_g',
                        'product_category_name',
                        'customer_order_frequency',
                        'unit_price',
                        'payment_type_count',
                        'payment_type_voucher',
                        'payment_type_debit_card',
                        'payment_type_credit_card',
                        'payment_type_boleto',
                        'customer_state',
                        'delivery_carrier_customer'])

'''splitting train-test dataset''' 

y_value = df['class'] # set the y
y_values = np.ravel(y_value)
# drop the y from the dataframe
x_values = df.drop('class', axis=1)
# split data
X_train, X_test, Y_train, Y_test = train_test_split(x_values, y_value, test_size = 0.2, random_state=4567, stratify=y_value)

'''undersampling mjority class'''

# how many of each class in training dataset
print(Y_train.value_counts()) # 1:0 = 5:1

rus = RandomUnderSampler(random_state=42, sampling_strategy='auto')
X_train, Y_train = rus.fit_resample(X_train, Y_train)
pd.Series(Y_train).value_counts()


RF_algo = RF()
RF_model = RF_algo.fit(X_train, Y_train)

''' Hyperparameter Random searched '''
RF_tuned_parameters = {
    'n_estimators': randint(50, 500), # Draw from a uniform distribution between 50 and 500
    'max_depth': randint(2, 7),  # Draw from a uniform distribution between 2 and 7
    'min_samples_split': randint(2, 7),  # Draw from a uniform distribution between 2 and 7
    'max_features': ['sqrt', 'log2', None]
}

'''============================================================================================================='''
# Model 3: 5 Feature Model with Macro Recall Scoring Criteria

# create a hyperparameter search function for re-usability
def random_search(algo, hyperparameters, X_train, Y_train):
  # do the search using 5 folds/chunks
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall_macro', n_iter=15, refit=True)

  # pass the data to fit/train
  clf.fit(X_train, Y_train)

  return clf.best_params_

# search and save best_parameters
RF_3_best_params = random_search(RF_algo, RF_tuned_parameters, X_train, Y_train)

# Train the models with random searched parameters 
RF_3_algo = RF(**RF_3_best_params)
RF_3_model = RF_3_algo.fit(X_train, Y_train)

'''implement Recall_Macro random searched RF model on Training data'''
# predict train data using random searched RF model
predict_train_3 = RF_3_model.predict(X_train)

# check the score of random searched model
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_3, average='macro')
print(f"Macro Precision: {precision}") # 0.6485
print(f"Macro Recall: {recall}") # 0.6368
print(f"Macro F1-score: {f1_score}") # 0.6295

precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_3)
print(f"Macro Precision: {precision}") # [0.69034608 0.60679899]
print(f"Macro Recall: {recall}") # [0.49624423 0.77741024]
print(f"Macro F1-score: {f1_score}") # [0.57741961 0.68159024]

'''implement Recall_Macro random searched RF model on Test data'''
# predict based on test data
predict_test_3 = RF_model.predict(X_test)

# Calculate precision, recall, and F1-score for Test data
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_3, average='macro')
print(f"Macro Precision: {precision}") # 0.6166
print(f"Macro Recall: {recall}") # 0.6406
print(f"Macro F1-score: {f1_score}") # 0.6236

# Calculate precision, recall, and F1-score for Test data
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_3)
print(f"Macro Precision: {precision}") # [0.38100209 0.85235834]
print(f"Macro Recall: {recall}") # [0.50303197 0.77830118]
print(f"Macro F1-score: {f1_score}") # [0.43359468 0.81364809]

# Disaply the confusion matrix
print("RF Confusion Matrix")
print(CM.from_predictions(Y_test, predict_test_3))

'''============================================================================================================='''
# Model 4: 5 Feature Model with Recall Scoring Criteria

# create a hyperparameter search function for re-usability
def random_search_recall(algo, hyperparameters, X_train, Y_train):
  # do the search using 5 folds/chunks
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall', n_iter=15, refit=True)

  # pass the data to fit/train
  clf.fit(X_train, Y_train)

  return clf.best_params_

# search and save best_parameters
RF_4_best_params_recall = random_search_recall(RF_algo, RF_tuned_parameters, X_train, Y_train)

# Train the models with random searched parameters 
RF_4_algo_recall = RF(**RF_4_best_params_recall)
RF_4_model_recall = RF_4_algo_recall.fit(X_train, Y_train)

# predict train data using random searched RF model
predict_train_recall_4 = RF_4_model_recall.predict(X_train)

'''implement Recall random searched RF on Training data'''
# check the score of random searched model, with avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_recall_4, average='macro')
print(f"Macro Precision: {precision}") # 0.67511
print(f"Macro Recall: {recall}") # 0.6345 
print(f"Macro F1-score: {f1_score}") # 0.6120

# check the score of random searched model, without avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_recall_4)
print(f"Macro Precision: {precision}") # [0.75943647 0.59079914]
print(f"Macro Recall: {recall}") # [0.39377024 0.87526704]
print(f"Macro F1-score: {f1_score}") # [0.51862945 0.70543475]


'''implement Recall random searched RF model on Test Data'''
# predict based on test data
predict_test_recall_4 = RF_4_model_recall.predict(X_test)

# Calculate score for Test data with avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_recall_4, average='macro')
print(f"Macro Precision: {precision}") # 0.6603
print(f"Macro Recall: {recall}") # 0.64074
print(f"Macro F1-score: {f1_score}") # 0.64872

# Calculate score for Test Data without avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_recall_4)
print(f"Macro Precision: {precision}") #[0.47642436 0.84420705]
print(f"Macro Recall: {recall}") # [0.40104741 0.88043966]
print(f"Macro F1-score: {f1_score}") # [0.43549835 0.86194276]

# Disaply the confusion matrix
print("RF Confusion Matrix")
print(CM.from_predictions(Y_test, predict_test_recall_4))

############################################################################################################################

# Data Preparation for Multiclass Models

df = pd.read_csv('df.csv')
df = df.drop(columns = ['class'])

trans = pd.read_csv('product_category_name_translation.csv')
trans_df = df.merge(trans, how = 'left', left_on = 'product_category_name', right_on = 'product_category_name')

# drop the original name
trans_df = trans_df.drop(columns=['product_category_name'])

# rename the english version variable to shorter form for later encoding
trans_df.rename(columns={'product_category_name_english': 'p_category_'}, inplace=True)

'''implement ONE-HOT encoding'''

df_encoded = pd.get_dummies(trans_df, columns=['customer_state', 'p_category_'])

# Convert only dummy columns to integers 0 and 1s
dummy_columns = [col for col in df_encoded if col.startswith('customer_state_') or col.startswith('p_category_')]
df_encoded[dummy_columns] = df_encoded[dummy_columns].astype(int)

'''splitting train-test dataset''' 

y_value = df_encoded['review_score'] # set the y
y_values = np.ravel(y_value)
# drop the y from the dataframe
x_values = df_encoded.drop('review_score', axis=1)
# split data
X_train, X_test, Y_train, Y_test = train_test_split(x_values, y_value, test_size = 0.2, random_state=4567, stratify=y_value)

# how many of each class in training dataset
print(Y_train.value_counts())

'''undersampling and oversampling'''
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

# Define undersampling for majority classes
undersample_strategy = {5: 5000, 4: 5000}
undersampler = RandomUnderSampler(sampling_strategy=undersample_strategy)
X_res, y_res = undersampler.fit_resample(X_train, Y_train)

# Apply oversampling (SMOTE) on the undersampled data
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='not majority', random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_res, y_res)

print(y_balanced.value_counts())

X_train = X_balanced
Y_train = y_balanced

'''RF training and testing'''
RF_algo = RF()
RF_model = RF_algo.fit(X_train, Y_train)

''' Hyperparameter Random searched '''

warnings.filterwarnings("ignore")

RF_tuned_parameters = {
    'n_estimators': randint(50, 500), # Draw from a uniform distribution between 50 and 500
    'max_depth': randint(2, 7),  # Draw from a uniform distribution between 2 and 7
    'min_samples_split': randint(2, 7),  # Draw from a uniform distribution between 2 and 7
    'max_features': ['sqrt', 'log2', None]
}

'''===================================================================================================================='''
# Model 5: 18 Feature Multiclass Model with Macro Recall Scoring Criteria

# create a hyperparameter search function for re-usability
def random_search(algo, hyperparameters, X_train, Y_train):
  # do the search using 5 folds/chunks
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall_macro', n_iter=15, refit=True)

  # pass the data to fit/train
  clf.fit(X_train, Y_train)

  return clf.best_params_

# search and save best_parameters
RF_5_best_params = random_search(RF_algo, RF_tuned_parameters, X_train, Y_train)

# Train the models with random searched parameters 
RF_5_algo = RF(**RF_5_best_params)
RF_5_model = RF_5_algo.fit(X_train, Y_train)

'''implement Recall_Macro random searched RF model on Training data'''
# predict train data using random searched RF model
predict_train_5 = RF_5_model.predict(X_train)

# check the score of random searched model
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_5, average='macro')
print(f"Macro Precision: {precision}") #0.3768
print(f"Macro Recall: {recall}") #0.365
print(f"Macro F1-score: {f1_score}") # 0.3464

precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_5)
print(f"Macro Precision: {precision}") #
print(f"Macro Recall: {recall}") #  [0.50331516 0.36908796 0.1745985 0.14542508 0.63577427]
print(f"Macro F1-score: {f1_score}") # 

'''implement Recall_Macro random searched RF model on Test data'''
# predict based on test data
predict_test_5 = RF_5_model.predict(X_test)

# Calculate precision, recall, and F1-score for Test data
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_5, average='macro')
print(f"Macro Precision: {precision}") #0.2747
print(f"Macro Recall: {recall}") #0.2988
print(f"Macro F1-score: {f1_score}") #0.2794

# Calculate precision, recall, and F1-score for Test data
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_5)
print(f"Macro Precision: {precision}") #
print(f"Macro Recall: {recall}") # [0.50559811 0.07456979 0.14985795 0.13738402 0.62673179]
print(f"Macro F1-score: {f1_score}") # 


# Disaply the confusion matrix
print("RF Confusion Matrix")
print(CM.from_predictions(Y_test, predict_test_5))

'''========================================================================================================================'''

# Model 6: 18 Feature Multiclass Model with Recall Scoring Criteria

# create a hyperparameter search function for re-usability
def random_search_recall(algo, hyperparameters, X_train, Y_train):
  # do the search using 5 folds/chunks
  clf = RandomizedSearchCV(algo, hyperparameters, cv=5, random_state=2015,
                          scoring='recall', n_iter=15, refit=True)

  # pass the data to fit/train
  clf.fit(X_train, Y_train)

  return clf.best_params_

# search and save best_parameters
RF_6_best_params_recall = random_search_recall(RF_algo, RF_tuned_parameters, X_train, Y_train)

# Train the models with random searched parameters 
RF_6_algo_recall = RF(**RF_6_best_params_recall)
RF_6_model_recall = RF_6_algo_recall.fit(X_train, Y_train)

# predict train data using random searched RF model
predict_train_recall_6 = RF_6_model_recall.predict(X_train)

'''implement Recall random searched RF on Training data'''
# check the score of random searched model, with avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_recall_6, average='macro')
print(f"Macro Precision: {precision}") #0.683
print(f"Macro Recall: {recall}") #0.6286 
print(f"Macro F1-score: {f1_score}") # 0.5987

# check the score of random searched model, without avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict_train_recall_6)
print(f"Macro Precision: {precision}") # [0.78310329 0.58321816]
print(f"Macro Recall: {recall}") # [0.35579905 0.90145407]
print(f"Macro F1-score: {f1_score}") # [0.48929113 0.70822956]


'''implement Recall random searched RF model on Test Data'''
# predict based on test data
predict_test_recall_6 = RF_6_model_recall.predict(X_test)

# Calculate score for Test data with avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_recall_6, average='macro')
print(f"Macro Precision: {precision}") #0.679
print(f"Macro Recall: {recall}") #0.640
print(f"Macro F1-score: {f1_score}") #0.654

# Calculate score for Test Data without avg
precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict_test_recall_6)
print(f"Macro Precision: {precision}") # [0.51561912 0.84259324]
print(f"Macro Recall: {recall}") # [0.37761852 0.90376851]
print(f"Macro F1-score: {f1_score}") # [0.43595863 0.87210938]


# Disaply the confusion matrix
print("RF Confusion Matrix")
print(CM.from_predictions(Y_test, predict_test_recall_6))













