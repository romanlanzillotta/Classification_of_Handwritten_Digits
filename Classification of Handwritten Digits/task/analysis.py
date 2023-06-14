# write your code here
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.datasets.mnist import load_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

def fit_predict_eval(model_name, model_class, features_train, features_test, target_train, target_test):
    # here you fit the model
    model_class = model_class.fit(features_train, target_train)
    # make a prediction
    preds = model_class.predict(features_test)
    # calculate accuracy and save it to score
    cm = confusion_matrix(target_test, preds)
    # accuracy
    score = np.diag(cm).sum() / cm.sum()
    print(f'Model: {model_name}\nAccuracy: {score.round(4)}\n')
    return model_name, round(score, 3)


(x_data, y_data) = load_data(path="mnist.npz")[0]
x_data = x_data[:6000]
y_data = y_data[:6000]
x_data = np.reshape(x_data, (6000, -1))
# print(x_data.shape)
# print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=40)
# print("x_train shape:", x_train.shape)
# print("x_test shape:", x_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)
# print("Proportion of samples per class in train set:")
# print(pd.Series.value_counts(y_train, normalize=True))

# Normalization
transformer = Normalizer()
x_train_norm = transformer.fit_transform(x_train)
transformer = Normalizer()
x_test_norm = transformer.fit_transform(x_test)

# Stage 4/4
# models = [(KNeighborsClassifier, {}),
#           (DecisionTreeClassifier, {'random_state': 40}),
#           (LogisticRegression, {'random_state': 40}),
#           (RandomForestClassifier, {'random_state': 40})]

# results = []
# for itm in models:
#     results.append(
#         fit_predict_eval(
#             model_name=str(itm).split("(")[0],
#             model_class=itm,
#             features_train=x_train_norm,
#             features_test=x_test_norm,
#             target_train=y_train,
#             target_test=y_test
#         )
#     )
#
# df = pd.DataFrame(results, columns=["Model", "Accuracy"])
# df.sort_values(by='Accuracy', ascending=False, inplace=True, ignore_index=True)

# Stage 5/5
# param grid
param_grid_KNN = {'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'brute']}
param_grid_RF = {'n_estimators': [300, 500], 'max_features': ['auto', 'log2'],
                 'class_weight': ['balanced', 'balanced_subsample']}

# estimators search
clf_KNN = GridSearchCV(estimator=KNeighborsClassifier(),
             param_grid=param_grid_KNN, scoring='accuracy', n_jobs=-1).fit(x_train_norm, y_train)

clf_RF = GridSearchCV(estimator=RandomForestClassifier(random_state=40),
             param_grid=param_grid_RF, scoring='accuracy', n_jobs=-1).fit(x_train_norm, y_train)

# print(clf_KNN.best_estimator_)
# print(clf_KNN.best_params_)
# best estimator KNN: KNeighborsClassifier(n_neighbors=4, weights='distance')
# best params KNN: {'algorithm': 'auto', 'n_neighbors': 4, 'weights': 'distance'}

# RandomForestClassifier(class_weight='balanced_subsample', max_features='auto',
#                       n_estimators=500, random_state=40)
# {'class_weight': 'balanced_subsample', 'max_features': 'auto', 'n_estimators': 500}

# Note: best_estimator_ is the call to the model initialization, including all parameters passed (from search and user)
#       best_params_ are the best parameters found by the grid search. Not the same as "all" parameters passed, example:
#       the used could have passed a random_state seed and this is not included in best_params.

print("K-nearest neighbours algorithm")
print("best estimator:", clf_KNN.best_estimator_)
print("accuracy:", round(clf_KNN.best_estimator_.score(x_test_norm, y_test), 3))
print()
print("Random forest algorithm")
print("best estimator:", clf_RF.best_estimator_)
print("accuracy:", round(clf_RF.best_estimator_.score(x_test_norm, y_test), 3))

