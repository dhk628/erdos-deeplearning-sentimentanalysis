from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from data import get_data
import joblib
from evaluation import evaluate_model, save_evaluation
from sklearn.feature_selection import SelectKBest, f_classif

RANDOM_SEED = 123

np.random.seed(RANDOM_SEED)
CLASS_NAMES = np.array([1, 2, 3, 4, 5])

X_train, X_val, X_outer_val, X_test, y_train, y_val, y_outer_val, y_test \
    = get_data(sst5='original',
               costco='none')

X_train = np.vstack([X_train, X_val])
y_train = np.hstack([y_train, y_val])

feature_selector = SelectKBest(f_classif, k=100)
X_train = feature_selector.fit_transform(X_train, y_train)
X_outer_val = feature_selector.transform(X_outer_val)

logreg = LogisticRegression(multi_class='ovr', solver='liblinear')
fit_model = logreg.fit(X_train, y_train)
# joblib.dump(fit_model, 'models/logreg.pkl')

y_pred = fit_model.predict(X_outer_val)
y_prob = fit_model.predict_proba(X_outer_val)

results, cm = evaluate_model(y_outer_val, y_pred, y_prob)
# save_evaluation('sst5', 'logreg', results, cm)
print(results['accuracy'])
