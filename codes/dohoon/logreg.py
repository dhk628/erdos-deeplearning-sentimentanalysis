from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from neural_network import load_data, RANDOM_SEED
import joblib
from evaluation import evaluate_model, save_evaluation

np.random.seed(RANDOM_SEED)
CLASS_NAMES = np.array([1, 2, 3, 4, 5])

X_train, X_val, X_test, y_train, y_val, y_test = load_data('sst5')
y_test = y_test.reshape(-1)

X_train = np.concatenate((X_train, X_val))
y_train = np.concatenate((y_train, y_val))

logreg = LogisticRegression(multi_class='ovr', solver='liblinear')
fit_model = logreg.fit(X_train, y_train.reshape(-1))
joblib.dump(fit_model, 'models/logreg.pkl')

y_pred = fit_model.predict(X_test)
y_prob = fit_model.predict_proba(X_test)

results, cm = evaluate_model(y_test, y_pred, y_prob)
save_evaluation('sst5', 'logreg', results, cm)
