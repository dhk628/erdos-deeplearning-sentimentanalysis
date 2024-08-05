from data import get_data, get_aug_data, add_aug_data
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, root_mean_squared_error, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

df_train = pd.read_parquet('data/sst5/sst-5_train.parquet')
df_train, df_val = train_test_split(df_train, test_size=0.15, random_state=123, shuffle=True, stratify=df_train['truth'])

X_train, X_val, X_outer_val, X_test, y_train, y_val, y_outer_val, y_test \
    = get_data(sst5='original',
               costco='none',
               inner_split=False,
               path='data/sst5/fine_tuned/0730/'  # None or 'data/sst5/fine_tuned/0730/'
               )

# texts_train = df_train['text'].tolist()
# texts_val = df_val['text'].tolist()
#
# sentences = ['This is a very positive review',
#              'This is a very good movie',
#              'This is a positive review',
#              'This is a good movie',
#              'This is a neutral review',
#              'This is an average movie',
#              'This is a negative review',
#              'This is a bad movie',
#              'This is a very negative review',
#              'This is a very bad movie']
#
# model = SentenceTransformer('thenlper/gte-large', device='cuda')
# embeddings = model.encode(sentences, convert_to_numpy=False, convert_to_tensor=True, device='cuda')
#
# X_train_sim = cos_sim(torch.from_numpy(X_train).to('cuda'), embeddings).to('cpu').numpy()
# X_val_sim = cos_sim(torch.from_numpy(X_val).to('cuda'), embeddings).to('cpu').numpy()
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train_sim)
# X_val = scaler.transform(X_val_sim)
#
# logreg = LogisticRegression(multi_class='ovr', n_jobs=-1)
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_val)
#
# acc = accuracy_score(y_val, y_pred)
# rmse = root_mean_squared_error(y_val, y_pred)
# cm = confusion_matrix(y_val, y_pred, normalize='true')
#
# print('Accuracy: %f' % acc)
# print('RMSE: %f' % rmse)
#
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array([1, 2, 3, 4, 5]))
# disp.plot()
# plt.show()