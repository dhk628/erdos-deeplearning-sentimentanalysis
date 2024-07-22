from data import get_data, get_aug_data, add_aug_data
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

df_train = pd.read_parquet('data/sst5/sst-5_train.parquet')
df_train, df_val = train_test_split(df_train, test_size=0.15, random_state=123, shuffle=True, stratify=df_train['truth'])
X_train, X_val, X_outer_val, X_test, y_train, y_val, y_outer_val, y_test \
    = get_data(sst5='original',
               costco='none')

X_train, y_train = add_aug_data(X_train, y_train, 'data/sst5/aug/sst-5_aug_4.parquet')

# texts_train = df_train['text'].tolist()
# texts_val = df_val['text'].tolist()
#
# sentences = ['very positive',
#              'very good',
#              'positive',
#              'good',
#              'neutral',
#              'average',
#              'negative',
#              'bad',
#              'very negative',
#              'very bad']
#
# model = SentenceTransformer('thenlper/gte-large', device='cuda')
# vectors_train = model.encode(texts_train, convert_to_numpy=False, convert_to_tensor=True, device='cuda')
# vectors_val = model.encode(texts_val, convert_to_numpy=False, convert_to_tensor=True, device='cuda')
# embeddings = model.encode(sentences, convert_to_numpy=False, convert_to_tensor=True, device='cuda')
#
# # X_train_sim = cos_sim(torch.from_numpy(X_train).to('cuda'), embeddings).to('cpu').numpy()
# # X_val_sim = cos_sim(torch.from_numpy(X_val).to('cuda'), embeddings).to('cpu').numpy()
#
# X_train_sim = cos_sim(vectors_train, embeddings).to('cpu').numpy()
# X_val_sim = cos_sim(vectors_val, embeddings).to('cpu').numpy()
#
# scaler = MinMaxScaler()
# X_train_sim = scaler.fit_transform(X_train_sim)
# X_val_sim = scaler.transform(X_val_sim)

logreg = LogisticRegression(multi_class='ovr', n_jobs=-1)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_val)

acc = accuracy_score(y_val, y_pred)

print(acc)
