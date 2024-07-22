from data import get_data
import numpy as np
import pandas as pd
import torch
import random
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import nlpaug.model.word_embs as nmw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import os
import winsound

df_train = pd.read_parquet('data/sst5/sst-5_inner-train.parquet')
df_val = pd.read_parquet('data/sst5/sst-5_inner-val.parquet')

texts_train = df_train['text'].tolist()
ratings_train = df_train['rating'].tolist()

aug = naw.SynonymAug(
    aug_src='wordnet',
    aug_max=1,
)

model = SentenceTransformer('thenlper/gte-large', device='cuda')

vectors_aug_list = []
texts_aug_list = []
num_iter = 10

for n in range(num_iter):
    print('Run ' + str(n + 1))
    texts_aug = aug.augment(texts_train)
    texts_aug_list = texts_aug_list + texts_aug
    vectors_aug = model.encode(texts_aug, convert_to_numpy=True, device='cuda')
    vectors_aug_list.append(vectors_aug)

all_vectors = np.vstack(vectors_aug_list).tolist()
all_ratings = ratings_train * num_iter

df_aug = pd.DataFrame({'vector': all_vectors, 'rating': all_ratings, 'text': texts_aug_list})
n = len([name for name in os.listdir('data/sst5/aug')])
# df_aug.to_parquet('data/sst5/aug/' + 'sst-5_aug_' + str(n) + '.parquet', index=False)
# df_aug.to_parquet('data/sst5/aug/aug.parquet', index=False)

X_train = np.vstack(df_train['vector'].tolist()).astype(np.float32)
y_train = np.vstack(df_train['rating'].tolist()).reshape(-1)
X_val = np.vstack(df_val['vector'].tolist()).astype(np.float32)
y_val = np.vstack(df_val['rating'].tolist()).reshape(-1)

X_aug = np.vstack(df_aug['vector'].tolist()).astype(np.float32)
y_aug = np.vstack(df_aug['rating'].tolist()).reshape(-1)

X_train = np.vstack((X_train, X_aug))
y_train = np.hstack((y_train, y_aug))

logreg = LogisticRegression(multi_class='ovr', n_jobs=-1)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_val)

acc = accuracy_score(y_val, y_pred)
print('Accuracy: %f' % acc)

# winsound.MessageBeep()
# Baseline: 0.5678627145085804
