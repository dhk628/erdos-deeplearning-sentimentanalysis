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
import nlpaug.augmenter.word as naw

df_train = pd.read_parquet('data/sst5/sst-5_inner-train.parquet')
texts_train = df_train['text'].tolist()
ratings_train = df_train['rating'].tolist()

aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute",
    device='cuda',
    aug_max=1,
    aug_min=1
)

model = SentenceTransformer('thenlper/gte-large', device='cuda')

vectors_aug_list = []

for n in range(10):
    print('Run ' + str(n + 1))
    texts_aug = aug.augment(texts_train)
    vectors_aug = model.encode(texts_aug, convert_to_numpy=True, device='cuda')
    vectors_aug_list.append(vectors_aug)
