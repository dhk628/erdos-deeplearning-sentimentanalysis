from data import get_data, get_aug_data, add_aug_data
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
import time

np.random.seed(123)
random.seed(123)

df_train = pd.read_parquet('../data/sst5/sst-5_train.parquet')
df_train, df_val = train_test_split(df_train, test_size=0.15, random_state=123, shuffle=True, stratify=df_train['truth'])
X_train, X_val, X_outer_val, X_test, y_train, y_val, y_outer_val, y_test \
    = get_data(sst5='original',
               costco='none')

texts_train = df_train['text'].tolist()

aug = naw.SynonymAug(
    aug_src='wordnet',
    aug_p=0.1,
)

model = SentenceTransformer('thenlper/gte-large', device='cuda')

vectors_train = torch.from_numpy(X_train).to('cuda')

mean_l, std_l, min_l, max_l, dup_l = [], [], [], [], []

start_time = time.time()
for n in range(10):
    print('Run ' + str(n + 1))
    texts_aug = aug.augment(texts_train)
    vectors_aug = model.encode(texts_aug, convert_to_numpy=False, convert_to_tensor=True, device='cuda')

    sim = []
    for i in range(len(texts_train)):
        sim.append(cos_sim(vectors_aug[i], vectors_train[i]).item())

    sim = np.array(sim)

    mean_l.append(np.mean(sim))
    std_l.append(np.std(sim))
    min_l.append(np.min(sim))
    max_l.append(np.max(sim))

    vectors_aug_cpu = vectors_aug.cpu().numpy()
    vectors_train_cpu = vectors_train.cpu().numpy()

    dup = 0
    for i in range(vectors_train_cpu.shape[0]):
        if sum(vectors_train_cpu[i, :] == vectors_aug_cpu[i, :]) == vectors_train_cpu.shape[1]:
            dup += 1

    dup_l.append(dup)

print('Mean: %f' % np.mean(mean_l))
print('Std: %f' % np.mean(std_l))
print('Min: %f' % np.mean(min_l))
print('Max: %f' % np.mean(max_l))
print('Duplicated: %f' % np.mean(dup_l))
print("--- %s seconds ---" % (time.time() - start_time))
