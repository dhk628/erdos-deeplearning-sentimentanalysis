import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from sklearn.model_selection import train_test_split
import ray


class RatingDataset(Dataset):
    def __init__(self, vector, rating, ohe='none'):
        self.vector = torch.tensor(vector, dtype=torch.float32)
        if ohe == 'none':
            self.rating = torch.tensor(rating, dtype=torch.int64)
        elif ohe == 'coral':
            self.rating = levels_from_labelbatch(rating, num_classes=5)
        else:
            self.rating = torch.tensor(rating, dtype=torch.int64)
        self.ratings_original = torch.tensor(rating, dtype=torch.int64)
        self.ohe = ohe

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        vector = self.vector[idx]
        rating = self.rating[idx]
        ratings_original = self.ratings_original[idx]
        return vector, rating, ratings_original


def get_data(sst5='original', costco='none', inner_split=True):
    df_train = pd.read_parquet('data/sst5/sst-5_train.parquet').rename(
        columns={'truth': 'rating', 'vectors': 'vector'})
    df_val = pd.read_parquet('data/sst5/sst-5_validation.parquet').rename(
        columns={'truth': 'rating', 'vectors': 'vector'})
    df_test = pd.read_parquet('data/sst5/sst-5_test.parquet').rename(
        columns={'truth': 'rating', 'vectors': 'vector'})

    # ndarray: (-1, 1024)
    X_train_s = np.vstack(df_train['vector'].tolist())
    X_val_s = np.vstack(df_val['vector'].tolist())
    X_test_s = np.vstack(df_test['vector'].tolist())

    # ndarray: (-1)
    y_train_s = np.vstack(df_train['rating'].tolist()).reshape(-1)
    y_val_s = np.vstack(df_val['rating'].tolist()).reshape(-1)
    y_test_s = np.vstack(df_test['rating'].tolist()).reshape(-1)

    if costco == 'under':
        X_train_c = np.load('data/costco_google_reviews/vector_under.npy')
        y_train_c = np.load('data/costco_google_reviews/rating_under.npy')

    if costco != 'none':
        X_train = np.vstack([X_train_c, X_train_s])
        y_train = np.hstack([y_train_c, y_train_s])
    else:
        X_train = X_train_s
        y_train = y_train_s

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=123,
                                                      shuffle=True, stratify=y_train)

    if np.min(y_train) != 0 and np.min(y_val) != 0 and np.min(y_val_s) != 0 and np.min(y_test_s) != 0:
        y_train -= np.array(1)
        y_val -= np.array(1)
        y_val_s -= np.array(1)
        y_test_s -= np.array(1)

    if inner_split:
        return X_train, X_val, X_val_s, X_test_s, y_train, y_val, y_val_s, y_test_s
    else:
        X_train = np.vstack([X_train, X_val])
        y_train = np.hstack([y_train, y_val])
        return X_train, X_val_s, X_test_s, y_train, y_val_s, y_test_s


def load_data(dataset, batch_size, generator):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def load_data_from_ray(dataset_id, batch_size, generator):
    DataLoader(ray.get(dataset_id), batch_size=batch_size, shuffle=True, generator=generator)

# def load_data_old(dataset, num=None):
#     if dataset == 'sst5':
#         df_train = pd.read_parquet('data/sst5/sst-5_train.parquet').rename(
#             columns={'truth': 'rating', 'vectors': 'vector'})
#         df_val = pd.read_parquet('data/sst5/sst-5_validation.parquet').rename(
#             columns={'truth': 'rating', 'vectors': 'vector'})
#         df_test = pd.read_parquet('data/sst5/sst-5_test.parquet').rename(
#             columns={'truth': 'rating', 'vectors': 'vector'})
#
#         # ndarray: (-1, 1024)
#         X_train = np.vstack(df_train['vector'].tolist())
#         X_val = np.vstack(df_val['vector'].tolist())
#         X_test = np.vstack(df_test['vector'].tolist())
#
#         # ndarray: (-1)
#         y_train = np.vstack(df_train['rating'].tolist()).reshape(-1)
#         y_val = np.vstack(df_val['rating'].tolist()).reshape(-1)
#         y_test = np.vstack(df_test['rating'].tolist()).reshape(-1)
#
#         if np.min(y_train) != 0 and np.min(y_val) != 0 and np.min(y_test) != 0:
#             y_train -= np.array(1)
#             y_val -= np.array(1)
#             y_test -= np.array(1)
#
#         return X_train, X_val, X_test, y_train, y_val, y_test
#
#     if dataset == 'costco':
#         df = pd.read_parquet('data/costco_google_reviews/costco_2021_reviews_filtered_vectorized_final.parquet')
#         X = np.array(df['vector'].tolist())
#         y = np.array(df['rating'].tolist())
#
#         if np.min(y) != 0:
#             y -= np.array(1)
#
#         return X, y
