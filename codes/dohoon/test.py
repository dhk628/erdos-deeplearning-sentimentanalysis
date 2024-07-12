from initialization import set_seed, use_gpu
from data import RatingDataset, load_data
from neural_network import train_func, eval_func, test_model, get_argmax, get_model_structure, save_model_structure, \
    EarlyStopper
from evaluation import evaluate_model, save_evaluation
import models

import os
import tempfile
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss
from sklearn.metrics import confusion_matrix, mean_squared_error, root_mean_squared_error, mean_absolute_error, \
    log_loss, accuracy_score, ConfusionMatrixDisplay, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from datetime import datetime


def final_train_model(config):
    start = 1
    device = use_gpu()
    g = set_seed(123, 'cpu')
    train_loader = DataLoader(data_train, batch_size=config['batch_size'], shuffle=True, generator=g)
    val_loader = DataLoader(data_val, batch_size=1024, shuffle=True, generator=g)

    model = models.FeedForwardNet(
        input_size=1024,
        output_size=5,
        hidden_layers=[config['n_neurons1']],
        dropout_p=[config['dropout_p1']]
    )
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=(config['beta1'], config['beta2']),
        weight_decay=config['weight_decay']
    )

    get_pred = get_argmax

    for epoch in range(start, config['max_num_epochs'] + 1):
        train_loss, train_acc = train_func(train_loader, model, optimizer, loss_fn, get_pred, device)
        val_loss, val_acc = eval_func(val_loader, model, loss_fn, get_pred, device)

        if epoch == config['max_num_epochs']:
            print("Epoch: %d, train_loss: %f, val_loss: %f, train_acc: %f, val_acc: %f"
                  % (epoch, float(train_loss), float(val_loss), float(train_acc), float(val_acc)))


if __name__ == '__main__':
    X_train, X_val, X_outer_val, X_test, y_train, y_val, y_outer_val, y_test = load_data(sst5='original',
                                                                                         costco=False)

    X_train = np.vstack([X_train, X_val])
    y_train = np.hstack([y_train, y_val])

    ohe = 'none'
    ray_tune = False

    data_train = RatingDataset(X_train, y_train, ohe=ohe)
    data_val = RatingDataset(X_val, y_val, ohe=ohe)
    data_test = RatingDataset(X_test, y_test, ohe=ohe)

    search_space = {'lr': 0.01,
                    'beta1': 0.9,
                    'beta2': 0.99,
                    'batch_size': 64,
                    'n_neurons1': 600,
                    # 'n_neurons2': tune.lograndint(1s, 2048),
                    'dropout_p1': 0.27087908376414493,
                    # 'dropout_p2': 0,
                    'weight_decay': 5.361919401182879e-06,
                    'max_num_epochs': 10
                    }

    final_train_model(search_space)
