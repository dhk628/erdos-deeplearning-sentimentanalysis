from initialization import set_seed, use_gpu
from data import RatingDataset, load_data
from neural_network import train_fnn, test_model, get_argmax, get_model_structure, save_model_structure, EarlyStopper
from evaluation import evaluate_model, save_evaluation
import models

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
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

if __name__ == '__main__':
    set_seed(123)
    # use_gpu()

    # Load data as np arrays
    X_outer_train, X_outer_val, X_test, y_outer_train, y_outer_val, y_test = load_data(sst5='original')
    # X_costco_train, y_costco_train = load_data('costco')
    #
    # # Combine data
    # X_train = np.vstack((X_train, X_costco_train))
    # y_train = np.hstack((y_train, y_costco_train))

    # # Preprocessing before changing to tensors
    # X_train, y_train = RandomOverSampler(random_state=RANDOM_SEED).fit_resample(X_train, y_train)

    # Inner split
    X_train, X_val, y_train, y_val = train_test_split(X_outer_train, y_outer_train, test_size=0.15, random_state=123,
                                                      shuffle=True, stratify=y_outer_train)
    _, class_counts = np.unique(y_train, return_counts=True)
    class_ratios = (class_counts / sum(class_counts)).tolist()
    class_weights = (sum(class_counts) / class_counts).tolist()
    # If ohe is none, i.e. we use the usual classification with integer class labels, we should use:
    # loss = CrossEntropyLoss or similar
    # get_prob = softmax
    # get_pred = get_argmax from get_prob

    # If ohe is ordinal, i.e. we use ordinal one hot encoding, we should use:
    # loss = CrossEntropyLoss or MSE or similar
    # get_prob does not exist?
    # get_pred = proba_to_label

    ohe = 'none'
    # loss = nn.CrossEntropyLoss()
    get_pred = get_argmax
    get_prob = torch.nn.Softmax(dim=1)

    # Load data using Pytorch
    data_train = RatingDataset(X_train, y_train, ohe=ohe)
    data_val = RatingDataset(X_val, y_val, ohe=ohe)
    data_outer_val = RatingDataset(X_outer_val, y_outer_val, ohe=ohe)
    data_test = RatingDataset(X_test, y_test)

    if ohe == 'ordinal':
        n_out = 4
    else:
        n_out = 5

    # Model definition
    hyper_config = {'batch_size': [64],
                    'null2': [0]}
    max_accuracy = 0.0
    max_index = [-1, -1]
    for hyper1 in hyper_config[list(hyper_config.keys())[0]]:
        for hyper2 in hyper_config[list(hyper_config.keys())[1]]:
            # model = models.FeedForwardNetOne(n_neurons=600, dropout_rate=0.27087908376414493)
            model = models.FeedForwardNet(1024, 5, [600], [0.27087908376414493])
            model = model.to('cuda:0')
            print('Training with ' + str(list(hyper_config.keys())[0]) + '=' + str(hyper1) + ', ' + str(list(hyper_config.keys())[1]) + '=' + str(hyper2))

            training_dict = {'train_data': 'sst5',
                             'ohe': ohe,
                             'batch_size': hyper1,
                             'loss_fn': nn.CrossEntropyLoss(),
                             'optimizer': optim.Adam(model.parameters(), lr=1e-2, fused=True, weight_decay=5.361919401182879e-06),
                             'n_epochs': 20,
                             'get_pred': get_pred}

            val_accuracy = train_fnn(data_train, data_val,
                                     model=model,
                                     batch_size=training_dict['batch_size'],
                                     loss_fn=training_dict['loss_fn'],
                                     optimizer=training_dict['optimizer'],
                                     n_epochs=training_dict['n_epochs'],
                                     get_pred=training_dict['get_pred'],
                                     data_loader_generator=set_seed(123),
                                     early_stop_patience=0,
                                     early_stop_min_delta=0)

            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                max_index = [hyper1, hyper2]

    print('Max accuracy: %f' % max_accuracy)
    print('Hyperparamaters: ' + str(max_index))




    # true, pred, prob = test_model(model, data_outer_val, get_pred, get_prob)
    # accuracy = accuracy_score(true.cpu().numpy(), pred.cpu().numpy())
    # print("Accuracy: %f" % (accuracy * 100))

    # model_structure = get_model_structure(model, training_dict)
    # model_structure['accuracy'] = accuracy
    # if accuracy > 0.55:
    #     save_model_structure(model_structure, 'sst5')
    #     print('Saved model.')
