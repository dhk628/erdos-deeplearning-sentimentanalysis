# from initialization import set_seed, use_gpu
# from data import RatingDataset, load_data
# from neural_network import train_fnn, test_model, get_argmax, get_model_structure, save_model_structure, EarlyStopper
# from evaluation import evaluate_model, save_evaluation
# import models
#
# import copy
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import os
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
# from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
# from coral_pytorch.layers import CoralLayer
# from coral_pytorch.losses import coral_loss
# from sklearn.metrics import confusion_matrix, mean_squared_error, root_mean_squared_error, mean_absolute_error, \
#     log_loss, accuracy_score, ConfusionMatrixDisplay, roc_auc_score, f1_score, matthews_corrcoef
# from sklearn.model_selection import train_test_split
# from ray import train, tune
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune.search.optuna import OptunaSearch
# from torch.utils.data import WeightedRandomSampler
#
#
# def train_func(train_loader, model, optimizer, loss_fn):
#     model.train()
#     for batch_idx, (vectors, ratings) in enumerate(train_loader):
#         vectors = vectors.to('cuda:0')
#         ratings = ratings.to('cuda:0')
#         batch_size = vectors.shape[0]
#         outputs = model(vectors.view(batch_size, -1))
#         loss = loss_fn(outputs, ratings)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#
# def eval_func(val_loader, model, get_pred):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (vectors, ratings) in enumerate(val_loader):
#             vectors = vectors.to('cuda:0')
#             ratings = ratings.to('cuda:0')
#             outputs = model(vectors)
#             predicted = get_pred(outputs)
#             total += ratings.shape[0]
#             correct += (predicted == ratings).sum().item()
#
#     return correct / total
#
#
# def train_model(config):
#     random.seed(123)
#     np.random.seed(123)
#     torch.manual_seed(123)
#     torch.cuda.manual_seed(123)
#     torch.cuda.manual_seed_all(123)
#     g = torch.Generator(device='cpu')
#     g.manual_seed(123)
#     train_loader = DataLoader(data_train, batch_size=64, shuffle=True, generator=g)
#     val_loader = DataLoader(data_val, batch_size=256, shuffle=False)
#
#     model = models.FeedForwardNetOne(n_neurons=config['n_neurons'], dropout_rate=config['dropout_rate'])
#     model = model.to('cuda:0')
#     optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=config['weight_decay'])
#
#     for i in range(max_num_epochs):
#         train_func(train_loader, model, optimizer, nn.CrossEntropyLoss())
#         acc = eval_func(val_loader, model, get_pred=get_argmax)
#
#         train.report({"mean_accuracy": acc})
#
#
# set_seed(123)
# use_gpu()
#
# # Load data as np arrays
# X_outer_train, X_outer_val, X_test, y_outer_train, y_outer_val, y_test = load_data('sst5')
#
# X_train, X_val, y_train, y_val = train_test_split(X_outer_train, y_outer_train, test_size=0.15, random_state=123,
#                                                   shuffle=True, stratify=y_outer_train)
#
# _, class_counts = np.unique(y_train, return_counts=True)
# class_ratios = (class_counts / sum(class_counts)).tolist()
# class_weights = (sum(class_counts) / class_counts).tolist()
#
# ohe = 'none'
#
# data_train = RatingDataset(X_train, y_train, ohe=ohe)
# data_val = RatingDataset(X_val, y_val, ohe=ohe)
# data_outer_val = RatingDataset(X_outer_val, y_outer_val, ohe=ohe)
# data_test = RatingDataset(X_test, y_test)
#
# max_num_epochs = 100
#
# search_space = {'weight_decay': tune.loguniform(1e-6, 1e-1),
#                 'dropout_rate': tune.uniform(0.0, 0.8),
#                 'n_neurons': tune.lograndint(1, 1000)
#                 }
#
# scheduler = ASHAScheduler(max_t=max_num_epochs,
#                           grace_period=2,
#                           reduction_factor=2)
#
# optuna_search = OptunaSearch(metric="mean_accuracy",
#                              mode="max")
#
# tuner = tune.Tuner(tune.with_resources(train_model,
#                                        resources={'cpu': 8, 'gpu': 0.5}),
#                    param_space=search_space,
#                    tune_config=tune.TuneConfig(metric='mean_accuracy',
#                                                mode='max',
#                                                scheduler=scheduler,
#                                                search_alg=optuna_search,
#                                                num_samples=5000
#                                                ))
#
# results = tuner.fit()
#
# best_result = results.get_best_result("mean_accuracy", mode="max")
# best_config = best_result.config
# best_acc = best_result.metrics['mean_accuracy']
# best_result_epochs = best_result.metrics['training_iteration']
#
# print('Best config: ' + str(best_config))
# print('Best accuracy: ' + str(best_acc))
# print('Epochs: ' + str(best_result_epochs))
