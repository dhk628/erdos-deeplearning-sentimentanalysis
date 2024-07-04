from initialization import set_seed, use_gpu
from data import RatingDataset, get_data, load_data, load_data_from_ray
from neural_network import train_func, eval_func, test_model, get_argmax, get_model_structure, save_model_structure, \
    EarlyStopper
from evaluation import evaluate_model, save_evaluation
import models

import os
import time
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
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import ray
from ray.train import Checkpoint
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif


def short_dirname(trial):
    return "trial_" + str(trial.trial_id)


def train_model(config, print_interval=None):
    start = 1
    device = use_gpu()
    g = set_seed(123, 'cpu')
    train_loader = DataLoader(ray.get(train_id), batch_size=config['batch_size'], shuffle=True, generator=g)
    val_loader = DataLoader(ray.get(val_id), batch_size=1024, shuffle=True, generator=g)

    model = models.FeedForwardNet(
        input_size=128,
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

    if ray_tune:
        checkpoint = train.get_checkpoint()
        if checkpoint:
            print('Checkpoint exists!')
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
                start = checkpoint_dict["epoch"] + 1
                model.load_state_dict(checkpoint_dict["model_state"])

    for epoch in range(start, config['max_num_epochs'] + 1):
        train_loss, train_acc = train_func(train_loader, model, optimizer, loss_fn, get_pred, device)
        val_loss, val_acc = eval_func(val_loader, model, loss_fn, get_pred, device)

        if not ray_tune:
            if (print_interval and (epoch % print_interval == 0 or epoch == 1)) or epoch == config['max_num_epochs']:
                print("Epoch: %d, train_loss: %f, val_loss: %f, train_acc: %f, val_acc: %f"
                      % (epoch, float(train_loss), float(val_loss), float(train_acc), float(val_acc)))
        else:
            if epoch % 10 == 0 or epoch == 1 or epoch == config['max_num_epochs']:
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(
                        {"epoch": epoch, "model_state": model.state_dict()},
                        os.path.join(tempdir, "checkpoint.pt"),
                    )
                    train.report(
                        {'mean_accuracy': val_acc, 'mean_loss': val_loss},
                        checkpoint=Checkpoint.from_directory(tempdir)
                    )


RAY_RESULTS_PATH = "D:/GitHub/Data Science/erdos-deeplearning-companydiscourse/codes/dohoon/.ray_results"
# RAY_RESULTS_PATH = "/export/dohoonk/.ray_results"

if __name__ == '__main__':
    X_train, X_val, X_outer_val, X_test, y_train, y_val, y_outer_val, y_test \
        = get_data(sst5='original',
                   costco='none')

    feature_selector = SelectKBest(f_classif, k=128)
    X_train = feature_selector.fit_transform(X_train, y_train)
    X_val = feature_selector.transform(X_val)

    in_features = X_train.shape[0]
    ohe = 'none'

    data_train = RatingDataset(X_train, y_train, ohe=ohe)
    data_val = RatingDataset(X_val, y_val, ohe=ohe)
    train_id = ray.put(data_train)
    val_id = ray.put(data_val)

    search_space = {'lr': 0.001,
                    'beta1': 0.8,
                    'beta2': 0.999,
                    'batch_size': 32,
                    'n_neurons1': 66,
                    # 'n_neurons2': tune.lograndint(1s, 2048),
                    'dropout_p1': 0,
                    # 'dropout_p2': 0,
                    'weight_decay': 0,
                    'max_num_epochs': 200,
                    }

    ray_tune = False
    resume = False
    search_name = 'sst5_costco_under-128best-lr_beta'

    if ray_tune:
        # Run with Ray tune
        scheduler = ASHAScheduler(
            max_t=search_space['max_num_epochs'],
            grace_period=2,
            reduction_factor=2
        )

        optuna_search = OptunaSearch(
            metric="mean_accuracy",
            mode="max"
        )

        if not resume:
            tuner = tune.Tuner(
                tune.with_resources(
                    train_model,
                    resources={'cpu': 8, 'gpu': 0.5}),
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    metric='mean_accuracy',
                    mode='max',
                    # scheduler=scheduler,
                    # search_alg=optuna_search,
                    # num_samples=200,
                    trial_dirname_creator=short_dirname
                ),
                run_config=train.RunConfig(
                    name=search_name + '-' + datetime.now().strftime('%Y%m%d_%H%M%S'),
                    storage_path=RAY_RESULTS_PATH
                )
            )
        else:
            # If tuning was stopped
            tuner = tune.Tuner.restore(
                path=RAY_RESULTS_PATH + '/' + 'sst5_costco_under-n1-20240701_230127',
                trainable=tune.with_resources(
                    train_model,
                    resources={'cpu': 8, 'gpu': 0.5}
                ),
                resume_unfinished=True,
                restart_errored=True
            )

        results = tuner.fit()

        best_result = results.get_best_result("mean_accuracy", mode="max")
        best_config = best_result.config
        best_acc = best_result.metrics['mean_accuracy']
        best_result_epochs = best_result.metrics['training_iteration']

        print('Best config: ' + str(best_config))
        print('Best accuracy: ' + str(best_acc))
        print('Epochs: ' + str(best_result_epochs))
    else:
        # Run without Ray tune
        start_time = time.time()
        train_model(search_space, print_interval=10)
        print("--- %s seconds ---" % (time.time() - start_time))
