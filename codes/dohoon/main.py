from initialization import set_seed, use_gpu, set_ray_settings, short_dirname
from data import RatingDataset, get_data, load_data, load_data_from_ray
from neural_network import train_func, eval_func, test_model, get_argmax, get_model_structure, save_model_structure, \
    EarlyStopper
from evaluation import evaluate_model, save_evaluation
import models

import os
import time
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label, corn_label_from_logits
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, CoralLoss, CornLoss
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch
import ray
from ray.train import Checkpoint
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif
from ray.tune.stopper import TrialPlateauStopper


RAY_RESULTS_PATH, RAY_RESOURCES = set_ray_settings('pc')


def train_model(config, print_interval=None, early_stop_patience=0, early_stop_min_delta=0.0):
    start = 1
    device = use_gpu()
    g = set_seed(123, 'cpu')
    train_loader = DataLoader(ray.get(train_id), batch_size=config['batch_size'], shuffle=True, generator=g)
    val_loader = DataLoader(ray.get(val_id), batch_size=1024, shuffle=True, generator=g)
    early_stopper = EarlyStopper(patience=early_stop_patience, min_delta=early_stop_min_delta)

    model = models.FeedForwardNet(
        input_size=in_features,
        output_size=out_features,
        hidden_layers=[config['n_neurons1']],
        dropout_p=[config.get('dropout_p1', 0)]
    )
    model = model.to(device)

    if ohe == 'coral':
        loss_fn = CoralLoss()  # CoralLoss() or nn.MSELoss()
    elif ohe == 'corn':
        loss_fn = CornLoss(5)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=(1 - config['alpha1'], 1 - config['alpha2']),
        weight_decay=config.get('weight_decay', 0)
    )

    if ohe == 'none':
        get_pred = get_argmax
    elif ohe == 'coral':
        get_pred = proba_to_label
    elif ohe == 'corn':
        get_pred = corn_label_from_logits
    else:
        get_pred = get_argmax

    if ray_tune:
        checkpoint = train.get_checkpoint()
        if checkpoint:
            print('Checkpoint exists!')
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))

            start = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
            for param_group in optimizer.param_groups:
                if "lr" in config:
                    param_group["lr"] = config["lr"]
                if "alpha1" in config:
                    param_group["alpha1"] = config["alpha1"]
                if "alpha2" in config:
                    param_group["alpha2"] = config["alpha2"]
    else:
        acc_list = []

    for epoch in range(start, config['max_num_epochs'] + 1):
        train_loss, train_acc = train_func(train_loader, model, optimizer, loss_fn, get_pred, device)
        val_loss, val_acc = eval_func(val_loader, model, loss_fn, get_pred, device)

        if not ray_tune:
            acc_list.append(val_acc)
            if early_stop_patience > 0:
                if early_stopper.early_stop(val_loss):
                    break
            if (print_interval and (epoch % print_interval == 0 or epoch == 1)) or epoch == config['max_num_epochs']:
                print("Epoch: %d, train_loss: %f, val_loss: %f, train_acc: %f, val_acc: %f"
                      % (epoch, float(train_loss), float(val_loss), float(train_acc), float(val_acc)))
        else:
            metrics = {'mean_accuracy': val_acc, 'mean_loss': val_loss}
            if epoch % config['checkpoint_interval'] == 0:
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                        },
                        os.path.join(tempdir, "checkpoint.pt"),
                    )
                    train.report(
                        metrics,
                        checkpoint=Checkpoint.from_directory(tempdir)
                    )
            else:
                train.report(metrics)

    if not ray_tune:
        max_acc = max(acc_list)
        print("Final epoch: %d, val_acc: %f"
              % (epoch, float(val_acc)))
        print("Best epoch: %d, val_acc: %f \n"
              % (np.argmax(acc_list) + 1, float(max_acc)))


if __name__ == '__main__':
    X_train, X_val, X_outer_val, X_test, y_train, y_val, y_outer_val, y_test \
        = get_data(sst5='original',
                   costco='none')

    feature_selector = SelectKBest(f_classif, k=128)
    X_train = feature_selector.fit_transform(X_train, y_train)
    X_val = feature_selector.transform(X_val)

    ohe = 'none'
    in_features = X_train.shape[1]
    if ohe == ('coral' or 'corn'):
        out_features = 4
    else:
        out_features = 5

    data_train = RatingDataset(X_train, y_train, ohe=ohe)
    data_val = RatingDataset(X_val, y_val, ohe=ohe)
    train_id = ray.put(data_train)
    val_id = ray.put(data_val)

    search_space = {'lr': 0.0008008715387630997,  # tune.loguniform(1e-4, 1e-1), tune.grid_search([0.1, 0.01, 0.001, 0.0001])
                    'alpha1': 0.001251181030793111,  # tune.loguniform(1e-3, 0.2), tune.grid_search([0.2, 0.1, 0.01, 0.001])
                    'alpha2': 0.0003545419723740253,  # tune.loguniform(1e-5, 1e-2), tune.grid_search([0.1, 0.01, 0.001, 0.0001])
                    'batch_size': 32,
                    'n_neurons1': 66,
                    'max_num_epochs': 200,
                    'min_num_epochs': 40,
                    'checkpoint_interval': 10,
                    }

    ray_tune = True
    resume = False
    search_name = 'sst5_128'

    if ray_tune:
        # Run with Ray tune
        scheduler_asha = ASHAScheduler(
            max_t=search_space['max_num_epochs'],
            grace_period=search_space['min_num_epochs'],
            reduction_factor=2
        )

        scheduler_pbt = PopulationBasedTraining(
            time_attr='training_iteration',
            perturbation_interval=search_space['checkpoint_interval'],
            burn_in_period=search_space['checkpoint_interval'],
            hyperparam_mutations={
                'lr': tune.loguniform(1e-4, 1e-1),
                'alpha1': tune.loguniform(1e-3, 0.2),
                'alpha2': tune.loguniform(1e-5, 1e-2),
            }
        )

        optuna_search = OptunaSearch(
            metric="mean_accuracy",
            mode="max",
            seed=123,
            points_to_evaluate=[{'lr': 0.0011667300285897403, 'alpha1': 0.0010889159512101896, 'alpha2': 7.094804335419697e-05}]
        )

        trial_stopper = TrialPlateauStopper(
            metric='mean_accuracy',
            std=0.0025,
            num_results=4,
            grace_period=search_space['min_num_epochs'],
            metric_threshold=0.55,
            mode='max'
        )

        if not resume:
            tuner = tune.Tuner(
                tune.with_resources(
                    train_model,
                    resources=RAY_RESOURCES
                ),
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    metric='mean_accuracy',
                    mode='max',
                    scheduler=scheduler_asha,
                    search_alg=optuna_search,
                    num_samples=200,
                    trial_dirname_creator=short_dirname,
                ),
                run_config=train.RunConfig(
                    name=search_name + '-' + datetime.now().strftime('%Y%m%d_%H%M%S'),
                    storage_path=RAY_RESULTS_PATH,
                    stop=trial_stopper,
                )
            )
        else:
            # If tuning was stopped
            tuner = tune.Tuner.restore(
                path=RAY_RESULTS_PATH + '/' + 'sst5_costco_under-n1-20240701_230127',
                trainable=tune.with_resources(
                    train_model,
                    resources=RAY_RESOURCES
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
        train_model(
            search_space,
            print_interval=1,
            # early_stop_patience=2,
            # early_stop_min_delta=0.001
        )
        print("--- %s seconds ---" % (time.time() - start_time))
