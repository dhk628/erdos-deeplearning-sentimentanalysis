from initialization import set_seed, use_gpu, set_ray_settings, short_dirname
from data import RatingDataset, get_data, load_data, load_data_from_ray, get_aug_data, add_aug_data
from neural_network import train_func, eval_func, test_model, get_argmax, get_model_structure, save_model_structure, \
    EarlyStopper
from evaluation import evaluate_model, save_evaluation
import models

import os
import copy
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, root_mean_squared_error, ConfusionMatrixDisplay

RAY_RESULTS_PATH, RAY_RESOURCES = set_ray_settings('math_a')


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
        hidden_layers=[config['n_neurons1'], config.get('n_neurons2', 0), config.get('n_neurons3', 0)],
        dropout_p=[config.get('dropout_p1', 0), config.get('dropout_p2', 0)],
        input_dropout=config.get('input_dropout', 0)
    )
    model = model.to(device)

    if ohe == 'coral':
        loss_fn = nn.MSELoss()  # CoralLoss() or nn.MSELoss()
    elif ohe == 'corn':
        loss_fn = CornLoss(5)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('lr', 1e-3),
        betas=(1 - config.get('alpha1', 0.1), 1 - config.get('alpha2', 0.001)),
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

    val_loss_list = []
    val_acc_list = []
    train_loss_list = []
    train_acc_list = []

    checkpt1, checkpt2, checkpt3 = -1, -1, -1
    passed_1, passed_2, passed_3 = False, False, False

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

    for epoch in range(start, config['max_num_epochs'] + 1):
        train_loss, train_acc = train_func(train_loader, model, optimizer, loss_fn, get_pred, device)
        val_loss, val_acc = eval_func(val_loader, model, loss_fn, get_pred, device)

        if not ray_tune:
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            if train_acc >= 0.6 and not passed_1:
                checkpt1 = copy.deepcopy(epoch)
                passed_1 = True
            if train_acc >= 0.65 and not passed_2:
                checkpt2 = copy.deepcopy(epoch)
                passed_2 = True
            if train_acc >= 0.7 and not passed_3:
                checkpt3 = copy.deepcopy(epoch)
                passed_3 = True
                # break
            if early_stop_patience > 0:
                if early_stopper.early_stop(val_loss):
                    break
            if (print_interval and (epoch % print_interval == 0 or epoch == 1)) or epoch == config['max_num_epochs']:
                print("Epoch: %d, train_loss: %f, val_loss: %f, train_acc: %f, val_acc: %f"
                      % (epoch, float(train_loss), float(val_loss), float(train_acc), float(val_acc)))
        else:
            metrics = {'val_acc': val_acc, 'val_loss': val_loss, 'train_acc': train_acc, 'train_loss': train_loss}
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
        max_val_acc = max(val_acc_list)
        max_train_acc = max(train_acc_list)
        min_val_loss = min(val_loss_list)
        min_train_loss = min(train_loss_list)
        print("Final epoch: %d, val_acc: %f"
              % (epoch, float(val_acc)))
        print("Highest validation accuracy @ epoch: %d, val_acc: %f"
              % (np.argmax(val_acc_list) + 1, float(max_val_acc)))
        print("Highest training accuracy @ epoch: %d, train_acc: %f"
              % (np.argmax(train_acc_list) + 1, float(max_train_acc)))
        print("Lowest validation loss @ epoch: %d, val_loss: %f"
              % (np.argmin(val_loss_list) + 1, float(min_val_loss)))
        print("Lowest training loss @ epoch: %d, train_loss: %f"
              % (np.argmin(train_loss_list) + 1, float(min_train_loss)))
        if checkpt1 > 0:
            print('60%% epoch: %d' % checkpt1)
        if checkpt2 > 0:
            print('65%% epoch: %d' % checkpt2)
        if checkpt3 > 0:
            print('70%% epoch: %d' % checkpt3)
        print('\n')
        return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list


if __name__ == '__main__':
    X_train, X_val, X_outer_val, X_test, y_train, y_val, y_outer_val, y_test \
        = get_data(sst5='original',
                   inner_split=True,
                   costco='none')

    # X_train, y_train = add_aug_data(X_train, y_train, 'data/sst5/aug/sst-5_aug_2.parquet')

    condition = None
    if condition:
        df_train = pd.DataFrame({'vector': [x for x in X_train], 'rating': y_train})
        df_train = df_train[df_train['rating'].isin(condition)]
        X_train = np.vstack(df_train['vector'].tolist())
        y_train = np.vstack(df_train['rating'].tolist()).reshape(-1)
        y_train = np.array([condition.index(x) for x in y_train])

        df_val = pd.DataFrame({'vector': [x for x in X_val], 'rating': y_val})
        df_val = df_val[df_val['rating'].isin(condition)]
        X_val = np.vstack(df_val['vector'].tolist())
        y_val = np.vstack(df_val['rating'].tolist()).reshape(-1)
        y_val = np.array([condition.index(x) for x in y_val])

    # feature_selector = SelectKBest(f_classif, k=256)
    # X_train = feature_selector.fit_transform(X_train, y_train)
    # X_val = feature_selector.transform(X_val)

    # X_train, y_train = RandomOverSampler(random_state=123).fit_resample(X_train, y_train)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)

    # # If running on dev set
    # X_train = np.vstack([X_train, X_val])
    # y_train = np.hstack([y_train, y_val])
    # X_val = X_outer_val
    # y_val = y_outer_val

    ohe = 'coral'
    in_features = X_train.shape[1]
    if ohe == ('coral' or 'corn'):
        out_features = len(np.unique(y_train)) - 1
    else:
        out_features = len(np.unique(y_train))

    data_train = RatingDataset(X_train, y_train, ohe=ohe)
    data_val = RatingDataset(X_val, y_val, ohe=ohe)
    train_id = ray.put(data_train)
    val_id = ray.put(data_val)

    search_space = {
        'lr': 3.4003933256295715e-05,
        'alpha1': 0.07344494001128585,
        'alpha2': 0.000442504638714582,
        'batch_size': 64,
        'n_neurons1': 10,
        'n_neurons2': 969,
        'n_neurons3': 0,
        'weight_decay': 6.169063238171836e-06,
        'input_dropout': 0,
        'dropout_p1': 0,
        'dropout_p2': 0,
        'max_num_epochs': 500,
        'min_num_epochs': 10,
        'checkpoint_interval': 10,
    }

    ray_tune = False
    resume = False
    save_plot = True
    search_name = 'sst5'

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
            metric="val_loss",
            mode="min",
            seed=123,
            # points_to_evaluate=[{'lr': 1e-2, 'alpha1': 1e-1, 'alpha2': 1e-3}],
        )

        trial_stopper = TrialPlateauStopper(
            metric='val_acc',
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
                    metric='val_loss',
                    mode='min',
                    scheduler=scheduler_asha,
                    search_alg=optuna_search,
                    num_samples=100,
                    trial_dirname_creator=short_dirname,
                ),
                run_config=train.RunConfig(
                    name=search_name + '-' + datetime.now().strftime('%Y%m%d_%H%M%S'),
                    storage_path=RAY_RESULTS_PATH,
                    # stop=trial_stopper,
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

        # best_result = results.get_best_result("val_acc", mode="max")
        # best_config = best_result.config
        # best_acc = best_result.metrics['val_acc']
        # best_result_epochs = best_result.metrics['training_iteration']

        best_result = results.get_best_result("val_loss", mode="min")
        best_config = best_result.config
        best_loss = best_result.metrics['val_loss']
        best_acc = best_result.metrics['val_acc']
        best_result_epochs = best_result.metrics['training_iteration']

        print('Best config: ' + str(best_config))
        print('Lowest loss: ' + str(best_loss))
        print('Highest accuracy: ' + str(best_acc))
        print('Epochs: ' + str(best_result_epochs))
    else:
        # Run without Ray tune
        start_time = time.time()
        trained_model, train_loss, val_loss, train_acc, val_acc = train_model(
            search_space,
            print_interval=10,
            # early_stop_patience=2,
            # early_stop_min_delta=0.005
        )
        print("--- %s seconds ---" % (time.time() - start_time))

        epochs = range(1, len(train_loss) + 1)

        if save_plot:
            plt.figure(1)
            plt.plot(epochs, train_loss, label='Training loss')
            plt.plot(epochs, val_loss, label='Validation loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(loc='best')
            plt.savefig('images_all/loss.png')

            plt.figure(2)
            plt.plot(epochs, train_acc, label='Training accuracy')
            plt.plot(epochs, val_acc, label='Validation accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend(loc='best')
            plt.savefig('images_all/acc.png')
