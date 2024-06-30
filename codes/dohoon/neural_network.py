from data import RatingDataset, load_data
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
from evaluation import evaluate_model, save_evaluation
from sklearn.metrics import confusion_matrix, mean_squared_error, root_mean_squared_error, mean_absolute_error, \
    log_loss, accuracy_score, ConfusionMatrixDisplay, roc_auc_score, f1_score, matthews_corrcoef
from torch.utils.data import WeightedRandomSampler


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_func(train_loader, model, optimizer, loss_fn, get_pred, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (vectors, ratings) in enumerate(train_loader):
        vectors = vectors.to(device)
        ratings = ratings.to(device)

        batch_size = vectors.shape[0]
        outputs = model(vectors)
        loss = loss_fn(outputs, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += batch_size
        running_loss += loss.item() * batch_size

        predicted = get_pred(outputs)
        correct += (predicted == ratings).sum().item()

    return running_loss / total, correct / total


def eval_func(val_loader, model, loss_fn, get_pred, device):
    model.eval()
    correct = 0
    running_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch_idx, (vectors, ratings) in enumerate(val_loader):
            vectors = vectors.to(device)
            ratings = ratings.to(device)

            batch_size = vectors.shape[0]
            outputs = model(vectors.view(batch_size, -1))
            loss = loss_fn(outputs, ratings)

            total += batch_size
            running_loss += loss.item() * batch_size

            predicted = get_pred(outputs)
            correct += (predicted == ratings).sum().item()

    return running_loss / total, correct / total



def train_fnn(data_train, data_val, model, batch_size, loss_fn, optimizer, n_epochs, get_pred, data_loader_generator,
              early_stop_patience=0, early_stop_min_delta=-1):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    g = torch.Generator(device='cpu')
    g.manual_seed(123)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, generator=g)
    vectors_val, ratings_val = data_val[:]
    ohe = data_train.ohe
    early_stopper = EarlyStopper(patience=early_stop_patience, min_delta=early_stop_min_delta)

    if ohe == 'ordinal':
        ratings_val_original = proba_to_label(ratings_val)
    else:
        ratings_val_original = ratings_val

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
        for vectors, ratings in train_loader:
            vectors = vectors.to(device='cuda')
            ratings = ratings.to(device='cuda')
            batch_size = vectors.shape[0]
            outputs = model(vectors.view(batch_size, -1))
            loss = loss_fn(outputs, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            running_total += batch_size

            if ohe == 'ordinal':
                ratings_original = proba_to_label(ratings)
            else:
                ratings_original = ratings
            predicted = get_pred(outputs)
            running_corrects += (predicted == ratings_original).sum().item()

        train_loss = running_loss / running_total
        train_acc = running_corrects / running_total

        # evaluate on validation data
        model.eval()
        with torch.no_grad():
            vectors_val = vectors_val.to(device='cuda')
            ratings_val = ratings_val.to(device='cuda')
            outputs_val = model(vectors_val)
            predicted_val = get_pred(outputs_val)
            val_loss = loss_fn(outputs_val, ratings_val)
            val_acc = accuracy_score(ratings_val_original.cpu().numpy(), predicted_val.cpu().numpy())
            if early_stop_patience > 0:
                if early_stopper.early_stop(val_loss):
                    break

        # if epoch % 10 == 9 or epoch == 0:
        if epoch >= 0:
            print("Epoch: %d, train_loss: %f, val_loss: %f, train_acc: %f, val_acc: %f"
                  % (epoch + 1, float(train_loss), float(val_loss), float(train_acc), float(val_acc)))

    print("Final epoch: %d, val_acc: %f \n"
          % (epoch + 1, float(val_acc)))
    return float(val_acc)


def test_model(model, data_test, get_pred, get_prob=None):
    model.eval()
    with torch.no_grad():
        vectors_test, ratings_test = data_test[:]
        outputs_test = model(vectors_test)
        predicted_test = get_pred(outputs_test)
        probabilities_test = get_prob(outputs_test) if get_prob is not None else outputs_test
    probabilities_test = probabilities_test.detach()
    predicted_test = predicted_test.detach()

    return ratings_test, predicted_test, probabilities_test


def get_model_structure(model, training_dict):
    model_structure = {'input_normalization': 'none',
                       'ohe': training_dict['ohe'],
                       'train_data': training_dict['train_data'],
                       'optimizer': str(training_dict['optimizer']),
                       'loss': str(training_dict['loss_fn']),
                       'batch_size': training_dict['batch_size'],
                       'n_epochs': training_dict['n_epochs'],
                       'modules': str([module for module in model.modules()]),
                       'state_dict': model.state_dict(),
                       'get_pred': str(training_dict['get_pred'].__name__)}

    return model_structure


def save_model_structure(model_structure, test_data_name):
    path = 'models/' + test_data_name
    n = len([name for name in os.listdir(path)])
    torch.save(model_structure, path + '/' + str(n) + '.pt')


def get_argmax(vector):
    return torch.argmax(vector, dim=1)
