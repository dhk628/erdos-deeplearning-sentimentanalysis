# import copy
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
# from coral_pytorch.dataset import levels_from_labelbatch
# from evaluation import evaluate_model, save_evaluation
# from sklearn.metrics import confusion_matrix, mean_squared_error, root_mean_squared_error, mean_absolute_error, \
#     log_loss, accuracy_score, ConfusionMatrixDisplay, roc_auc_score, f1_score, matthews_corrcoef
#
#
# class RatingDataset(Dataset):
#     def __init__(self, vector, rating, ohe=None):
#         self.vector = torch.tensor(vector, dtype=torch.float32)
#         if ohe is None:
#             self.rating = torch.tensor(rating, dtype=torch.int64)
#         if ohe == 'ordinal':
#             self.rating = levels_from_labelbatch(rating, num_classes=5)
#         self.ohe = ohe
#
#     def __len__(self):
#         return len(self.rating)
#
#     def __getitem__(self, idx):
#         vector = self.vector[idx]
#         rating = self.rating[idx]
#         ohe = self.ohe
#         return vector, rating, ohe
#
#
# def load_data(dataset):
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
#
# def train_fnn(data_train, data_val, model, batch_size, loss_fn, optimizer, n_epochs):
#     train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
#     vectors_val, ratings_val = data_val[:]
#
#     for epoch in range(n_epochs):
#         model.train()
#         running_loss = 0.0
#         running_corrects = 0
#         running_total = 0
#         for vectors, ratings in train_loader:
#             batch_size = vectors.shape[0]
#             outputs = model(vectors.view(batch_size, -1))
#             loss = loss_fn(outputs, ratings.view(batch_size))
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item() * batch_size
#             running_total += batch_size
#
#             _, predicted = torch.max(outputs, dim=1)
#             running_corrects += (predicted == ratings.view(batch_size)).sum().item()
#
#         train_loss = running_loss / running_total
#         train_acc = running_corrects / running_total
#
#         # evaluate on validation data
#         model.eval()
#         with torch.no_grad():
#             outputs_val = model(vectors_val)
#             _, predicted_val = torch.max(outputs_val, dim=1)
#             val_loss = loss_fn(outputs_val, ratings_val.view(-1))
#             val_acc = accuracy_score(ratings_val.cpu().numpy(), predicted_val.cpu().numpy())
#
#         if epoch % 10 == 0 or epoch == n_epochs - 1:
#             print("Epoch: %d, train_loss: %f, val_loss: %f, train_acc: %f, val_acc: %f"
#                   % (epoch, float(train_loss), float(val_loss), float(train_acc), float(val_acc)))
#
#
# def test_model(model, data_test):
#     model.eval()
#     with torch.no_grad():
#         outputs_test = model(X_test)
#         _, predicted_test = torch.max(outputs_test, dim=1)
#
#
# RANDOM_SEED = 123
# CLASS_NAMES = np.array([1, 2, 3, 4, 5])
# DATA_NAME = 'sst5'
# N_OUT = 5
#
# if __name__ == '__main__':
#     np.random.seed(RANDOM_SEED)
#     torch.manual_seed(RANDOM_SEED)
#     torch.cuda.manual_seed(RANDOM_SEED)
#
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#     torch.set_default_device(device)
#     torch.backends.cudnn.benchmark = True
#
#     # Loading data
#     X_train, X_val, X_test, y_train, y_val, y_test = load_data(DATA_NAME)
#
#     # # Preprocessing before changing to tensors
#     # X_train, y_train = RandomOverSampler(random_state=RANDOM_SEED).fit_resample(X_train, y_train)
#
#     # Model definition
#     model = nn.Sequential(nn.Linear(X_train.shape[1], 128),
#                           nn.ReLU(),
#                           nn.Linear(128, N_OUT),
#                           nn.LogSoftmax(dim=1)
#                           )
#
#     # Hyperparameters
#     # batch_size = 64
#     # learning_rate = 1e-3
#     # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
#     # loss_fn = nn.NLLLoss()
#     # n_epochs = 500
#
#     # Load data using Pytorch
#     data_train = RatingDataset(X_train, y_train, ohe=None)
#     data_val = RatingDataset(X_val, y_val, ohe=None)
#     data_test = RatingDataset(X_test, y_test, ohe=None)
#
#     # train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
#     #
#     # for epoch in range(n_epochs):
#     #     running_loss = 0.0
#     #     running_corrects = 0
#     #     running_total = 0
#     #     for vectors, ratings in train_loader:
#     #         batch_size = vectors.shape[0]
#     #         outputs = model(vectors.view(batch_size, -1))
#     #         loss = loss_fn(outputs, ratings.view(batch_size))
#     #
#     #         optimizer.zero_grad()
#     #         loss.backward()
#     #         optimizer.step()
#     #
#     #         running_loss += loss.item() * batch_size
#     #         running_total += batch_size
#     #
#     #         _, predicted = torch.max(outputs, dim=1)
#     #         running_corrects += (predicted == ratings.view(batch_size)).sum().item()
#     #
#     #     train_loss = running_loss / running_total
#     #     train_acc = running_corrects / running_total
#     #
#     #     # evaluate on validation data
#     #     model.eval()
#     #     with torch.no_grad():
#     #         outputs_val = model(X_val)
#     #         _, predicted_val = torch.max(outputs_val, dim=1)
#     #         val_loss = loss_fn(outputs_val, y_val.view(-1))
#     #         val_acc = accuracy_score(y_val.cpu().numpy(), predicted_val.cpu().numpy())
#     #
#     #     if epoch % 10 == 0 or epoch == n_epochs - 1:
#     #         print("Epoch: %d, train_loss: %f, val_loss: %f, train_acc: %f, val_acc: %f"
#     #               % (epoch, float(train_loss), float(val_loss), float(train_acc), float(val_acc)))
#     #
#     #     model.train()
#
#     train_fnn(data_train, data_val,
#               model=model,
#               batch_size=32,
#               loss_fn=nn.NLLLoss(),
#               optimizer=optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True),
#               n_epochs=200)
#
#     # model_structure = {'input_normalization': 'none',
#     #                    'learning_rate': learning_rate,
#     #                    'optimizer': optimizer,
#     #                    'loss': loss_fn,
#     #                    'batch_size': 32,
#     #                    'n_epochs': n_epochs,
#     #                    'type': 'FNN',
#     #                    'layers': [module for module in model.modules() if not isinstance(module, nn.Sequential)],
#     #                    'state_dict': model.state_dict()}
#
#     # torch.save(model_structure, 'models/sst5/1.pt')
#
#     vectors_test, ratings_test = data_test[:]
#     model.eval()
#     with torch.no_grad():
#         outputs_test = model(vectors_test)
#         _, predicted_test = torch.max(outputs_test, dim=1)
#
#     probabilities_test = outputs_test.detach().cpu().numpy()
#     predicted_test = predicted_test.detach().cpu().numpy()
#     accuracy = accuracy_score(ratings_test.cpu().numpy(), predicted_test)
#     print("Accuracy: %f" % accuracy)
