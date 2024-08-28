from initialization import set_seed, use_gpu, set_ray_settings, short_dirname
from data import RatingDataset, get_data, load_data, load_data_from_ray, get_aug_data, add_aug_data
from neural_network import train_func, eval_func, test_model, get_argmax, get_model_structure, save_model_structure, \
    EarlyStopper
from evaluation import evaluate_model, save_evaluation
import models

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label, corn_label_from_logits
from sklearn.metrics import accuracy_score, confusion_matrix, root_mean_squared_error, ConfusionMatrixDisplay

device = 'cuda'

_, _, _, X_test, _, _, _, y_test \
    = get_data(sst5='original',
               path='data/sst5/fine_tuned/0730/',
               costco='none')

ohe = 'none'

if ohe == 'none':
    get_pred = get_argmax
elif ohe == 'coral':
    get_pred = proba_to_label
elif ohe == 'corn':
    get_pred = corn_label_from_logits
else:
    get_pred = get_argmax

if ohe == ('coral' or 'corn'):
    out_features = len(np.unique(y_test)) - 1
else:
    out_features = len(np.unique(y_test))

num_samples = X_test.shape[0]

data_test = RatingDataset(X_test, y_test, ohe=ohe)

test_loader = DataLoader(data_test, batch_size=num_samples)

model = models.FeedForwardNet(
    input_size=1024,
    output_size=out_features,
    hidden_layers=[19, 11],
    input_dropout=0.3924090355094955,
    dropout_p=[0.04924066174932384, 0.1],
)

state_dict = torch.load('models/sst5/3.pt')
model.load_state_dict(state_dict)
model.to(device)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_idx, (vectors, ratings, ratings_original) in enumerate(test_loader):
        vectors = vectors.to(device)
        ratings_original = ratings_original.to(device)

        batch_size = vectors.shape[0]
        outputs = model(vectors.view(batch_size, -1))

        total += batch_size
        predicted = get_pred(outputs)
        correct += (predicted == ratings_original).sum().item()

print('Accuracy: %f' % (correct / total))

y_prob = outputs.cpu().numpy()
y_pred = predicted.cpu().numpy()

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, normalize='true')

print('Accuracy: %f' % acc)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array([1, 2, 3, 4, 5]))
disp.plot()
plt.show()
