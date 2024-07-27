from initialization import set_seed, use_gpu, set_ray_settings, short_dirname
from data import RatingDataset, get_data, load_data, load_data_from_ray, get_aug_data, add_aug_data
from neural_network import train_func, eval_func, test_model, get_argmax, get_model_structure, save_model_structure, \
    EarlyStopper
from evaluation import evaluate_model, save_evaluation
import models

import torch
from torch.utils.data import Dataset, DataLoader
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label, corn_label_from_logits


device = 'cuda'

_, _, _, X_test, _, _, _, y_test \
    = get_data(sst5='original',
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

num_samples = X_test.shape[0]

data_test = RatingDataset(X_test, y_test, ohe=ohe)

test_loader = DataLoader(data_test, batch_size=num_samples)

model = models.FeedForwardNet(
    input_size=1024,
    output_size=5,
    hidden_layers=[251],
)

state_dict = torch.load('models/sst5/1.pt')
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
