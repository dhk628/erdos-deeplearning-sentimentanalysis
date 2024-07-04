import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error, root_mean_squared_error, mean_absolute_error, \
    log_loss, accuracy_score, ConfusionMatrixDisplay, roc_auc_score, f1_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
import csv
import os

CLASS_NAMES = np.array([1, 2, 3, 4, 5])


def evaluate_model(y_test, y_pred, y_prob):
    results = {'model': 'logreg',
               'accuracy': accuracy_score(y_test, y_pred),
               'root_mean_squared_error': root_mean_squared_error(y_test, y_pred),
               'mean_absolute_error': mean_absolute_error(y_test, y_pred),
               'log_loss': log_loss(y_test, y_prob) if y_prob is not None else 'na',
               'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
               'f1_macro': f1_score(y_test, y_pred, average='macro'),
               'f1_micro': f1_score(y_test, y_pred, average='micro'),
               'roc_auc_weighted': roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted') if y_prob is not None else 'na',
               'roc_auc_macro': roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro') if y_prob is not None else 'na',
               'roc_auc_micro': roc_auc_score(y_test, y_prob, multi_class='ovr', average='micro') if y_prob is not None else 'na',
               'gmean_weighted': geometric_mean_score(y_test, y_pred, average='weighted'),
               'gmean_macro': geometric_mean_score(y_test, y_pred, average='macro'),
               'gmean_micro': geometric_mean_score(y_test, y_pred, average='micro'),
               'matthews': matthews_corrcoef(y_test, y_pred)}

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES, normalize='true')

    return results, cm


def save_evaluation(data_name, model_name, results, cm):
    eval_folder = 'evaluations/' + data_name
    csv_path = eval_folder + '/evaluations.csv'
    cm_folder = eval_folder + '/confusion_matrix'
    cm_path = cm_folder + '/' + model_name + '.png'

    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)
    else:
        with open(csv_path) as f:
            existing_models = [line.split(',')[0] for line in f]
            existing_models = existing_models[1:]
        if model_name not in existing_models:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=results.keys())
                writer.writerow(results)
        else:
            print('Model already exists in ' + csv_path)

    if os.path.exists(cm_path):
        print('Confusion Matrix already exists in ' + cm_folder)
    else:
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot().figure_.savefig(cm_path)
