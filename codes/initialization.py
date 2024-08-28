import random
import numpy as np
import torch


def set_seed(random_seed, device='cpu'):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    g = torch.Generator(device=device)
    g.manual_seed(random_seed)

    return g


def use_gpu():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # torch.set_default_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    return device


def set_ray_settings(machine='pc'):
    if machine == 'math_a':
        path = "/export/dohoonk/erdos/codes/dohoon/.ray_results"
        resources = {'cpu': 2, 'gpu': 0.01785714285}  # 56 concurrent trials
    elif machine == 'math_b':
        path = "/export/dohoonk/erdos/codes/dohoon/.ray_results"
        resources = {'cpu': 2, 'gpu': 0.125}  # 64 concurrent trials
    else:
        path = "D:/GitHub/Data Science/erdos-deeplearning-companydiscourse/codes/dohoon/.ray_results"
        resources = {'cpu': 8, 'gpu': 0.5}

    return path, resources


def short_dirname(trial):
    return "trial_" + str(trial.trial_id)