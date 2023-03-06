import numpy as np
import torch
from .utils import extract_unlabeled_prob

def random_sampling(
    model, dataloader, dataset, nb_query_samples: int, labeled_idx: np.ndarray):
    
    # set unlabeled index
    unlabeled_idx = np.arange(len(dataset))[~labeled_idx]
    np.random.shuffle(unlabeled_idx)
    select_idx = unlabeled_idx[:nb_query_samples]
    
    return select_idx