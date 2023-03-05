import numpy as np
import torch
from .utils import extract_unlabeled_prob

def random_sampling(
    model, dataloader, dataset, nb_query_samples: int, labeled_idx: np.ndarray):
    
    # set unlabeled index
    unlabeled_idx = np.arange(len(dataset))[~labeled_idx]
    
    # extract unlabeled probs
    probs = extract_unlabeled_prob(model=model, dataloader=dataloader, dataset=dataset, unlabeled_idx=unlabeled_idx)
    
    # select random select
    max_confidence = probs.max(1)[0]
    shuffled_idx = torch.randperm(len(max_confidence))
    select_idx = unlabeled_idx[shuffled_idx[:nb_query_samples]]
    
    return select_idx