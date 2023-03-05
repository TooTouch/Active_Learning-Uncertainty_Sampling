import numpy as np
import torch
from .utils import extract_unlabeled_prob

def entropy_sampling(
    model, dataloader, dataset, nb_query_samples: int, labeled_idx: np.ndarray):
    
    # set unlabeled index
    unlabeled_idx = np.arange(len(dataset))[~labeled_idx]
    
    # extract unlabeled probs
    probs = extract_unlabeled_prob(model=model, dataloader=dataloader, dataset=dataset, unlabeled_idx=unlabeled_idx)
    
    # select maximum entropy
    entropy = (-(probs*torch.log(probs))).sum(dim=1)
    select_idx = unlabeled_idx[entropy.sort(descending=True)[1][:nb_query_samples]]
    
    return select_idx