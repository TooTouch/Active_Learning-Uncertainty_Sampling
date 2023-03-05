import numpy as np
from .utils import extract_unlabeled_prob

def least_confidence(
    model, dataloader, dataset, nb_query_samples: int, labeled_idx: np.ndarray):
    
    # set unlabeled index
    unlabeled_idx = np.arange(len(dataset))[~labeled_idx]
    
    # extract unlabeled probs
    probs = extract_unlabeled_prob(model=model, dataloader=dataloader, dataset=dataset, unlabeled_idx=unlabeled_idx)
    
    # select least confidence
    max_confidence = probs.max(1)[0]
    select_idx = unlabeled_idx[max_confidence.sort()[1][:nb_query_samples]]
    
    return select_idx