import numpy as np
from .utils import extract_unlabeled_prob

def margin_sampling(
    model, dataloader, dataset, nb_query_samples: int, labeled_idx: np.ndarray):
    
    # set unlabeled index
    unlabeled_idx = np.arange(len(dataset))[~labeled_idx]
    
    # extract unlabeled probs
    probs = extract_unlabeled_prob(model=model, dataloader=dataloader, dataset=dataset, unlabeled_idx=unlabeled_idx)
    
    # select margin between top two class probability
    sorted_desc_prob, _ = probs.sort(descending=True)
    prob_margin = sorted_desc_prob[:,0] - sorted_desc_prob[:,1]
    select_idx = unlabeled_idx[(prob_margin).sort()[1][:nb_query_samples]]
    
    return select_idx