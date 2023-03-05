import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn

def dataset_sampling(dataset, sample_idx: np.ndarray):
    # define unlabeled dataset
    unlabeled_dataset = deepcopy(dataset)
    unlabeled_dataset.data = dataset.data[sample_idx]
    unlabeled_dataset.targets = np.array(dataset.targets)[sample_idx]

    return unlabeled_dataset


def predict_prob(model, dataloader):        
    # predict
    probs = []
    
    # eval mode
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            outputs = model(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)
            probs.append(outputs.cpu())
            
    return torch.vstack(probs)


def extract_unlabeled_prob(model, dataloader, dataset, unlabeled_idx):
    # extract unlabeled dataset
    unlabeled_dataset = dataset_sampling(dataset=dataset, sample_idx=unlabeled_idx)
    
    # set unlabeled dataloader
    dataloader.dataset.data = unlabeled_dataset.data
    dataloader.dataset.targets = unlabeled_dataset.targets
    
    # predict unlabeled dataset
    probs = predict_prob(
        model       = model,
        dataloader  = dataloader
    )
    
    return probs