import numpy as np
import pandas as pd
import os
import random
import wandb
import torch
import argparse
import yaml
import logging
from copy import deepcopy

from train import fit
from datasets import create_dataset, create_dataloader
from models import *
from log import setup_default_logging

from accelerate import Accelerator

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def run(cfg):

    # make save directory
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['DATASET']['dataname'], cfg['MODEL']['modelname'], cfg['EXP_NAME'])
    os.makedirs(savedir, exist_ok=True)

    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg['TRAIN']['grad_accum_steps'],
        mixed_precision             = cfg['TRAIN']['mixed_precision']
    )

    setup_default_logging(log_path=os.path.join(savedir, 'log.txt'))
    torch_seed(cfg['SEED'])

    # set device
    _logger.info('Device: {}'.format(accelerator.device))

    # load dataset
    trainset, testset = create_dataset(datadir=cfg['DATASET']['datadir'], dataname=cfg['DATASET']['dataname'])

    # set loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # set active learning arguments
    nb_round = (cfg['AL']['n_end'] - cfg['AL']['n_start'])/cfg['AL']['n_query']
    
    if nb_round % int(nb_round) != 0:
        nb_round = int(nb_round) + 1
    else:
        nb_round = int(nb_round)
    
    # logging
    _logger.info('[total samples] {}, [initial samples] {} [qeury samples] {} [end samples] {} [total round] {}'.format(
        len(trainset), cfg['AL']['n_start'], cfg['AL']['n_query'], cfg['AL']['n_end'], nb_round))
    
    # inital sampling labeling
    sample_idx = np.arange(len(trainset))
    np.random.shuffle(sample_idx)
    
    labeled_idx = np.zeros_like(sample_idx, dtype=bool)
    labeled_idx[sample_idx[:cfg['AL']['n_start']]] = True
    
    # select strategy
    strategy = __import__('query_strategies').__dict__[cfg['AL']['strategy']](
        n_query     = cfg['AL']['n_query'], 
        labeled_idx = labeled_idx, 
        dataset     = trainset,
        batch_size  = cfg['DATASET']['batch_size'],
        num_workers = cfg['DATASET']['num_workers']
    )
    
    # define train dataloader
    trainloader = create_dataloader(
        dataset     = strategy.dataset_sampling(sample_idx=labeled_idx), 
        batch_size  = cfg['DATASET']['batch_size'], 
        shuffle     = True, 
        num_workers = cfg['DATASET']['num_workers']
    )
    
     # define test dataloader
    testloader = create_dataloader(
        dataset     = testset, 
        batch_size  = cfg['DATASET']['test_batch_size'], 
        shuffle     = False, 
        num_workers = cfg['DATASET']['num_workers']
    )
    
    # load init model
    model_init = __import__('models').__dict__[cfg['MODEL']['modelname']](num_classes=cfg['DATASET']['num_classes']) 
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model_init.parameters()])))
    
    # define log df
    log_df = pd.DataFrame(
        columns=['round','acc']
    )
    
    # run
    for r in range(nb_round+1):
        
        if r != 0:    
            # query sampling    
            query_idx = strategy.query(model)
            trainloader = strategy.update(query_idx)
            
        # logging
        _logger.info('[Round {}/{}] training samples: {}'.format(r, nb_round, len(trainloader.dataset)))
        
        # build Model
        model = deepcopy(model_init)
        
        # optimizer
        optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg['OPTIMIZER']['opt_name']](model.parameters(), lr=cfg['OPTIMIZER']['lr'])

        # scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['TRAIN']['epochs'], T_mult=1, eta_min=0.00001)
        
        # prepraring accelerator
        model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
            model, optimizer, trainloader, testloader, scheduler
        )
        
        # initialize wandb
        if cfg['TRAIN']['use_wandb']:
            wandb.init(name=cfg['EXP_NAME']+f'_round{r}', project='Active Learning - Benchmark', config=cfg)        

        # fitting model
        test_results = fit(
            model        = model, 
            trainloader  = trainloader, 
            testloader   = testloader, 
            criterion    = criterion, 
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            epochs       = cfg['TRAIN']['epochs'], 
            use_wandb    = cfg['TRAIN']['use_wandb'],
            log_interval = cfg['TRAIN']['log_interval'],
        )

        # save results
        log_df = log_df.append({
            'round' : r,
            'acc'   : test_results['acc']
        }, ignore_index=True)
        
        log_df.to_csv(
            os.path.join(savedir, f"total_{len(trainset)}-init_{cfg['AL']['n_start']}-query_{cfg['AL']['n_query']}-round_{nb_round}.csv"),
            index=False
        )    
        
        _logger.info('append result [shape: {}]'.format(log_df.shape))
        
        wandb.finish()
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Active Learning - Benchmark')
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')    

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    run(cfg)