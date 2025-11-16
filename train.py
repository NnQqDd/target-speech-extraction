import random
random.seed(42)

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json
import argparse
import uuid
from setproctitle import setproctitle
setproctitle("AV-TSE training")

import yaml
from loguru import logger
logger.remove()
logger.add(sys.stdout)

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from utilities import *
from dataset import *
from models import *


def compute_SDR_loss(preds, targets, eps=1e-10):
    def l2norm_sq(mat, keepdim=False):
        return (mat * mat).sum(dim=-1, keepdim=keepdim)

    preds_zm = preds - torch.mean(input=preds, dim=-1, keepdim=True)
    targets_zm = targets - torch.mean(input=targets, dim=-1, keepdim=True)
    targets_zm_scale = targets_zm * torch.sum(preds_zm * targets_zm, dim=-1, keepdim=True) / (l2norm_sq(targets_zm, keepdim=True) + eps)
    A = l2norm_sq(targets_zm_scale)
    B = l2norm_sq(preds_zm - targets_zm_scale)
    return -10 * torch.log10(eps + A / (B + eps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script")
    parser.add_argument("--config", "-c", type=str, help="Checkpoint path")
    parser.add_argument("--ckpt", "-ck", type=str, help="Config path")
    args = parser.parse_args()

    BASE_PATH = os.path.abspath(os.path.dirname(__file__))
    fill_path = lambda x: os.path.join(BASE_PATH, x)

    run_id = str(uuid.uuid4())
    logger.info(f'Run ID = {run_id}')
    os.makedirs(fill_path(f'runs/{run_id}'), exist_ok=True)
    
    # now = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"; print(now)
    config_path = fill_path('config.yaml') if args.config is None else args.config 
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(fill_path(f'runs/{run_id}/config.json'), 'w') as f:    
        # print(config, file=f)
        json.dump(config, f, indent=3)
    print(json.dumps(config, indent=3))

    logger.info("Loading speech metadata...")
    speech_metadatas = get_speech_metadata(config)

    logger.info("Preparing datasets and dataloaders...")
    dataloaders = get_dataloaders(speech_metadatas, config['dataset'], config['dataloader'])
    
    '''
    device = None
    current_free = 0
    for i in range(0, torch.cuda.device_count()):
        free_mem, total_mem = torch.cuda.mem_get_info(torch.device(f"cuda:{i}"))
        if free_mem > current_free:
            current_free = max(current_free, free_mem)
            device = torch.device(f"cuda:{i}")
    assert device is not None, "No CUDA device found."
    '''
    device = torch.device("cuda")
    logger.info(f"Current device has {torch.cuda.mem_get_info(device)[0]/(1024 ** 3):.2f} GB free memory.")

    model = MinimalAVTSE(config['model']).to(device)

    # Optimizer and Scheduler
    OptimizerClass = load_class(config['optimizer']['name'])
    optimizer = OptimizerClass(model.parameters(), config['optimizer']['args'])
    SchedulerClass = load_class(config['scheduler']['name'])
    scheduler = SchedulerClass(optimizer, **config['scheduler']['args'])

    os.makedirs(fill_path(f'runs/{run_id}/weights'), exist_ok=True)
    curr_epoch = 0
    if args.ckpt is not None and os.path.exists(args.ckpt):
        pattern = re.compile(r'epoch_(\d+)\.pth$')
        matches = pattern.findall(args.ckpt)   # returns list of all matches
        if matches:
            curr_epoch = int(matches[-1])      # last match
        logger.info(f'Recovering from {args.ckpt}, epoch {curr_epoch}...')
        state_dicts = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state_dicts['model_state_dict'])
        optimizer.load_state_dict(state_dicts['optimizer_state_dict'])
        scheduler.load_state_dict(state_dicts['scheduler_state_dict'])

    best_valid_loss = float('inf')
    for epoch in range(1, config['trainer']['n_epochs'] + 1):
        logger.info(f'{run_id}')
        if epoch <= curr_epoch:
            continue
        model.train()
        train_loss = 0.0
        pbar = tqdm(dataloaders['train'], desc=f"Epoch {epoch} - Train")
        for idx, (mixes, videos, targets) in enumerate(pbar):
            mixes = mixes
            preds = model(mixes, videos) 
            loss = compute_SDR_loss(preds, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()                
            optimizer.zero_grad()
            pbar.set_postfix(loss=train_loss/(idx + 1))
        avg_train_loss = train_loss / (idx + 1) # type: ignore

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(dataloaders['valid'], desc=f"Epoch {epoch} - Valid")
            for idx, (mixes, videos, targets) in enumerate(pbar):
                mixes, videos, targets = mixes.to(device), videos.to(device), targets.to(device)
                preds = model(mixes, videos)
                loss = compute_SDR_loss(preds, targets)
                valid_loss += loss.item()
                pbar.set_postfix(loss=valid_loss/(idx + 1))
        avg_valid_loss = valid_loss / (idx + 1) # type: ignore

        scheduler.step(avg_valid_loss)

        if avg_valid_loss < best_valid_loss:
            logger.info(f"Epoch {epoch}: best {avg_valid_loss:.6f} (previous: {best_valid_loss:.6f}).")
            best_valid_loss = avg_valid_loss
            torch.save({'model_state_dict': model.state_dict()}, fill_path(f'runs/{run_id}/weights/best.pth'))
            logger.info(f"Saved model.")
        
        if epoch % config['trainer']['save_frequency'] == 0:
            ckpt = {       
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(ckpt, fill_path(f'runs/{run_id}/weights/epoch_{epoch}.pth'))
            logger.info(f"Saved checkpoint.")

            