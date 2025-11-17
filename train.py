# WANDB_MODE=disabled
import random
random.seed(42)

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import argparse
import datetime
from setproctitle import setproctitle
setproctitle("AV-TSE training")

import yaml
from loguru import logger
logger.remove()
logger.add(sys.stdout)

import torch
import torchvision.transforms.v2 as v2
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cf", type=str, help="Config path")
    parser.add_argument("--ckpt", "-ck", type=str, help="Checkpoint path")
    args = parser.parse_args()

    BASE_PATH = os.path.abspath(os.path.dirname(__file__))
    fill_path = lambda x: os.path.join(BASE_PATH, x)

    config_path = fill_path('config.yaml') if args.config is None else args.config 
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    wandb.login(key=config['wandb']['api_key'])
    entity = config['wandb']['entity']
    project = config['wandb']['project']

    try:
        n_emptied = remove_empty_runs(entity, project, least=1)
        logger.info(f'{n_emptied} empty run(s) deleted.')
    except Exception as e:
        logger.info(e)

    run = wandb.init(
        project=project,
        config=config,
        entity=entity,
        name=datetime.datetime.now().strftime("%m.%d.%H.%M.%S"),
        # mode="offline"
    )
    os.makedirs(fill_path(f"weights/{run.id}"), exist_ok=True)
    with open(fill_path(f'weights/{run.id}/config.yaml'), 'w') as f:  
        yaml.dump(config, f, default_flow_style=False)  
        # json.dump(config, f, indent=3)

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

    ModelClass = load_class(config['model']['name'])
    model = ModelClass(**config['model']['args']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params:     {n_params:,}")
    logger.info(f"Trainable params: {n_trainable_params:,}")

    assert isinstance(model, torch.nn.Module)

    OptimizerClass = load_class(config['optimizer']['name'])
    optimizer = OptimizerClass(model.parameters(), **config['optimizer']['args'])
    assert isinstance(optimizer, torch.optim.Optimizer)
    
    SchedulerClass = load_class(config['scheduler']['name'])
    scheduler = SchedulerClass(optimizer, **config['scheduler']['args'])
    assert isinstance(scheduler,(
        torch.optim.lr_scheduler._LRScheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ))

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
    
    v2_transform = v2.Compose([
            v2.Resize(240),
            v2.CenterCrop(224),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def transform(videos):
        B, T, C, H, W = videos.shape
        videos = videos.reshape(B * T, C, H, W)  # merge batch & time
        videos = v2_transform(videos)  # shape: (B*T, 3, 224, 224)
        videos = videos.reshape(B, T, C, 224, 224)  # reshape back
        return videos

    best_valid_loss = float('inf')
    for epoch in range(1, config['trainer']['n_epochs'] + 1):
        if epoch <= curr_epoch:
            continue
        model.train()
        train_loss = 0.0
        pbar = tqdm(dataloaders['train'], desc=f"Epoch {epoch} - Train")
        for idx, (mixes, videos, targets) in enumerate(pbar):
            mixes, videos, targets = mixes.to(device), videos.to(device), targets.to(device)
            videos = transform(videos)
            preds = model(mixes, videos) 
            loss = compute_SDR_loss(preds, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()                
            optimizer.zero_grad()
            pbar.set_postfix(loss=train_loss/(idx + 1))
        avg_train_loss = train_loss / (idx + 1)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(dataloaders['valid'], desc=f"Epoch {epoch} - Valid")
            for idx, (mixes, videos, targets) in enumerate(pbar):
                mixes, videos, targets = mixes.to(device), videos.to(device), targets.to(device)
                videos = transform(videos)
                preds = model(mixes, videos)
                loss = compute_SDR_loss(preds, targets)
                valid_loss += loss.item()
                pbar.set_postfix(loss=valid_loss/(idx + 1))
        avg_valid_loss = valid_loss / (idx + 1)

        wandb.log({
            "Metrics/Train Loss": avg_train_loss,
            "Metrics/Valid Loss": avg_valid_loss,
            "Others/Learning Rate": optimizer.param_groups[0]["lr"],
        }, step=epoch)
        
        scheduler.step(avg_valid_loss)

        if avg_valid_loss < best_valid_loss:
            logger.info(f"Epoch {epoch}: best {avg_valid_loss:.6f} (previous: {best_valid_loss:.6f}).")
            best_valid_loss = avg_valid_loss
            torch.save({'model_state_dict': model.state_dict()}, fill_path(f'weights/{run.id}/best.pth'))
            logger.info(f"Saved model.")
        
        if epoch % config['trainer']['save_frequency'] == 0:
            ckpt = {       
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(ckpt, fill_path(f'weights/{run.id}/epoch_{epoch}.pth'))
            logger.info(f"Saved checkpoint in weights/{run.id}")

            