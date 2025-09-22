# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py -c base.yaml
# CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29600 train.py -c base.yaml
import warnings
warnings.filterwarnings("ignore")   # suppress all warnings

from collections import defaultdict
import gc
import shutil
import os
from pathlib import Path
import time
import datetime
import argparse
import pathlib
import subprocess
import yaml
from tqdm import tqdm
import torch
import torch.distributed as dist
from setproctitle import setproctitle
from loguru import logger

from modules.wandb_helper import *
from modules.ddp_dataset import get_dataloaders
from modules.ddp_engine import Engine


log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log/loguru.log')
logger.add(log_file_path, level='DEBUG', mode='w')


def get_checkpoint_path_n_latest_epoch(engine_mode, prefix):
    # checkpoint_dir is the same as weights_path
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log', 'pretrain_weights')
    
    if engine_mode in ('test', 'infer_sample'):
        check_lst = [f'{prefix}.test.pth', f'{prefix}.test.pt', f'{prefix}.best.pth', f'{prefix}.best.pt']
        for checkpoint_file in check_lst:
            file_path = os.path.join(checkpoint_dir, checkpoint_file)
            if os.path.exists(file_path):
                return (file_path, 0)    
    
    checkpoint_filenames = [f for f in os.listdir(checkpoint_dir) if prefix + '.' in f]
    epochs = []
    if not checkpoint_filenames: 
        logger.info('There are no checkpoint files!')
        return (None, 0)
    else:
        for f in checkpoint_filenames:
            try:
                epoch = int(f.split('.')[1])
                epochs.append(epoch)
            except:
                stem = pathlib.Path(f).stem
                if 'best' not in stem and 'test' not in stem:
                    logger.warning(f'Error parsing {f}.')
                epochs.append(-1)
                pass
        index = epochs.index(max(epochs))
        latest_checkpoint_file = os.path.join(checkpoint_dir, checkpoint_filenames[index])
        latest_epoch = epochs[index]
        return (latest_checkpoint_file, latest_epoch)


def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])

    parser = argparse.ArgumentParser(description='OMG!')
    parser.add_argument('-c', '--config')
    args = parser.parse_args()
    
    process_name = f'Vi-SepReformer-TSE Distributed'
    setproctitle(process_name)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if dist.get_rank() == 0:
        print(config['wandb'], end='\n')
        print(config['dataset'], end='\n')
        print(config['dataloader'], end='\n')
        print(config['engine'], end='\n')
        print('Number of gpus/gpu list will be ignored.')

    checkpoint = get_checkpoint_path_n_latest_epoch('train', Path(args.config).stem)

    if dist.get_rank() == 0:
        # 1) Log in
        wandb.login(key=config['wandb']['api_key'])
        entity = config['wandb']['entity']
        project = config['wandb']['project']

        # 2) Clean out any empty / zero‐step runs
        try:
            n_emptied = remove_empty_runs(entity, project, least=1)
            logger.info(f'{n_emptied} empty run(s) deleted.')
        except Exception as e:
            logger.info(e)
            
        # 3) Create a new run
        recent_name = ''
        if config['wandb']['resume'] == True:
            _, recent_name = get_most_recent_run_id(entity, project)
            recent_name = f'<-{recent_name}'
        run = wandb.init(
            project=project,
            config=config,
            entity=entity,
            dir='log',
            name=datetime.datetime.now().strftime("%m%d%H%M%S") + recent_name,
        )
    # wandb = None
    dataloaders = get_dataloaders(config)

    '''
    if dist.get_rank() == 0:
        label_count = defaultdict(int) 
        for _, _, labels in tqdm(dataloaders['train']):
            for label in labels:
                label_count[label] += 1
        for key in sorted(label_count.keys()):
            print(key, ':', label_count[key])
    '''

    engine = Engine(Path(args.config).stem, config, checkpoint, local_rank, wandb)
    engine.train(dataloaders) 
            
    wandb.finish()

    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    main()
