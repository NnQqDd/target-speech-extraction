# python3 main.py --engine-mode train
# python3 main.py --engine-mode test
# python3 main.py --engine-mode infer_sample --sample_file "/home/duyn/ActableDuy/Speech Separation/Vi_SepReformer_Base/samples/vi_overlap_1.wav" --output_dir "/home/duyn/ActableDuy/Speech Separation/samples"

import warnings
warnings.filterwarnings("ignore")   # suppress all warnings

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
import torch
from setproctitle import setproctitle
from loguru import logger
from wandb_helper import *
from dataset import get_dataloaders
from engine import Engine


log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log/loguru.log')
logger.add(log_file_path, level='DEBUG', mode='w')


def get_gpus_by_vram():
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'
    ]
    output = subprocess.check_output(cmd).decode('utf-8').strip().splitlines()
    usage = []
    for line in output:
        idx_str, mem_str = line.split(',')
        usage.append((int(idx_str.strip()), int(mem_str.strip())))
    usage.sort(key=lambda x: x[1])
    return usage


def get_checkpoint_path_n_latest_epoch(engine_mode, prefix):
    # checkpoint_dir is the same as weights_path
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log', 'pretrain_weights')
    
    if engine_mode in ('test', 'infer_sample', 'extract'):
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
    parser = argparse.ArgumentParser(description='OMG!')
    parser.add_argument('-m', '--mode', choices=['train', 'test', 'infer_sample', 'extract'])
    parser.add_argument('-c', '--config')
    parser.add_argument('-f', '--sample-file')
    parser.add_argument('-o', '--output-dir')
    args = parser.parse_args()
    
    if args.mode == 'train':
        process_name = f'Vi-SepReformer-{Path(args.config).stem.capitalize()} Training'
        setproctitle(process_name)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.mode == 'train':
        print(config['wandb'], end='\n')
        print(config['dataset'], end='\n')
        print(config['dataloader'], end='\n')
        print(config['engine'], end='\n')

    checkpoint = get_checkpoint_path_n_latest_epoch(args.mode, Path(args.config).stem)
    gpus = get_gpus_by_vram()
    if args.mode == 'infer_sample':
        logger.info('GPU with the highest available VRAM will be used.')
        engine = Engine(Path(args.config).stem, config, checkpoint, [gpus[0][0]], None)
        engine.inference_sample(args.sample_file, args.output_dir)

    elif args.mode == 'test':
        logger.info('GPU with the highest available VRAM will be used.')
        dataloaders = get_dataloaders(args.mode, config, 1)
        engine = Engine(Path(args.config).stem, config, checkpoint, [gpus[0][0]], None)
        engine.test(dataloaders['test'])

    elif args.mode == 'extract':
        engine = Engine(Path(args.config).stem, config, checkpoint, [gpus[0][0]], None)
        engine.extract()
        
    else:
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

        n_gpus = config['engine']['gpus']
        if not isinstance(n_gpus, list): # must be int
            gpus = gpus[:n_gpus]
            gpu_ids = [gpu[0] for gpu in gpus]
        else:
            gpu_ids = n_gpus
        logger.info(f'Training on gpu(s) {gpu_ids}...')
        dataloaders = get_dataloaders(args.mode, config, len(gpu_ids))
        # dataloaders = None
        engine = Engine(Path(args.config).stem, config, checkpoint, gpu_ids, wandb)
        engine.train(dataloaders) 
                
        wandb.finish()
        return 0


if __name__ == '__main__':
    main()
