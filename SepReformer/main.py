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

from modules.wandb_helper import *
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
    parser.add_argument('-r', '--reference-file')
    parser.add_argument('-o', '--output-dir')
    args = parser.parse_args()
    
    if args.mode == 'train':
        process_name = f'Vi-SepReformer-{Path(args.config).stem.capitalize()} Training'
        setproctitle(process_name)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    checkpoint = get_checkpoint_path_n_latest_epoch(args.mode, Path(args.config).stem)
    gpus = get_gpus_by_vram()

    if args.mode == 'train':
        print('Please use train.py!')

    if args.mode == 'infer_sample':
        print('This mode is not supported yet!')

    elif args.mode == 'test':
        print('This mode is no longer supported!')

    elif args.mode == 'extract':
        engine = Engine(Path(args.config).stem, config, checkpoint, [gpus[0][0]], None)
        engine.extract()
        
    return 0


if __name__ == '__main__':
    main()
