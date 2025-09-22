### DPP NOTE - self.model.module ### 

import gc
import os
from collections import defaultdict
import csv
import datetime
import pathlib
import soundfile as sf
import librosa
from loguru import logger
from tqdm import tqdm
import torch

from .model import Model
from . import soc_implement


def peak_normalize(waveform, always=False):
    mx = waveform.abs().max()
    if mx > 1 or always:
        waveform /= mx
    return waveform


class Engine(object):
    def __init__(self, prefix, config, checkpoint, gpu_ids, wandb_logger=None): # do not call wandb_logger.finish() !
        model = Model(**config['model'])
        n_params = sum(p.numel() for p in model.parameters())
        n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Total parameters    : {n_params:,}')
        logger.info(f'Trainable parameters: {n_train_params:,}')
        logger.info(f'Note that BatchNorm1d has been disabled.')
        
        if wandb_logger is not None:
            wandb_logger.watch(model, log=config['wandb']['log'], log_freq=config['wandb']['log_freq'])

        criterions = soc_implement.CriterionFactory(config['criterion'], torch.device(f'cuda:{gpu_ids[0]}')).get_criterions()
        optimizers = soc_implement.OptimizerFactory(config['optimizer'], model.parameters()).get_optimizers()
        schedulers = soc_implement.SchedulerFactory(config['scheduler'], optimizers).get_schedulers()

        self.prefix = prefix
        self.config = config
        self.gpu_ids = gpu_ids
        self.device = torch.device(f'cuda:{gpu_ids[0]}')
        self.wandb_logger = wandb_logger

        self.main_scheduler, self.warmup_scheduler = schedulers
        self.main_optimizer = optimizers[0]

        self.PIT_SISNR_mag_loss, self.PIT_SISNR_time_loss = criterions
        
        # weights_path is the same as checkpoint_dir, use to save checkpoints
        self.weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..' ,'log', 'pretrain_weights')
        os.makedirs(self.weights_path, exist_ok=True)
        
        self.model = model.to(self.device)
        self.model_num_stages = self.model.num_stages
        self.start_checkpoint_path = checkpoint[0]
        if checkpoint[1] >= 1:
            checkpoint_dict = torch.load(checkpoint[0], map_location=self.device)
            self.model.load_state_dict(checkpoint_dict['model_state_dict'])
            try:
                self.main_optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            except Exception as e:
                logger.info(e)
            logger.info(f'Loaded model from {checkpoint[0]} .....')
            
        if len(self.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids, output_device=self.device.index)
          
        self.start_epoch = checkpoint[1] + 1

    def _inference_sample(self, mixture):
        self.model.eval()
        model_stride = self.config['model']['module_audio_enc']['stride']
        model_sr = self.config['dataset']['sample_rate']
        if len(mixture.shape) == 1:
            mixture = mixture[None]

        remains = mixture.shape[-1] % model_stride
        if remains != 0:
            padding = model_stride - remains
            mixture_padded = torch.nn.functional.pad(mixture, (0, padding), 'constant', 0)
        else:
            mixture_padded = mixture

        # with torch.inference_mode():
        with torch.no_grad():
            nnet_input = mixture_padded.to(self.device)
            estims, _ = self.model(nnet_input)

            results = []
            for i in range(self.config['model']['num_spks']):
                results.append(estims[i][..., :mixture.shape[-1]][0])
            return results

    def inference_sample(self, filepath, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        model_stride = self.config['model']['module_audio_enc']['stride']
        model_sr = self.config['dataset']['sample_rate']

        mixture, _ = librosa.load(filepath, sr=model_sr, mono=True)
        mixture = torch.tensor(mixture, dtype=torch.float32).to(self.device)
        results = self._inference_sample(mixture)

        stem_name = pathlib.Path(filepath).stem
        sf.write(
            os.path.join(output_dir, stem_name + '-in.wav'), 
            peak_normalize(mixture).cpu().numpy(), 
            model_sr
        )
        
        for estim in enumerate(results):
            sf.write(
                os.path.join(output_dir, stem_name + f'-out-{index}.wav'), 
                peak_normalize(estim).cpu().numpy(), 
                model_sr
            )

    def extract(self):
        if len(self.gpu_ids) > 1:
            model_instance = self.model.module
        else:
            model_instance = self.model
        path = os.path.join(self.weights_path, f'{self.prefix}.extracted.pth') 
        torch.save(
            {
                'model_state_dict': model_instance.state_dict()
            },
            path
        )
        logger.info(f'Model saved to {path}.')