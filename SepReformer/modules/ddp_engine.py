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
import torch.distributed as dist
import torch

from .model import Model
from . import soc_implement


def apply_cmvn(feats):
    feats = feats - feats.mean(1, keepdims=True)
    feats = feats / feats.std(1, keepdims=True)
    return feats


class Engine(object):
    def __init__(self, prefix, config, checkpoint, local_rank, wandb_logger=None): # do not call wandb_logger.finish() !
        model = Model(**config['model'])
        if dist.get_rank() == 0:
            n_params = sum(p.numel() for p in model.parameters())
            n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'Total parameters    : {n_params:,}')
            logger.info(f'Trainable parameters: {n_train_params:,}')
            logger.info(f'Note that BatchNorm1d has been disabled.')
        

        if dist.get_rank() == 0 and wandb_logger is not None:
            logger.info('Gradient and parameter logging will be skipped.')

        criterions = soc_implement.CriterionFactory(config['criterion'], torch.device(f'cuda:{local_rank}')).get_criterions()
        optimizers = soc_implement.OptimizerFactory(config['optimizer'], model.parameters()).get_optimizers()
        schedulers = soc_implement.SchedulerFactory(config['scheduler'], optimizers).get_schedulers()

        self.prefix = prefix
        self.config = config
        self.device = torch.device(f'cuda:{local_rank}')
        self.wandb_logger = wandb_logger

        self.main_scheduler, self.warmup_scheduler = schedulers
        self.main_optimizer = optimizers[0]

        self.PIT_SISNR_mag_loss, self.PIT_SISNR_time_loss = criterions
        
        # weights_path is the same as checkpoint_dir, use to save checkpoints
        self.weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'log', 'pretrain_weights')
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
                if dist.get_rank() == 0:
                    logger.info(e)
            if dist.get_rank() == 0:
                logger.info(f'Loaded model from {checkpoint[0]} .....')
        
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            find_unused_parameters=False
        )
        self.start_epoch = checkpoint[1] + 1

    def _train(self, dataloader, epoch):
        self.model.train()
        tot_loss_freq = [0 for _ in range(self.model_num_stages)]
        tot_loss_time, num_batch = 0, 0
        tot_loss = 0
        if dist.get_rank() == 0:
            pbar = tqdm(total=len(dataloader), unit='bs', colour='YELLOW', dynamic_ncols=True)
        for mixtures, srcs in dataloader:
            nnet_input = mixtures
            nnet_input = functions.apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
            num_batch += 1
            if dist.get_rank() == 0:
                pbar.update(1)
            # Scheduler learning rate for warm-up (Iteration-based update for transformers)
            if epoch == 1: self.warmup_scheduler.step()
            nnet_input = nnet_input.to(self.device)
            self.main_optimizer.zero_grad()
            estims, estims_bn = self.model(nnet_input)
            cur_loss_s_bn = 0
            cur_loss_s_bn = []
            for idx, estims_value in enumerate(estims_bn):
                cur_loss_s_bn.append(self.PIT_SISNR_mag_loss(estims=estims_value, idx=idx, targets=srcs))
                tot_loss_freq[idx] += cur_loss_s_bn[idx].item() / (self.config['model']['num_spks'])
            cur_loss_s = self.PIT_SISNR_time_loss(estims=estims, targets=srcs)
            tot_loss_time += cur_loss_s.item() / self.config['model']['num_spks']
            alpha = 0.4 * 0.8**(1+(epoch-101)//5) if epoch > 100 else 0.4
            cur_loss = (1 - alpha)*cur_loss_s + alpha * sum(cur_loss_s_bn) / len(cur_loss_s_bn)
            cur_loss = cur_loss / self.config['model']['num_spks']
            tot_loss += cur_loss.item()
            cur_loss.backward()
            if self.config['engine']['clip_norm']: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['engine']['clip_norm'])
            self.main_optimizer.step()
            if dist.get_rank() == 0:
                dict_loss = {'Loss': tot_loss / num_batch}
                pbar.set_postfix(dict_loss)
        if dist.get_rank() == 0:
            pbar.close()
        tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)
        return tot_loss_time / num_batch, tot_loss_freq / num_batch, tot_loss / num_batch

    def _validate(self, dataloader, is_test=False):
        self.model.eval()
        tot_loss, num_batch = 0, 0
        if dist.get_rank() == 0:
            pbar = tqdm(total=len(dataloader), unit='bs', colour='RED')
        with torch.inference_mode():
            for mixtures, srcs in dataloader:
                nnet_input = mixtures
                nnet_input = functions.apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
                nnet_input = nnet_input.to(self.device)
                num_batch += 1
                if dist.get_rank() == 0:
                    pbar.update(1)
                estims, estims_bn = self.model(nnet_input)
                cur_loss_s = self.PIT_SISNR_time_loss(estims=estims, targets=srcs, is_test=is_test)
                tot_loss += cur_loss_s.item() / self.config['model']['num_spks']
                if dist.get_rank() == 0:
                    dict_loss = {'Loss': tot_loss / num_batch}
                    pbar.set_postfix(dict_loss)
        if dist.get_rank() == 0:
            pbar.close()
        loss = torch.tensor([tot_loss/num_batch], device=self.device)
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= dist.get_world_size()
        return loss.item()

    def train(self, dataloaders):
        test_loss = self._validate(dataloaders['test'], is_test=True)
        
        if dist.get_rank() == 0:
            logger.info(f'Current learning rate: {self.main_optimizer.param_groups[0]["lr"]}.')
            logger.info(f'[TEST] Loss = {test_loss:.4f} dB')   
        
        if self.start_epoch > 1:
            init_loss = self._validate(dataloaders['val'])
        else:
            init_loss = 1e9

        if dist.get_rank() == 0:
            logger.info(f'[INIT] Epoch {self.start_epoch-1:2d}: Loss = {init_loss:.4f} dB')   
            if self.wandb_logger is not None and init_loss < 1e9:
                self.wandb_logger.log({
                    'Valid Batch Loss': init_loss,
                    'Learning Rate'   : self.main_optimizer.param_groups[0]["lr"]
                }, step=self.start_epoch - 1)                
        
        valid_loss_best = init_loss
        for epoch in range(self.start_epoch, self.config['engine']['max_epoch']):
            
            train_loss_time, train_loss_freq, train_loss = self._train(dataloaders['train'], epoch)
            valid_loss = self._validate(dataloaders['val'])
            if epoch > self.config['engine']['start_scheduling']: self.main_scheduler.step(valid_loss)
            # If CUDA runs out of memory, nothing will be logged.
            if dist.get_rank() == 0:
                logger.info(f'[TRAIN] Epoch {epoch:2d}: Loss_t = {train_loss_time:.4f} dB | Loss_f = {train_loss_freq:.4f} dB | Loss = {train_loss:.4f} dB')
                logger.info(f'[VALID] Epoch {epoch:2d}: Loss = {valid_loss:.4f} dB')
                if self.wandb_logger is not None:
                    self.wandb_logger.log({
                        'Train Loss_T': train_loss_time,
                        'Train Loss_F': train_loss_freq,
                        'Train Loss': train_loss,
                        'Valid Loss': valid_loss,
                        'Learning Rate'   : self.main_optimizer.param_groups[0]["lr"]
                    }, step=epoch)

                    if epoch > self.start_epoch and epoch % 10 == 0:
                        test_loss = self._validate(dataloaders['test'], is_test=True)                            
                        self.wandb_logger.log({
                            'Test Loss': test_loss,
                        }, step=epoch//10)

            if dist.get_rank() == 0 and epoch % self.config['engine']['save_checkpoint_freq'] == 0:
                try:
                    torch.save(
                        {
                            'model_state_dict': self.model.module.state_dict(),
                            'optimizer_state_dict': self.main_optimizer.state_dict(),
                        },
                        os.path.join(self.weights_path, f'{self.prefix}.{epoch}.pth')
                    )
                except Exception as e:
                    logger.debug(e)

            if dist.get_rank() == 0 and valid_loss_best > valid_loss:   
                logger.info(f'Found a better checkpoint at epoch {epoch}.')
                valid_loss_best = valid_loss
                try:
                    torch.save(
                        {
                            'model_state_dict': self.model.module.state_dict(),
                            'optimizer_state_dict': self.main_optimizer.state_dict(),
                        },
                        os.path.join(self.weights_path, f'{self.prefix}.best.pth')
                    )
                except Exception as e:
                    logger.debug(e)

        if dist.get_rank() == 0:
            logger.info(f'Training for {self.config["engine"]["max_epoch"]} epochs done!')