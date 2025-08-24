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

from model import Model
import soc_implement


def apply_cmvn(feats):
    feats = feats - feats.mean(1, keepdims=True)
    feats = feats / feats.std(1, keepdims=True)
    return feats


class Engine(object):
    def __init__(self, prefix, config, checkpoint, gpu_ids, wandb_logger=None): # do not call wandb_logger.finish() !
        model = Model(**config['model'])
        n_params = sum(p.numel() for p in model.parameters())
        n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Total parameters    : {n_params:,}')
        logger.info(f'Trainable parameters: {n_train_params:,}')
        
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
        self.weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log', 'pretrain_weights')
        os.makedirs(self.weights_path, exist_ok=True)
        
        self.model = model.to(self.device)
        self.model_num_stages = self.model.num_stages
        self.start_checkpoint_path = checkpoint[0]
        if checkpoint[0] is not None:
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

    def _inference_sample(self, mixture, sampling_rate):
        self.model.eval()
        model_stride = self.config['model']['module_audio_enc']['stride']
        model_sr = self.config['dataset']['sample_rate']

        if model_sr != sampling_rate:
            mixture = librosa.resample(mixture, orig_sr=sampling_rate, target_sr=model_sr)
        if not isinstance(mixture, torch.Tensor):
            mixture = torch.tensor(mixture, dtype=torch.float32).to(self.device)
        if len(mixture.shape) == 1:
            mixture = mixture[None]
        
        remains = mixture.shape[-1] % model_stride
        if remains != 0:
            padding = model_stride - remains
            mixture_padded = torch.nn.functional.pad(mixture, (0, padding), 'constant', 0)
        else:
            mixture_padded = mixture

        with torch.inference_mode():
            nnet_input = mixture_padded.to(self.device)
            estims, _ = self.model(nnet_input)

            results = []
            for i in range(self.config['model']['num_spks']):
                results.append((estims[i].mean(dim=0)[..., :mixture.shape[-1]], model_sr))
            return results

    def _train(self, dataloader, epoch):
        self.model.train()
        tot_loss_freq = [0 for _ in range(self.model_num_stages)]
        tot_loss_time, num_batch = 0, 0
        tot_loss = 0
        pbar = tqdm(total=len(dataloader), unit='bs', colour='YELLOW', dynamic_ncols=True)
        for mixtures, srcs in dataloader:
            nnet_input = mixtures
            nnet_input = functions.apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
            num_batch += 1
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
            dict_loss = {'Loss': tot_loss / num_batch}
            pbar.set_postfix(dict_loss)
        pbar.close()
        tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)
        return tot_loss_time / num_batch, tot_loss_freq / num_batch, tot_loss / num_batch

    def _finetune(self, dataloader, epoch): # tot_loss_time is tot_loss
        self.model.train()
        tot_loss, num_batch = 0, 0
        pbar = tqdm(total=len(dataloader), unit='bs', colour='YELLOW', dynamic_ncols=True)
        for mixtures, srcs in dataloader:
            nnet_input = mixtures
            nnet_input = functions.apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
            num_batch += 1
            pbar.update(1)
            # Scheduler learning rate for warm-up (Iteration-based update for transformers)
            if epoch == 1 or (self.config['engine']['reset_so'][0] == True): self.warmup_scheduler.step()
            nnet_input = nnet_input.to(self.device)
            self.main_optimizer.zero_grad()
            estims, _= self.model(nnet_input)
            cur_loss_s = self.PIT_SISNR_time_loss(estims=estims, targets=srcs, finetune=True)
            cur_loss = cur_loss_s / self.config['model']['num_spks']
            cur_loss.backward()
            tot_loss += cur_loss_s.item()
            if self.config['engine']['clip_norm']: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['engine']['clip_norm'])
            self.main_optimizer.step()
            dict_loss = {'Loss': tot_loss / num_batch}
            pbar.set_postfix(dict_loss)
        pbar.close()
        return tot_loss / num_batch
    
    def _validate(self, dataloader, finetune=False):
        self.model.eval()
        tot_loss, num_batch = 0, 0
        pbar = tqdm(total=len(dataloader), unit='bs', colour='RED')
        with torch.inference_mode():
            for mixtures, srcs in dataloader:
                nnet_input = mixtures
                nnet_input = functions.apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
                nnet_input = nnet_input.to(self.device)
                num_batch += 1
                pbar.update(1)
                estims, estims_bn = self.model(nnet_input)
                cur_loss_s = self.PIT_SISNR_time_loss(estims=estims, targets=srcs, finetune=finetune)
                tot_loss += cur_loss_s.item() / self.config['model']['num_spks']
                dict_loss = {'Loss': tot_loss / num_batch}
                pbar.set_postfix(dict_loss)
        pbar.close()
        return tot_loss/ num_batch

    def inference_sample(self, filepath, output_dir):
        def l2norm_sq(mat, keepdim=False):
            return (mat * mat).sum(dim=-1, keepdim=keepdim)
        
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        model_stride = self.config['model']['module_audio_enc']['stride']
        model_sr = self.config['dataset']['sample_rate']

        mixture, _ = librosa.load(filepath, sr=model_sr, mono=True)
        mixture = torch.tensor(mixture, dtype=torch.float32).to(self.device)
        results = self._inference_sample(mixture, model_sr)

        stem_name = pathlib.Path(filepath).stem
        sf.write(
            os.path.join(output_dir, stem_name + '-in.wav'), 
            0.9*(mixture/(mixture.abs().max() + 1e-9)).cpu().numpy(), 
            model_sr
        )
        
        for index, (estim, sr) in enumerate(results):
            print(f'SPEAKER {index}:')
            print(f'L2NORM^2: {l2norm_sq(estim.unsqueeze(0)).detach().cpu().numpy()}')
            sf.write(
                os.path.join(output_dir, stem_name + f'-out-{index}.wav'), 
                0.9*(estim/(estim.abs().max() + 1e-9)).cpu().numpy(), 
                sr
            )

    def test(self, dataloader): # Currently only using validate's metric
        if self.start_epoch > 1:
            if self.start_epoch >= self.config['engine']['finetune_epoch']:
                logger.info('Evaluation will be done on fintune mode.')
                loss = self._validate(dataloader, finetune=True)
            else:
                loss = self._validate(dataloader)
        else:
            loss = 1e9
        if loss < 1e9:
            logger.info(f'[TEST] Loss = {loss:.4f}')
        else:
            logger.info('UNK')

    def train(self, dataloaders):
        logger.info(f'Current learning rate: {self.main_optimizer.param_groups[0]["lr"]}.')
        if len(self.gpu_ids) > 1:
            model_instance = self.model.module
        else:
            model_instance = self.model
                        
        if self.start_epoch > 1:
            if self.start_epoch >= self.config['engine']['finetune_epoch']:
                init_loss = self._validate(dataloaders['val'], finetune=True)
            else:
                init_loss = self._validate(dataloaders['val'])
        else:
            init_loss = 1e9

        logger.info(f'[INIT] Epoch {self.start_epoch-1:2d}: Loss = {init_loss:.4f} dB')   
        if self.wandb_logger is not None and init_loss < 1e9:
            self.wandb_logger.log({
                'Valid Batch Loss': init_loss,
                'Learning Rate'   : self.main_optimizer.param_groups[0]["lr"]
            }, step=self.start_epoch - 1)                
        valid_loss_best = init_loss
        
        for epoch in range(self.start_epoch, self.config['engine']['max_epoch']):
            if epoch < self.config['engine']['finetune_epoch']:
                train_loss_time, train_loss_freq, train_loss = self._train(dataloaders['train'], epoch)
                valid_loss = self._validate(dataloaders['val'])
                if epoch > self.config['engine']['start_scheduling']: self.main_scheduler.step(valid_loss)
                # If CUDA runs out of memory, nothing will be logged.
                logger.info(f'[TRAIN] Epoch {epoch:2d}: Loss_t = {train_loss_time:.4f} dB | Loss_f = {train_loss_freq:.4f} dB | Loss = {train_loss:.4f} dB')
                logger.info(f'[VALID] Epoch {epoch:2d}: Loss = {valid_loss:.4f} dB')
                if self.wandb_logger is not None:
                    self.wandb_logger.log({
                        'Train Batch Loss_T': train_loss_time,
                        'Train Batch Loss_F': train_loss_freq,
                        'Train Batch Loss': train_loss,
                        'Valid Batch Loss': valid_loss,
                        'Learning Rate'   : self.main_optimizer.param_groups[0]["lr"]
                    }, step=epoch)
            else:
                if epoch > self.config['engine']['finetune_epoch']:
                    self.config['engine']['reset_so'][0] = False
        
                if self.config['engine']['reset_so'][0] == True:
                    main_optimizer_name = self.config['optimizer']['name'][0]
                    self.config['optimizer'][main_optimizer_name]['lr'] = self.config['engine']['reset_so'][1]
                    self.config['scheduler']['WarmupConstantSchedule']['warmup_steps'] = 1
                    optimizers = soc_implement.OptimizerFactory(self.config['optimizer'], model_instance.parameters()).get_optimizers()
                    schedulers = soc_implement.SchedulerFactory(self.config['scheduler'], optimizers).get_schedulers()
                    self.main_scheduler, self.warmup_scheduler = schedulers
                    self.main_optimizer = optimizers[0]
                    logger.info('Learning rate reset.')

                train_loss = self._finetune(dataloaders['train'], epoch)
                valid_loss = self._validate(dataloaders['val'], finetune=True)
                if epoch > self.config['engine']['start_scheduling']: self.main_scheduler.step(valid_loss)
                # If CUDA runs out of memory, nothing will be logged.
                logger.info(f'[FINETUNE] Epoch {epoch:2d}: Loss = {train_loss:.4f}')
                logger.info(f'[VALID   ] Epoch {epoch:2d}: Loss = {valid_loss:.4f}')
                if self.wandb_logger is not None:
                    self.wandb_logger.log({
                        'Train Batch Loss': train_loss,
                        'Valid Batch Loss': valid_loss,
                        'Learning Rate'   : self.main_optimizer.param_groups[0]["lr"]
                    }, step=epoch)

                self.config['engine']['reset_so'][0] = False # MUST SET TO FALSE HERE

            if epoch % self.config['engine']['save_checkpoint_freq'] == 0:
                try:
                    torch.save(
                        {
                            'model_state_dict': model_instance.state_dict(),
                            'optimizer_state_dict': self.main_optimizer.state_dict(),
                        },
                        os.path.join(self.weights_path, f'{self.prefix}.{epoch}.pth')
                    )
                except Exception as e:
                    logger.debug(e)

            if valid_loss_best > valid_loss:   
                logger.info(f'Found a better checkpoint at epoch {epoch}.')
                valid_loss_best = valid_loss
                try:
                    torch.save(
                        {
                            'model_state_dict': model_instance.state_dict(),
                            'optimizer_state_dict': self.main_optimizer.state_dict(),
                        },
                        os.path.join(self.weights_path, f'{self.prefix}.best.pth')
                    )
                except Exception as e:
                    logger.debug(e)

        logger.info(f'Training for {self.config["engine"]["max_epoch"]} epochs done!')

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