import os
import random
from typing import List
from collections import defaultdict
import numpy as np
import pandas as pd
import librosa # Fully CPU
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader


def get_dataloaders(config, distributed=True):    # return test, val, train
    ds_config = config['dataset']
    loader_config = config['dataloader']
    partitions = ['train', 'val', 'test']
    dataset_probs, dataset_metadatas = [], []
    for prob, path in ds_config['speech_metadata_paths']:
        dataset_metadatas.append(pd.read_csv(path))
        dataset_probs.append(prob)
    
    dataloaders = {}
    for partition in partitions:
        dataset = MyDataset(
            partition, # Currently not being used
            dataset_size=ds_config['dataset_size'][partition],
            wave_len=ds_config['wave_len'],
            sample_rate=ds_config['sample_rate'],
            dataset_probs=dataset_probs,
            dataset_metadatas=[df[df['partition'] == partition] for df in dataset_metadatas],
        )
        if partition == 'val' or partition == 'test':
            batch_size = 2*config['dataloader']['batch_size_per_gpu']
        elif partition == 'train':
            batch_size = config['dataloader']['batch_size_per_gpu']

        if distributed:
            sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        else:
            sampler = None

        dataloader = DataLoader(
            dataset = dataset,
            sampler=sampler,
            batch_size = batch_size,
            # worker_init_fn=worker_init_fn,
            num_workers = loader_config['num_workers'],
            collate_fn = _collate
        )
        dataloaders[partition] = dataloader
    return dataloaders


def _collate(samples):
    mixtures = []
    sources = [[] for _ in range(len(samples[0]['src']))] # List of torch.tensor of shape (T,)
    for sample in samples:
        mixtures.append(torch.tensor(sample['mix'], dtype=torch.float32))
        for index, source in enumerate(sample['src']):
            sources[index].append(torch.tensor(source, dtype=torch.float32))
    mixtures = torch.stack(mixtures, dim=0)
    for index in range(len(sources)):
        sources[index] = torch.stack(sources[index], dim=0)
    return mixtures, sources


class MyDataset(Dataset): # Currently only support Vietnamese and English
    def __init__(self, partition, **kwargs):
        self.dataset_size = kwargs['dataset_size'] 
        self.partition = partition
        self.wave_len = kwargs['wave_len']
        self.sample_rate = kwargs['sample_rate']  

        self.dataset_probs = kwargs['dataset_probs']
        
        self.speaker_to_paths = []
        for df in kwargs['dataset_metadatas']:
            ds_speaker_to_paths = defaultdict(list)
            for _, row in df.iterrows():
                ds_speaker_to_paths[row['speaker_id']].append(row['filepath'])
            self.speaker_to_paths.append(ds_speaker_to_paths)
        
        self.speaker_ids = []
        for idx, df in enumerate(kwargs['dataset_metadatas']):
            self.speaker_ids.append(df['speaker_id'].unique().tolist())


    def __len__(self):
        return self.dataset_size
            
    def __getitem__(self, _):
        def rms(wave: np.ndarray, eps: float = 1e-9) -> np.ndarray:
            return np.sqrt(np.mean(wave ** 2, axis=0, keepdims=True) + eps)

        def random_select_segment(wave, length):
            if random.random() <= 0.5:
                start = random.randint(0, wave.shape[0] - length)
                wave = wave[start:start + length]
            else:
                end = random.randint(length, wave.shape[0])
                wave = wave[end-length:end]
            return wave

        def peak_normalize(wave: np.ndarray, optional=True):
            peak = np.max(np.abs(wave))
            if peak > 1 or not optional:
                wave /= peak
            return wave

        choice = random.choices(range(len(self.dataset_probs)), weights=self.dataset_probs, k=1)[0]
        speaker_ids = self.speaker_ids[choice]
        speaker_to_paths = self.speaker_to_paths[choice]

        i, j = random.sample(range(len(speaker_ids)), 2)
        spk1, spk2 = speaker_ids[i], speaker_ids[j]
        ref_audio_path = random.choice(speaker_to_paths[spk1])
        tar_audio_path = random.choice(speaker_to_paths[spk1])
        int_audio_path = random.choice(speaker_to_paths[spk2])

        ref_wave, _ = librosa.load(ref_audio_path, sr=self.sample_rate, mono=True)
        tar_wave, _ = librosa.load(tar_audio_path, sr=self.sample_rate, mono=True)
        int_wave, _ = librosa.load(int_audio_path, sr=self.sample_rate, mono=True)

        ref_wave = random_select_segment(ref_wave, min(len(ref_wave), self.wave_len))
        tar_wave = random_select_segment(tar_wave, min(len(tar_wave), self.wave_len))
        int_wave = random_select_segment(int_wave, min(len(int_wave), self.wave_len))

        ref_wave = np.pad(ref_wave, (self.wave_len - len(ref_wave), 0), mode='constant', constant_values=0)
        tar_wave = np.pad(tar_wave, (0, self.wave_len - len(tar_wave)), mode='constant', constant_values=0)
        int_wave = np.pad(int_wave, (0, self.wave_len - len(int_wave)), mode='constant', constant_values=0)
        
        mixed_wave = peak_normalize(tar_wave + int_wave)
        
        ref_wave += 1 # important
        cat_wave = np.concatenate([tar_wave, ref_wave])
        int_wave = np.pad(int_wave, (0, self.wave_len), mode='constant', constant_values=0)
        mixed_wave = np.concatenate([mixed_wave, ref_wave])

        return {'mix': mixed_wave, 'src': [cat_wave, int_wave]}
