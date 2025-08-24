import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import librosa # Fully CPU
import torch
from torch.utils.data import Dataset, DataLoader


def get_dataloaders(engine_mode, config, n_gpus):    # return test, val, train
    if config['model']['num_spks'] != 2:
        raise NotImplementedError("Please be patient!")

    ds_config = config['dataset']
    loader_config = config['dataloader']
    partitions = ['test'] if 'test' in engine_mode else ['train', 'val']
    speech_dataset_probs, speech_metadatas = [], []
    for prob, path in ds_config['speech_metadata_paths']:
        speech_metadatas.append(pd.read_csv(path))
        speech_dataset_probs.append(prob)
    noise_metadatas = [pd.read_csv(path) for path in ds_config['noise_metadata_paths']]

    dataloaders = {}
    for partition in partitions:
        dataset = MyDataset(
            partition, 
            dataset_size=ds_config['dataset_size'][partition],
            wave_len=ds_config['wave_len'],
            sample_rate=ds_config['sample_rate'],
            lowest_snr=ds_config['lowest_snr'],
            highest_snr=ds_config['highest_snr'],
            speech_dataset_probs=speech_dataset_probs,
            speech_metadatas=[df[df['partition'] == partition] for df in speech_metadatas],
            noise_metadatas=[df[df['partition'] == partition] for df in noise_metadatas]
        )
        if partition == 'val' or partition == 'test':
            batch_size = 2*loader_config['batch_size_per_gpu']*n_gpus
        elif partition == 'train':
            batch_size = loader_config['batch_size_per_gpu']*n_gpus
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            # worker_init_fn=worker_init_fn,
            num_workers = loader_config['num_workers'],
            collate_fn = _collate
        )
        dataloaders[partition] = dataloader
    return dataloaders


def _collate(samples):
    mixtures = []
    source_1s = []
    source_2s = []
    for sample in samples:
        mixtures.append(torch.tensor(sample['mix'], dtype=torch.float32))
        source_1s.append(torch.tensor(sample['src'][0], dtype=torch.float32))
        source_2s.append(torch.tensor(sample['src'][1], dtype=torch.float32))
    mixtures = torch.stack(mixtures, dim=0)
    source_1s = torch.stack(source_1s, dim=0)
    source_2s = torch.stack(source_2s, dim=0)
    return mixtures, [source_1s, source_2s]


class MyDataset(Dataset): # Currently only support Vietnamese and English
    def __init__(self, partition, **kwargs):
        self.dataset_size = kwargs['dataset_size'] 
        self.partition = partition # Currently not being used
        self.wave_len = kwargs['wave_len']
        self.sample_rate = kwargs['sample_rate']  

        self.lowest_snr = kwargs['lowest_snr']
        self.highest_snr = kwargs['highest_snr']
        self.speech_dataset_probs = kwargs['speech_dataset_probs']
        
        self.speaker_to_paths = []
        for df in kwargs['speech_metadatas']:
            ds_speaker_to_paths = defaultdict(list)
            for _, row in df.iterrows():
                ds_speaker_to_paths[row['speaker_id']].append(row['filepath'])
            self.speaker_to_paths.append(ds_speaker_to_paths)
        
        self.speaker_ids = []
        for idx, df in enumerate(kwargs['speech_metadatas']):
            self.speaker_ids.append(df['speaker_id'].unique().tolist())

        self.noise_paths = [] 
        for df in kwargs['noise_metadatas']:
            self.noise_paths.extend(df['filepath'])

        if self.partition == 'test':
            random.seed(42)
            np.random.seed(42)

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

        def add_noise(
            speech_wave: np.ndarray, # 1-D, same sample rate as noise wave
            noise_wave: np.ndarray,
            lowest_snr: float,
            highest_snr: float
        ) -> np.ndarray:
            
            speech_len = speech_wave.shape[0]
            noise_len = noise_wave.shape[0]
            if noise_len >= speech_len:
                noise_wave = random_select_segment(noise_wave, speech_len)
            else:
                reps = int(np.ceil(speech_len/ noise_len))
                noise_wave = np.tile(noise_wave, reps)[:speech_len]

            snr_db = random.uniform(lowest_snr, highest_snr)
            rms_s = rms(speech_wave)
            rms_n = rms(noise_wave)

            # 20*log10(rms_s / (gain * rms_n)) = snr_db
            # gain = (rms_s / rms_n) * 10^(-snr_db/20)
            gain = (rms_s / (rms_n + 1e-9)) * (10 ** (-snr_db / 20))
            noise_scaled = noise_wave * gain

            mixed = speech_wave + noise_scaled
            return peak_normalize(mixed)

        choice = random.choices(range(len(self.speech_dataset_probs)), weights=self.speech_dataset_probs, k=1)[0]
        speaker_ids = self.speaker_ids[choice]
        speaker_to_paths = self.speaker_to_paths[choice]

        i, j = random.sample(range(len(speaker_ids)), 2)
        spk1, spk2 = speaker_ids[i], speaker_ids[j]
        ref_audio_path = random.choice(speaker_to_paths[spk1])
        tar_audio_path = random.choice(speaker_to_paths[spk1])
        int_audio_path = random.choice(speaker_to_paths[spk2])
        noise_path = random.choice(self.noise_paths)

        # wave_1, sr = sf.read('audio.wav', always_2d=True)
        ref_wave, _ = librosa.load(ref_audio_path, sr=self.sample_rate, mono=True)
        tar_wave, _ = librosa.load(tar_audio_path, sr=self.sample_rate, mono=True)
        int_wave, _ = librosa.load(int_audio_path, sr=self.sample_rate, mono=True)
        noise, _ = librosa.load(noise_path, sr=self.sample_rate, mono=True)

        ref_wave = random_select_segment(ref_wave, min(len(ref_wave), self.wave_len))
        tar_wave = random_select_segment(tar_wave, min(len(tar_wave), self.wave_len))
        int_wave = random_select_segment(int_wave, min(len(int_wave), self.wave_len))

        ref_wave = np.pad(ref_wave, (self.wave_len - len(ref_wave), 0), mode='constant', constant_values=0)
        tar_wave = np.pad(tar_wave, (0, self.wave_len - len(tar_wave)), mode='constant', constant_values=0)
        int_wave = np.pad(int_wave, (0, self.wave_len - len(int_wave)), mode='constant', constant_values=0)
        
        mixed_wave = peak_normalize(tar_wave + int_wave)
        mixed_wave = add_noise(mixed_wave, noise, self.lowest_snr, self.highest_snr)

        tar_wave = np.concatenate([tar_wave, ref_wave])
        int_wave = np.pad(int_wave, (0, self.wave_len), mode='constant', constant_values=0)
        mixed_wave = np.concatenate([mixed_wave, ref_wave])
        return {'mix': mixed_wave, 'src': [tar_wave, int_wave]}