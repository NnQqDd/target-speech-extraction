from collections import defaultdict
import csv
import random
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io
import torchvision.transforms.v2 as v2
import torchaudio.functional as F


def get_speech_metadata(config):
    metadata_path = config['dataset']['speech_metadata_path']
    speech_metadata = defaultdict(lambda: defaultdict(list))

    with open(metadata_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = {"file_path", "url", "speaker_id", "partition"}
        if not required_cols.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        for row in reader:
            file_path = row["file_path"]
            speaker_id = row["speaker_id"]
            partition = row["partition"]

            if file_path and speaker_id and partition:
                speech_metadata[partition][speaker_id].append(file_path)

    return speech_metadata


def extract_audio_and_frames(video_path: str, target_sr=16000, target_fps=16, resnet18_norm=True, array=True):
    # Use VideoReader for streaming (memory-efficient)
    video_reader = io.VideoReader(video_path, "video")
    metadata = video_reader.get_metadata()
    duration = metadata['video']['duration'][0]  # in seconds
    original_fps = metadata['video']['fps'][0]
    
    # Calculate number of frames at target FPS
    num_frames = int(duration * target_fps)
    video_frames = []
    
    for i in range(num_frames):
        # Target PTS (timestamp) for this frame
        target_pts = i / target_fps
        video_reader.seek(target_pts)
        try:
            frame = next(video_reader)['data']  # (C, H, W) uint8 tensor
            video_frames.append(frame)
        except StopIteration:
            break  # In case of edge issues at end
    
    if video_frames:
        video_frames = torch.stack(video_frames)  # (T, C, H, W)
    else:
        raise ValueError("No video frames extracted.")
    
    # Convert to float [0,1]
    video_frames = video_frames.float() / 255.0
    
    # Apply transforms: Resize, CenterCrop, Normalize (using v2 for batched tensor support)
    if resnet18_norm:
        transform = v2.Compose([
            v2.Resize(240),
            v2.CenterCrop(224),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        video_frames = transform(video_frames)  # (T, C, 224, 224) normalized
    
    # Audio handling (unchanged, but using a separate reader for audio)
    audio_reader = io.VideoReader(video_path, "audio")
    audio_metadata = audio_reader.get_metadata()
    original_sr = audio_metadata['audio']['fps'][0] if 'audio' in audio_metadata else None
    if original_sr is None:
        raise ValueError("Audio sample rate not found in metadata.")
    
    # Read all audio frames
    audio_frames = torch.cat([frame['data'] for frame in audio_reader], dim=1)  # (channels, samples)
    
    if original_sr != target_sr:
        audio_waveform = F.resample(audio_frames, orig_freq=original_sr, new_freq=target_sr)
    else:
        audio_waveform = audio_frames

    audio_waveform = audio_waveform.mean(dim=0)
    
    if not array:
        return audio_waveform, video_frames
    return audio_waveform.numpy(), video_frames.numpy()


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


class MyDataset(Dataset):
    def __init__(
            self, 
            speech_metadatas, 
            dataset_size, 
            wave_length, 
            sample_rate,  
            fps,
            n_spks_probs,
            augmentation,
        ):
        assert wave_length % sample_rate == 0

        self.speech_metadatas = speech_metadatas
        
        self.dataset_size = dataset_size
        self.wave_length = wave_length
        self.sample_rate = sample_rate
        self.video_length = fps*(wave_length//sample_rate)
        self.fps = fps

        assert self.wave_length % self.video_length == 0
        self.fts = self.wave_length//self.video_length # frame to sample 

        assert len(n_spks_probs) >= 1
        assert n_spks_probs[0] == 0
        self.n_spks_chooser = {
            "choices": list(range(len(n_spks_probs))),
            "weights": n_spks_probs
        }

        self.augmentation = augmentation

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, _):
        n_speakers = random.choices(
            self.n_spks_chooser["choices"], 
            weights=self.n_spks_chooser["weights"], 
            k=1
        )[0]
        speech_dataset = self.speech_metadatas
        
        speakers = random.sample(list(speech_dataset.keys()), k=n_speakers)
        waveforms = []
        for i in range(n_speakers - 1):
            speaker = speakers[i]
            src_audio_path = random.choice(speech_dataset[speaker])
            waveform, _ = librosa.load(src_audio_path, sr=self.sample_rate, mono=True)
            
            if len(waveform) < self.wave_length:
                pad_width = self.wave_length - len(waveform)
                pad_start = random.randint(0, pad_width)
                pad_end = pad_width - pad_start
                waveform = np.pad(waveform, (pad_start, pad_end), mode='constant', constant_values=0)
            else:
                waveform = random_select_segment(waveform, min(len(waveform), self.wave_length))
            
            waveforms.append(waveform)

        target_wave, video = extract_audio_and_frames(
            src_audio_path, sample_rate=self.sample_rate, fps=self.fps
        )
        
        if len(video) < self.video_length:
            video = np.pad(
                video,
                pad_width=((0, self.video_length - len(video), (0, 0), (0, 0), (0, 0))),
                mode='edge'
            )
            target_wave = target_wave[:self.wave_length]
            target_wave = np.pad(
                target_wave, (0, self.wave_length-len(target_wave)), mode='constant', constant_values=0
            )
        else:
            start = random.randint(0, len(video) - self.video_length)
            end = start + self.video_length
            video = video[start:end, ...]
            target_wave = target_wave[start*self.fts:end*self.fts]
            target_wave = np.pad(
                target_wave, (0, self.wave_length-len(target_wave)), mode='constant', constant_values=0
            )
        
        waveforms.append(target_wave)
        do_augment = random.rand() > self.augmentation['probability']
        if do_augment:
            mix_wave = np.zeros(self.wave_length, dtype=float)
            for i in range(len(waveforms)):
                k = random.uniform(self.augmentation['volume']['lowest'], self.augmentation['volume']['highest'])
                waveforms[i] *= waveforms[i]
                mix_wave += waveforms[i]
        else:
            mix_wave = np.sum(np.stack(waveforms, axis=0), axis=0)
        mix_wave = peak_normalize(mix_wave, optional=True)
        
        return mix_wave, video, waveforms[-1] 
    
# * iterable unpacking, ** dictionary unpacking
def collate(batch):
    mixes, videos, targets = zip(*batch)
    mixes = [torch.tensor(m, dtype=torch.float32) for m in mixes]
    mixes = torch.stack(mixes, dim=0)
    videos = [torch.tensor(v, dtype=torch.float32) for v in videos]
    videos = torch.stack(videos, dim=0)
    targets = [torch.tensor(t, dtype=torch.float32) for t in targets]
    targets = torch.stack(targets, dim=0)
    return mixes, videos, targets


def get_dataloaders(speech_metadatas, ds_config, dl_config):
    datasets = dict()
    datasets['train'] = MyDataset(
        speech_metadatas['train'], 
        dataset_size=ds_config['size']['train'], 
        wave_length=ds_config['wave_length'], 
        sample_rate=ds_config['sample_rate'],
        fps = ds_config['fps'],
        n_spks_probs=ds_config['n_spks_probs'],
        augmentation=ds_config['augmentation']
    )

    datasets['valid'] = MyDataset(
        speech_metadatas['valid'], 
        dataset_size=ds_config['size']['valid'], 
        wave_length=ds_config['wave_length'], 
        sample_rate=ds_config['sample_rate'],
        fps = ds_config['fps'],
        n_spks_probs=ds_config['n_spks_probs'],
        augmentation=ds_config['augmentation']
    )

    dataloaders = dict()
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_size=dl_config['batch_size_per'],
        num_workers=dl_config['n_workers'],
        shuffle=False,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate
    )

    dataloaders['valid'] = DataLoader(
        datasets['valid'],
        batch_size=dl_config['dataloader']['batch_size_per'],
        num_workers=dl_config['dataloader']['n_workers'],
        shuffle=False,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate
    )

    return dataloaders