import torch
from math import ceil
from itertools import permutations
from torchaudio.transforms import MelScale
from dataclasses import dataclass, field, fields
from typing import List, Type, Any, Callable, Optional, Union


def l2norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)

def l2norm_sq(mat, keepdim=False):
    return (mat * mat).sum(dim=-1, keepdim=keepdim)

def l1norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim, p=1)

@dataclass(slots=True)
class STFTBase(torch.nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it 
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """
    device: torch.device
    frame_length: int
    frame_shift: int
    window: str
    K: torch.nn.Parameter = field(init=False)
    num_bins: int = field(init=False)

    def __post_init__(self):
        super(STFTBase, self).__init__()  # Initialize the torch.nn.Module base class
        K = self._init_kernel(self.frame_length, self.frame_shift)
        self.K = torch.nn.Parameter(K, requires_grad=False).to(self.device)
        self.num_bins = self.K.shape[0] // 2
    
    def _init_kernel(self, frame_len, frame_hop):
        # FFT points
        N = frame_len
        # window
        if self.window == 'hann':
            W = torch.hann_window(frame_len)
        if N//4 == frame_hop:
            const = (2/3)**0.5       
            W = const*W
        elif N//2 == frame_hop:
            W = W**0.5
        S = 0.5 * (N * N / frame_hop)**0.5
        
        # Updated FFT calculation for efficiency
        K = torch.fft.rfft(torch.eye(N) / S, dim=1)[:frame_len]
        K = torch.stack((torch.real(K), torch.imag(K)), dim=2)
        K = torch.transpose(K, 0, 2) * W # 2 x N/2+1 x F
        K = torch.reshape(K, (N + 2, 1, frame_len)) # N+2 x 1 x F
        return K

    def extra_repr(self):
        return (f"window={self.window}, stride={self.frame_shift}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}")


@dataclass(slots=True)
class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def forward(self, x, cplx=False):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, N x C x S or N x S
        return
            m: magnitude, N x C x F x T or N x F x T
            p: phase, N x C x F x T or N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__name__, x.dim()))
        # if N x S, reshape N x 1 x S
        N_frame = ceil(x.shape[-1] / self.frame_shift)
        len_padded = N_frame * self.frame_shift
        if x.dim() == 2:
            
            x = torch.cat((x, torch.zeros(x.shape[0], len_padded-x.shape[-1], device=x.device)), dim=-1)
            x = torch.unsqueeze(x, 1)
            # N x 2F x T
            c = torch.nn.functional.conv1d(x, self.K, stride=self.frame_shift, padding=0)
            # N x F x T
            r, i = torch.chunk(c, 2, dim=1)
        else:        
            x = torch.cat((x, torch.zeros(x.shape[0], x.shape[1], len_padded-x.shape[-1])), dim=-1)
            N, C, S = x.shape
            x = x.reshape(N * C, 1, S)
            # NC x 2F x T
            c = torch.nn.functional.conv1d(x, self.K, stride=self.frame_shift, padding=0)
            # N x C x 2F x T
            c = c.reshape(N, C, -1, c.shape[-1])
            # N x C x F x T
            r, i = torch.chunk(c, 2, dim=2)

        if cplx:
            return r, i
        m = (r**2 + i**2 + 1.0e-10)**0.5
        p = torch.atan2(i, r)
        return m, p


@dataclass(slots=True)
class PIT_SISNR_mag:
    device: torch.device
    frame_length: int
    frame_shift: int
    window: str
    num_stages: int
    num_spks: int
    mel_opt: bool
    
    stft: List[Any] = field(init=False)
    mel_fb: Callable[[torch.Tensor], torch.Tensor] = field(init=False)
    
    def __post_init__(self):
        # STFT is expected to be a callable object like your STFT wrapper
        self.stft = [STFT(self.device, self.frame_length, self.frame_shift, self.window) for _ in range(self.num_stages)]
        self.mel_fb = MelScale(n_mels=80, sample_rate=16000, n_stft=int(self.frame_length / 2) + 1).to(self.device) if self.mel_opt else (lambda x: x)

    def __repr__(self):
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]

        stft_repr = f"stft = [STFT instance for {len(self.stft)} layers]"
        mel_fb_repr = "mel_fb = MelScale" if self.mel_opt else "mel_fb=Identity"
        post_init_reprs = [stft_repr, mel_fb_repr]

        return f"<{class_name}({', '.join(field_strs + post_init_reprs)})>"
    
    def __call__(self, **kwargs):
        estims = kwargs['estims']   # expected: indexable by speaker: estims[s]
        idx = kwargs['idx']        # which STFT stage to use
        targets = [t.to(self.device) for t in kwargs["targets"]]

        def _freq_SDR_score(prd, tar, eps=1e-10):
            prd_zm = prd - torch.mean(prd, dim=-1, keepdim=True)
            tar_zm = tar - torch.mean(tar, dim=-1, keepdim=True)
            scale = torch.sum(prd_zm * tar_zm, dim=-1, keepdim=True) / (l2norm(tar_zm, keepdim=True)**2 + eps)
            tar_zm = torch.clamp(scale, min=1e-2) * tar_zm

            prd_spec = self.stft[idx](prd_zm.to(self.device))[0]
            tar_spec = self.stft[idx](tar_zm.to(self.device))[0]

            if self.mel_opt:
                prd_spec = self.mel_fb(prd_spec)
                tar_spec = self.mel_fb(tar_spec)

            return 20 * torch.log10(eps + l2norm(l2norm(tar_spec)) / (l2norm(l2norm(prd_spec - tar_spec)) + eps))    

        losses = []
        wave_len = estims[0].shape[-1]//2
        for s in range(self.num_spks): # self.num_spks is based on the original purpose of sepreformer
            if s == 0:
                prd_src = estims[s][..., :wave_len]
                prd_ref = estims[s][..., wave_len:]
                tar_src = targets[s][..., :wave_len]
                tar_ref = targets[s][..., wave_len:]
                tse_loss = -_freq_SDR_score(prd_src, tar_src)
                ref_loss = -_freq_SDR_score(prd_ref, tar_ref)
                losses.extend([tse_loss, ref_loss])    

            else:
                prd_int = estims[s][:wave_len]
                tar_int = targets[s][:wave_len]
                loss = -_freq_SDR_score(prd_int, tar_int)
                losses.append(loss) 
            
        final_loss = (losses[0]*(len(losses) - 1) + sum(losses[1:]))/(2*len(losses) - 1)
        return torch.sum(final_loss) / estims[0].shape[0]  # mean over batch


@dataclass(slots=True)
class PIT_SISNR_time:
    device: torch.device
    num_spks: int

    def __repr__(self):
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]
        return f"<{class_name}({', '.join(field_strs)})>"
    
    def __call__(self, **kwargs):
        estims = kwargs['estims']
        targets = [t.to(self.device) for t in kwargs["targets"]]
        is_test = kwargs.get('is_test', False)

        def _SDR_score(prd, tar, eps=1e-10):
            prd_zm = prd - torch.mean(input=prd, dim=-1, keepdim=True)
            tar_zm = tar - torch.mean(input=tar, dim=-1, keepdim=True)
            tar_zm_scale = tar_zm * torch.sum(prd_zm * tar_zm, dim=-1, keepdim=True) / (l2norm_sq(tar_zm, keepdim=True) + eps)
            A = l2norm_sq(tar_zm_scale)
            B = l2norm_sq(prd_zm - tar_zm_scale)
            return 10 * torch.log10(eps + A / (B + eps)) 

        losses = []
        wave_len = estims[0].shape[-1]//2
        for s in range(self.num_spks): # self.num_spks is based on the original purpose of sepreformer
            if s == 0:
                prd_src = estims[s][..., :wave_len]
                prd_ref = estims[s][..., wave_len:]
                tar_src = targets[s][..., :wave_len]
                tar_ref = targets[s][..., wave_len:]
                tse_loss = torch.clamp(-_SDR_score(prd_src, tar_src), min=-32)
                ref_loss = torch.clamp(-_SDR_score(prd_ref, tar_ref), min=-32)
                losses.extend([tse_loss, ref_loss])    
            
            elif not is_test:
                prd_int = estims[s][..., :wave_len]
                tar_int = targets[s][..., :wave_len]
                loss = torch.clamp(-_SDR_score(prd_int, tar_int), min=-32)
                losses.append(loss) 
        
        if is_test:
            final_loss = losses[0]
        else:
            final_loss = (losses[0]*(len(losses) - 1) + sum(losses[1:]))/(2*len(losses) - 1)
     
        return torch.sum(final_loss) / estims[0].shape[0]  # mean over batch


