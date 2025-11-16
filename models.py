from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from torchvision import models, transforms


class VideoResNetEmbedder(torch.nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = torch.nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.emb_size = 512

    def forward(self, x: torch.Tensor):
        """
        x: NORMALIZED image of shape (B, T, 3, H, W)
        returns: (B, T, 512)
        """
        if x.dim() != 5:
            raise ValueError("Input must be 5D tensor (B, T, H, W, 3) or (B, T, 3, H, W). Got shape: " + str(x.shape))

        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)  # (B*T, 3, H, W)
        feats = self.backbone(x)  # (B*T, 512)
        feats = feats.reshape(B, T, -1)
        assert feats.shape[-1] == self.emb_size

        return feats

    
class FiLMFusion(torch.nn.Module):
    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        self.channels = channels
        self.proj = torch.nn.Conv1d(in_channels=channels, out_channels=2 * channels,
                                    kernel_size=1, bias=bias)

        # Initialize to be identity-ish: gamma ≈ 1, beta ≈ 0
        torch.nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            with torch.no_grad():
                self.proj.bias[:channels].fill_(1.0)
                self.proj.bias[channels:].zero_()

    def forward(self, audio: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        audio:   (B, C, T)
        context: (B, C, t)
        returns: (B, C, T)
        """
        assert audio.dim() == 3 and context.dim() == 3, \
            "audio and context must be (B, C, T)"
        assert audio.shape[0] == context.shape[0] and audio.shape[1] % context.shape[1] == 0, \
            "batch and channel dims must match between audio and context"
        _, C, T = audio.shape
        _, C, t = context.shape
        assert C == self.channels, f"audio channels ({C}) must match module channels ({self.channels})"

        cond = self.proj(context) # (B, 2*C, t)
        gamma, beta = cond.split(C, dim=1) # each (B, C, t)
        gamma = gamma.repeat_interleave(T//t, dim=-1) # (B, C, T)
        beta = beta.repeat_interleave(T//t, dim=-1) # (B, C, T)
        out = gamma * audio + beta
        return out
    

class UNetAVTSE(torch.nn.Module):
    def __init__(self,
                 visual_embedder: torch.nn.Module,
                 encoder_blocks: Tuple[List[torch.nn.Module], List[torch.nn.Module]], # Audio -> Visual
                 fusion_blocks: List[torch.nn.Module],
                 decoder_blocks: List[torch.nn.Module]):
        super().__init__()
        # R -> R down blocks, R up blocks, R + 1 encoder blocks, R + 1 fusion blocks, R decoder blocks.
        assert(len(encoder_blocks)[0] - 1 == len(decoder_blocks) == len(fusion_blocks) - 1)
        assert(len(encoder_blocks)[0] == len(encoder_blocks)[1])
        for idx in range(1, len(encoder_blocks[0])):
            assert encoder_blocks[0][idx].channels == encoder_blocks[0][idx - 1].channels
            assert encoder_blocks[1][idx].channels == encoder_blocks[1][idx - 1].channels
        for idx in range(1, len(fusion_blocks)):
            assert fusion_blocks[idx].channels == fusion_blocks[idx - 1].channels
        for idx in range(1, len(decoder_blocks)):
            assert decoder_blocks[idx].channels == decoder_blocks[idx - 1].channels

        self.video_embedder = visual_embedder
        self.encoder_blocks = encoder_blocks
        self.fusion_blocks = fusion_blocks
        self.decoder_blocks = decoder_blocks

        self.enc_channels = encoder_blocks[0][0].channels
        self.dec_channels = decoder_blocks[0].channels

        self.audio_conv1x1 = torch.nn.Conv1d(1, self.enc_channels, kernel_size=1)
        self.visual_conv1x1 = torch.nn.Conv1d(visual_embedder.emb_size, self.enc_channels, kernel_size=1)
        self.concat_conv1x1s = [torch.nn.Conv1d(2*self.enc_channels, self.dec_channels) for i in range(len(decoder_blocks))]
        self.output_conv1x1 = torch.nn.Conv1d(self.dec_channels, 1, kernel_size=1)
        self.fusion = FiLMFusion(self.enc_channels)

        self.gelu = torch.nn.GELU() # The only activation

    def _encode(self, x):
      return x

    def _decode(self, x, int_feats):
      return x

    def forward(self, audio: torch.Tensor, visual: torch.Tensor, int_feats=False) -> torch.Tensor:
      '''
      audio : (B, T)
      visual: (B, t, 3, H, W)
      '''
      assert audio.shape[1] % visual.shape[1] == 0
      assert not int_feats, 'int_feats=True is currently not supported.'

      audio = audio.unsqueeze(1) # (B, 1, T)
      audio = self.audio_conv1x1(audio) # (B, C, T)
      audio = self.gelu(audio)
      audio = self._encode(audio) # (B, C, T)

      visual = self.video_embedder(visual).permute(0, 2, 1) # (B, E, t)
      visual = self.visual_conv1x1(visual) # (B, C, t)
      visual = self.gelu(visual)
      visual = self._encode(visual) # (B, C, t)

      audio = self.fusion(audio, visual) # (B, C, T)
      audio = self.output_conv1(audio) # (B, 1, T)
      return audio.squeeze(1) # (B, T)
    

class MinimalAVTSE(torch.nn.Module):
    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels

        self.visual_embedder = VideoResNetEmbedder()
        self.audio_conv1x1 = torch.nn.Conv1d(1, self.enc_channels, kernel_size=1)
        self.visual_conv1x1 = torch.nn.Conv1d(self.visual_embedder.emb_size, self.enc_channels, kernel_size=1)
        self.GELU = torch.nn.GELU()
        self.fusion = FiLMFusion(self.enc_channels)
        self.output_conv1x1 = torch.nn.Conv1d(self.enc_channels, 1, kernel_size=1)

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        """
        audio : (B, T)
        visual: (B, t, 3, H, W)
        """
        audio = audio.unsqueeze(1) # (B, 1, T)
        audio = self.audio_conv1x1(audio) # (B, C, T)
        audio = self.GELU(audio)

        visual = self.visual_embedder(visual).permute(0, 2, 1) # (B, E, t)
        visual = self.visual_conv1x1(visual) # (B, C, t)
        visual = self.GELU(visual)

        audio = self.fusion(audio, visual) # (B, C, T)
        audio = self.output_conv1x1(audio) # (B, 1, T)
        return audio.squeeze(1)
