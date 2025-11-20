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

    def forward(self, x):
        """
        x: NORMALIZED image of shape (B, T, 3, H, W)
        returns: (B, T, 512)
        """
        if x.dim() != 5:
            raise ValueError("Input must be 5D tensor of shape (B, T, H, W, 3). Got shape: " + str(x.shape))

        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)  # (B*T, 3, H, W)
        feats = self.backbone(x)  # (B*T, 512)
        feats = feats.reshape(B, T, -1)
        assert feats.shape[-1] == self.emb_size

        return feats


class FiLMFusion(torch.nn.Module):
    def __init__(self, in_channels: int, con_channels: int, bias: bool = True):
        super().__init__()
        self.con_channels = con_channels
        self.in_channels = in_channels
        self.context_project = torch.nn.Conv1d(
            in_channels=con_channels,
            out_channels=2*in_channels,
            kernel_size=1,
            bias=bias
        )
        # Initialize to be identity-ish: gamma ≈ 1, beta ≈ 0
        torch.nn.init.zeros_(self.context_project.weight)
        if self.context_project.bias is not None:
            with torch.no_grad():
                self.context_project.bias[:in_channels].fill_(1.0)
                self.context_project.bias[in_channels:].zero_()

    def forward(self, x, context):
        """
        audio:   (B, C, T)
        context: (B, c, t)
        returns: (B, C, T)
        """
        assert x.dim() == 3 and context.dim() == 3, "input and context must be (batch, channels, time)"
        assert x.shape[0] == context.shape[0], "batch dim must match between input and context"
        assert x.shape[1] == self.in_channels and context.shape[1] == self.con_channels,\
            f"Channels ({x.shape[1]}, {context.shape[1]}) must match module channels ({self.in_channels}, {self.con_channels})"
        assert x.shape[-1] % context.shape[-1] == 0
        condition = self.context_project(context) # (B, 2*C, t)
        gamma, beta = condition.split(self.in_channels, dim=1) # each (B, C, t)
        gamma = gamma.repeat_interleave(x.shape[-1]//context.shape[-1], dim=-1) # (B, C, T)
        beta = beta.repeat_interleave(x.shape[-1]//context.shape[-1], dim=-1) # (B, C, T)
        x = gamma * x + beta
        return x


# R -> R down blocks, R up blocks, R + 1 encoder blocks, R + 1 fusion blocks, R decoder blocks.
# No explicit normalization or activations.
class SepReAVTSE(torch.nn.Module):
    def __init__(self,
                 avd_channels: Tuple[int],
                 visual_embedder: torch.nn.Module,
                 audio_blocks: torch.nn.ModuleList, # Audio -> Visual
                 visual_blocks: torch.nn.ModuleList,
                 fusion_blocks: torch.nn.ModuleList,
                 dec_blocks: torch.nn.ModuleList,
                 out_block: torch.nn.Module,
                 check: bool=True):
        super().__init__()
        assert(len(audio_blocks) - 1 == len(visual_blocks) - 1 == len(dec_blocks) == len(fusion_blocks) - 1)
        if check:
            for idx, (audio_block, visual_block) in enumerate(zip(audio_blocks, visual_blocks)):
                assert audio_block.channels == avd_channels[0]
                assert visual_block.channels == avd_channels[1]
                if idx < len(dec_blocks):
                    assert dec_blocks[idx].channels == avd_channels[2]
            for idx in range(len(fusion_blocks)):
                assert fusion_blocks[idx].in_channels == audio_blocks[idx].channels
                assert fusion_blocks[idx].con_channels == visual_blocks[idx].Channels

        self.R = len(audio_blocks) - 1
        self.video_embedder = visual_embedder

        # Wrap lists of modules in ModuleList for parameter counting
        self.audio_blocks = torch.nn.ModuleList(audio_blocks)
        self.visual_blocks = torch.nn.ModuleList(visual_blocks)
        self.fusion_blocks = torch.nn.ModuleList(fusion_blocks)
        self.dec_blocks = torch.nn.ModuleList(dec_blocks)

        self.audio_channels = avd_channels[0] # C
        self.visual_channels = avd_channels[1] # c
        self.dec_channels = avd_channels[2] # C'

        self.audio_conv1x1 = torch.nn.Conv1d(1, self.audio_channels, kernel_size=1)
        self.visual_conv1x1 = torch.nn.Conv1d(visual_embedder.emb_size, self.visual_channels, kernel_size=1)
        self.concat_conv1x1s = torch.nn.ModuleList([
            torch.nn.Conv1d(2*self.audio_channels, self.dec_channels, kernel_size=1) for _ in range(self.R)
        ])
        self.out_block = out_block
        self.audio_downs = torch.nn.ModuleList([
            (torch.nn.Conv1d(self.audio_channels, self.audio_channels, kernel_size=3, stride=2, padding=1))
            for _ in range(self.R)
        ]) 
        self.visual_downs = torch.nn.ModuleList([
            (torch.nn.Conv1d(self.visual_channels, self.visual_channels, kernel_size=3, stride=2, padding=1))
            for _ in range(self.R)
        ]) 
        self.upsamples = [
            (lambda x: torch.nn.functional.interpolate(x, scale_factor=2))
            for _ in range(self.R)
        ]

    def forward(self, audio, visual, int_feats=False):
      '''
      audio : (B, T)
      visual: (B, t, 3, H, W)
      '''
      assert audio.shape[1] % visual.shape[1] == 0
      r = visual.shape[1] % (1 << self.R)
      if r != 0:
          m = (1 << self.R)
          k = audio.shape[1] // visual.shape[1]
          visual = F.pad( # Maxium 3 last dims, use torch.pad for functional equivalence to np.pad
                visual,
                (0, 0, 0, 0, 0, m - r)
          )
          audio = F.pad(
                audio,
                (0, 0, 0, k*(m - r))
          )
      assert not int_feats, 'int_feats=True is currently not supported.'

      audio = audio.unsqueeze(1) # (B, 1, T)
      audio = self.audio_conv1x1(audio) # (B, C, T)

      visual = self.video_embedder(visual).permute(0, 2, 1) # (B, E, t)
      visual = self.visual_conv1x1(visual) # (B, C, t)

      dec_outputs: List[Tuple[torch.Tensor]] = []
      for idx, (audio_block, visual_block) in enumerate(zip(self.audio_blocks, self.visual_blocks)):
          audio = audio_block(audio)
          visual = visual_block(visual)
          dec_outputs.append((audio, visual))
          if idx < self.R:
              audio = self.audio_downs[idx](audio)
              visual = self.visual_downs[idx](visual)

      fusions: List[torch.Tensor] = []
      for (audio, visual), block in zip(dec_outputs, self.fusion_blocks):
          fusions.append(block(audio, visual))

      out_audio = fusions[-1]
      fusions = fusions[:-1]
      for fusion, upsample, concat_conv, dec_block in zip(
          reversed(fusions),
          self.upsamples,
          self.concat_conv1x1s,
          self.dec_blocks,
      ):
          out_audio = upsample(out_audio)
          out_audio = concat_conv(torch.cat([out_audio, fusion], dim=1))
          out_audio = dec_block(out_audio)

      out_audio = self.out_block(out_audio) # (B, T)
      return out_audio.squeeze(dim=1)


def build_basic_SepReAVTSE(R = 4, audio_channels = 128, visual_channels = 512, dec_channels = 128):
    audio_blocks = [torch.nn.GELU() for _ in range(R + 1)]
    visual_blocks = [torch.nn.GELU() for _ in range(R + 1)]
    fusion_blocks = [FiLMFusion(audio_channels, visual_channels) for _ in range(R + 1)]
    dec_blocks = [torch.nn.Identity() for _ in range(R)]
    out_block = torch.nn.Sequential(
        torch.nn.Conv1d(dec_channels, 4*dec_channels, kernel_size=1),
        torch.nn.GLU(dim=1),
        torch.nn.Conv1d(2*dec_channels, 1, kernel_size=1)
    )
    model = SepReAVTSE(
        avd_channels=(audio_channels, visual_channels, dec_channels),
        visual_embedder=VideoResNetEmbedder(),
        audio_blocks=audio_blocks,
        visual_blocks=visual_blocks,
        fusion_blocks=fusion_blocks,
        dec_blocks=dec_blocks,
        out_block=out_block,
        check=False
    )
    return model

# Different channels sizes
class MinimalAVTSE(torch.nn.Module):
    def __init__(self, audio_channels, visual_channels, dec_channels):
        super().__init__()
        self.audio_channels = audio_channels
        self.visual_channels = visual_channels
        self.dec_channels = dec_channels

        self.visual_embedder = VideoResNetEmbedder()
        self.audio_conv1x1 = torch.nn.Conv1d(1, self.audio_channels, kernel_size=1)
        self.visual_conv1x1 = torch.nn.Conv1d(self.visual_embedder.emb_size, self.visual_channels, kernel_size=1)
        self.GELU = torch.nn.GELU()
        self.fusion = FiLMFusion(self.audio_channels, self.visual_channels)
        self.out_block = torch.nn.Sequential(
            torch.nn.Conv1d(dec_channels, 4*dec_channels, kernel_size=1),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(2*dec_channels, 1, kernel_size=1)
        )

    def forward(self, audio, visual):
        """
        audio : (B, T)
        visual: (B, t, 3, H, W)
        """
        audio = audio.unsqueeze(1) # (B, 1, T)
        audio = self.audio_conv1x1(audio) # (B, C, T)
        audio = self.GELU(audio)

        visual = self.visual_embedder(visual).permute(0, 2, 1) # (B, E, t)
        visual = self.visual_conv1x1(visual) # (B, c, t)
        visual = self.GELU(visual)

        audio = self.fusion(audio, visual) # (B, C, T)
        audio = self.out_block(audio) # (B, 1, T)
        return audio.squeeze(dim=1)