from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventPyramidLatentProjector(nn.Module):
    """
    Project HDRev's multi-scale RGB-event features into Stable Diffusion latent space.

    For a 256x256 crop, the expected feature pyramid is:
      [B, 64, 256, 256], [B, 128, 128, 128],
      [B, 256, 64, 64], [B, 512, 32, 32].
    The output is [B, 4, 32, 32], matching SD VAE latents for one-step denoising.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        conditioning_channels: Sequence[int] = (64, 128, 256, 512),
        block_out_channels: Sequence[int] = (16, 32, 96, 256),
        downsample: bool = True,
    ):
        super().__init__()
        if len(conditioning_channels) != 4 or len(block_out_channels) != 4:
            raise ValueError("Stage 3 expects four HDRev feature levels and four projector channel levels.")

        conditioning_channels = tuple(conditioning_channels)
        block_out_channels = tuple(block_out_channels)

        self.conditioning_channels: Tuple[int, ...] = conditioning_channels
        self.block_out_channels: Tuple[int, ...] = block_out_channels

        self.conv_in = nn.Conv2d(conditioning_channels[0], block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i] + conditioning_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2 if downsample else 1),
                    nn.SiLU(),
                )
            )

        # Do not zero-initialize this layer: unlike ControlNet residual branches,
        # this tensor is the actual denoising start latent and must carry signal
        # from the first training step.
        self.conv_out = nn.Conv2d(
            block_out_channels[-1] + conditioning_channels[-1],
            latent_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, conditioning):
        if len(conditioning) != 4:
            raise ValueError(f"Expected four HDRev feature maps, got {len(conditioning)}.")

        embedding = F.silu(self.conv_in(conditioning[0]))
        for block, condition in zip(self.blocks, conditioning):
            # Each block receives the current embedding plus the same-resolution
            # HDRev feature, then downsamples to the next feature scale.
            embedding = block(torch.cat([embedding, condition], dim=1))

        return self.conv_out(torch.cat([embedding, conditioning[-1]], dim=1))
