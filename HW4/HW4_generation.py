import os
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image



@dataclass
class CFG:
    ckpt_path: str = "./hw4_gan_adan.pt"
    out_dir: str = "./hw4_out_REPRODUCE"

    latent_dim: int = 100
    channels: int = 3


cfg = CFG()



def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]"""
    return (x.clamp(-1, 1) + 1) / 2



class Generator(nn.Module):
    def __init__(self, z_dim=100, ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base * 8, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(base),
            nn.ReLU(True),

            nn.ConvTranspose2d(base, ch, 4, 2, 1, bias=False),  # 64x64
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)



@torch.no_grad()
def reproduce_hw4(
    ckpt_path: str = "./hw4_gan_adan.pt",
    out_dir: str = "./hw4_out_REPRODUCE",
    n_images: int = 10,
    seed: int = 42
):
    """
    Loads the trained generator and generates n_images total (unconditional GAN).
    Saves:
      - reproduce_10.png  (a 2x5 grid)
      - reproduce_single_00.png ... reproduce_single_09.png (optional singles)
    """
    os.makedirs(out_dir, exist_ok=True)

    dev = device()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- Load checkpoint ----
    ckpt = torch.load(ckpt_path, map_location=dev)
    stateG = ckpt["G"] if isinstance(ckpt, dict) and "G" in ckpt else ckpt

    G = Generator(cfg.latent_dim, cfg.channels, base=64).to(dev)
    G.load_state_dict(stateG, strict=True)
    G.eval()

    z = torch.randn(n_images, cfg.latent_dim, 1, 1, device=dev)
    fake = G(z)                  # [-1, 1] because of Tanh
    fake01 = denorm(fake).cpu()  # [0, 1]

    grid_path = os.path.join(out_dir, "reproduce_10.png")
    save_image(fake01, grid_path, nrow=5)

    for i in range(n_images):
        save_image(fake01[i], os.path.join(out_dir, f"reproduce_single_{i:02d}.png"))

    print(f"[reproduce_hw4] Loaded: {ckpt_path}")
    print(f"[reproduce_hw4] Saved grid: {grid_path}")
    print(f"[reproduce_hw4] Saved singles to: {out_dir}")

    return grid_path


if __name__ == "__main__":
    print(f"Using device: {device()}")
    reproduce_hw4(cfg.ckpt_path, cfg.out_dir, n_images=10, seed=42)
