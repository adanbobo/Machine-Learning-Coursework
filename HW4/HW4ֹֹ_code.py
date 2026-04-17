
import os
import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import matplotlib.pyplot as plt


# =========================
# Config (for ~8000 images)
# =========================
@dataclass
class CFG:
    data_dir = "/content/JPG/"


    out_dir: str = "./hw4_out"
    ckpt_path: str = "./hw4_gan_adan.pt"

    image_size: int = 64
    channels: int = 3

    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 120      # <<< for 8000 images (~125 steps/epoch -> ~3750 steps)

    latent_dim: int = 100
    lr: float = 2e-4
    betas: tuple = (0.5, 0.999)

    log_every: int = 100
    sample_every_steps: int = 1200
    use_amp: bool = True
    seed: int = 42


cfg = CFG()


# =========================
# Utils
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denorm(x):
    # [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1) / 2


# =========================
# Dataset (flat folder)
# =========================
class FlatImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder = folder_path
        self.transform = transform
        self.files = [f for f in os.listdir(folder_path)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(self.files) == 0:
            raise ValueError(f"No images found in directory: {folder_path}")
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def make_loader():
    tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = FlatImageDataset(cfg.data_dir, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
        drop_last=True
    )
    return dl


# =========================
# DCGAN models (64x64)
# =========================
class Generator(nn.Module):
    def __init__(self, z_dim=100, ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base*8, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(base*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(base*8, base*4, 4, 2, 1, bias=False), # 8x8
            nn.BatchNorm2d(base*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1, bias=False), # 16x16
            nn.BatchNorm2d(base*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base*2, base, 4, 2, 1, bias=False),   # 32x32
            nn.BatchNorm2d(base),
            nn.ReLU(True),

            nn.ConvTranspose2d(base, ch, 4, 2, 1, bias=False),       # 64x64
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, base, 4, 2, 1, bias=False),       # 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base, base*2, 4, 2, 1, bias=False),   # 16
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base*2, base*4, 4, 2, 1, bias=False), # 8
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base*4, base*8, 4, 2, 1, bias=False), # 4
            nn.BatchNorm2d(base*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base*8, 1, 4, 1, 0, bias=False),      # 1
        )

    def forward(self, x):
        return self.net(x).view(-1)  # logits


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# =========================
# Plot helper
# =========================
def plot_two_curves(a, b, labels, title, xlabel, ylabel, path):
    plt.figure()
    plt.plot(a, label=labels[0])
    plt.plot(b, label=labels[1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


@torch.no_grad()
def save_samples(G, step, n=25):
    os.makedirs(cfg.out_dir, exist_ok=True)
    G.eval()
    z = torch.randn(n, cfg.latent_dim, 1, 1, device=device())
    fake = denorm(G(z)).cpu()
    save_image(fake, os.path.join(cfg.out_dir, f"samples_step_{step:06d}.png"), nrow=5)


# =========================
# Training
# =========================
def train():
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    dl = make_loader()
    print(f"Found {len(dl.dataset)} images. Steps/epoch: {len(dl)}")
    print(f"Expected total steps ~ {len(dl) * cfg.epochs}")

    G = Generator(cfg.latent_dim, cfg.channels, base=64).to(device())
    D = Discriminator(cfg.channels, base=64).to(device())
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    optG = optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    optD = optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and torch.cuda.is_available()))

    d_losses_epoch = []
    g_losses_epoch = []
    global_step = 0

    for epoch in range(cfg.epochs):
        G.train()
        D.train()

        epoch_d = []
        epoch_g = []

        for i, real in enumerate(dl):
            real = real.to(device(), non_blocking=True)
            bsz = real.size(0)

            # labels (real label smoothing)
            real_targets = torch.empty(bsz, device=device()).uniform_(0.85, 1.0)
            fake_targets = torch.zeros(bsz, device=device())

            # -------------------------
            # Train D
            # -------------------------
            optD.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and torch.cuda.is_available())):
                real_logits = D(real)
                lossD_real = criterion(real_logits, real_targets)

                z = torch.randn(bsz, cfg.latent_dim, 1, 1, device=device())
                fake = G(z).detach()
                fake_logits = D(fake)
                lossD_fake = criterion(fake_logits, fake_targets)

                lossD = 0.5 * (lossD_real + lossD_fake)

            scaler.scale(lossD).backward()
            scaler.step(optD)

            # -------------------------
            # Train G
            # -------------------------
            optG.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and torch.cuda.is_available())):
                z = torch.randn(bsz, cfg.latent_dim, 1, 1, device=device())
                fake = G(z)
                fake_logits_for_G = D(fake)
                lossG = criterion(fake_logits_for_G, real_targets)

            scaler.scale(lossG).backward()
            scaler.step(optG)
            scaler.update()

            epoch_d.append(lossD.item())
            epoch_g.append(lossG.item())

            if global_step % cfg.log_every == 0:
                print(f"[E {epoch+1:03d}/{cfg.epochs}] [B {i:04d}/{len(dl)}] "
                      f"D: {lossD.item():.4f} | G: {lossG.item():.4f}")

            if global_step % cfg.sample_every_steps == 0:
                save_samples(G, global_step, n=25)

            global_step += 1

        d_losses_epoch.append(float(np.mean(epoch_d)))
        g_losses_epoch.append(float(np.mean(epoch_g)))

        plot_two_curves(
            d_losses_epoch, g_losses_epoch,
            labels=("D loss (epoch avg)", "G loss (epoch avg)"),
            title="GAN losses per epoch",
            xlabel="epoch", ylabel="loss",
            path=os.path.join(cfg.out_dir, "loss_per_epoch.png")
        )

        torch.save({"G": G.state_dict(), "D": D.state_dict(), "cfg": cfg.__dict__}, cfg.ckpt_path)

    # final samples
    with torch.no_grad():
        save_samples(G, global_step, n=25)
        z = torch.randn(16, cfg.latent_dim, 1, 1, device=device())
        final = denorm(G(z)).cpu()
        save_image(final, os.path.join(cfg.out_dir, "final_samples.png"), nrow=4)

    print(f"Saved model to: {cfg.ckpt_path}")
    print(f"Outputs in: {cfg.out_dir}")
    return G


# =========================
# Similar/Dissimilar pairs + PCA + L2 in z (FASTER)
# =========================
@torch.no_grad()
def analyze_latent_pairs(G, pool=200, pairs=3, trials=2500):
    os.makedirs(cfg.out_dir, exist_ok=True)
    G.eval()

    z = torch.randn(pool, cfg.latent_dim, 1, 1, device=device())
    x = G(z)                 # [-1,1]
    x01 = denorm(x).cpu()    # [0,1]

    # quick similarity features
    gray = x01.mean(dim=1, keepdim=True)
    small = F.interpolate(gray, size=(16, 16), mode="area")
    feat = small.view(pool, -1)
    feat = (feat - feat.mean(1, keepdim=True)) / (feat.std(1, keepdim=True) + 1e-8)

    cand = []
    for _ in range(trials):
        i = random.randrange(pool)
        j = random.randrange(pool)
        if i == j:
            continue
        d_img = torch.norm(feat[i] - feat[j], p=2).item()
        cand.append((d_img, i, j))

    cand.sort(key=lambda t: t[0])
    similar = cand[:pairs]
    dissim = cand[-pairs:]

    def zL2(i, j):
        zi = z[i].view(-1)
        zj = z[j].view(-1)
        return torch.norm(zi - zj, p=2).item()

    def save_pair(i, j, name):
        grid = make_grid(torch.stack([x01[i], x01[j]], dim=0), nrow=2)
        save_image(grid, os.path.join(cfg.out_dir, f"{name}.png"))

    print("=== Similar pairs ===")
    for k, (dimg, i, j) in enumerate(similar):
        zl2 = zL2(i, j)
        print(f"sim{k}: imgDist={dimg:.4f}, zL2={zl2:.4f}")
        save_pair(i, j, f"pair_sim_{k}_imgDist_{dimg:.2f}_zL2_{zl2:.2f}")

    print("=== Dissimilar pairs ===")
    for k, (dimg, i, j) in enumerate(dissim):
        zl2 = zL2(i, j)
        print(f"dis{k}: imgDist={dimg:.4f}, zL2={zl2:.4f}")
        save_pair(i, j, f"pair_dis_{k}_imgDist_{dimg:.2f}_zL2_{zl2:.2f}")

    # PCA on selected z points
    used = []
    tags = []
    for (_, i, j) in similar:
        used += [i, j]
        tags += ["sim", "sim"]
    for (_, i, j) in dissim:
        used += [i, j]
        tags += ["dis", "dis"]

    Z = z[used].view(len(used), -1).detach().cpu().numpy()
    Z2 = PCA(n_components=2).fit_transform(Z)

    plt.figure(figsize=(8, 6))
    for label in ["sim", "dis"]:
        mask = np.array(tags) == label
        plt.scatter(Z2[mask, 0], Z2[mask, 1], label=label)
    plt.title("Latent z PCA (selected pairs)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "latent_pairs_pca.png"))
    plt.close()

    print("Saved latent PCA plot: latent_pairs_pca.png")


if __name__ == "__main__":
    print(f"Using device: {device()} | AMP: {cfg.use_amp and torch.cuda.is_available()}")
    G = train()


