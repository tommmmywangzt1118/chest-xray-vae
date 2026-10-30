"""Convolutional VAE on the Kaggle chest X-ray (pneumonia) dataset."""
import argparse
import csv
import glob
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image


# ---------------------------------------------------------------- data
def load_split(root, split, size, cache_dir):
    """Load a split as a uint8 tensor [N,1,H,W], cached to disk after the first pass."""
    cache = os.path.join(cache_dir, f"{split}_{size}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    paths = sorted(glob.glob(os.path.join(root, split, "*", "*.jpeg")))
    tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(size),          # shorter side -> size
        transforms.CenterCrop(size),
        transforms.PILToTensor(),
    ])
    x = torch.stack([tf(Image.open(p)) for p in paths])
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(x, cache)
    return x


# ---------------------------------------------------------------- model
def block(cin, cout, down=True):
    conv = (nn.Conv2d(cin, cout, 4, 2, 1) if down
            else nn.ConvTranspose2d(cin, cout, 4, 2, 1))
    return nn.Sequential(conv, nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))


class VAE(nn.Module):
    def __init__(self, img_size=128, latent_dim=128, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.feat = img_size // 2 ** len(channels)
        self.top = channels[-1]

        enc, cin = [], 1
        for c in channels:
            enc.append(block(cin, c))
            cin = c
        self.encoder = nn.Sequential(*enc)
        flat = self.top * self.feat ** 2
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flat)
        rev = list(channels[::-1])
        dec = [block(rev[i], rev[i + 1], down=False) for i in range(len(rev) - 1)]
        dec.append(nn.ConvTranspose2d(rev[-1], 1, 4, 2, 1))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.top, self.feat, self.feat)
        return torch.sigmoid(self.decoder(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # reparameterisation
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        return self.decode(torch.randn(n, self.latent_dim, device=device))


def vae_loss(recon, x, mu, logvar, beta):
    """Per-image negative ELBO: Bernoulli reconstruction term + beta * KL(q(z|x) || N(0, I))."""
    n = x.size(0)
    rec = F.binary_cross_entropy(recon, x, reduction="sum") / n
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n
    return rec + beta * kl, rec, kl


# ---------------------------------------------------------------- train
def run_epoch(model, loader, device, beta, opt=None):
    train = opt is not None
    model.train(train)
    tot = {"loss": 0.0, "rec": 0.0, "kl": 0.0}
    n = 0
    with torch.set_grad_enabled(train):
        for (x,) in loader:
            x = x.to(device, non_blocking=True).float() / 255.0
            recon, mu, logvar = model(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar, beta)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            b = x.size(0)
            for k, v in zip(tot, (loss, rec, kl)):
                tot[k] += v.item() * b
            n += b
    return {k: v / n for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dir containing train/ val/ test/")
    p.add_argument("--out", default="runs/vae")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    cache = os.path.join(args.data, "..", "cache")
    xtr = load_split(args.data, "train", args.img_size, cache)
    # The official val split has only 16 images, so evaluate on the test split instead.
    xte = load_split(args.data, "test", args.img_size, cache)
    print(f"train {tuple(xtr.shape)}  test {tuple(xte.shape)}", flush=True)
    tr = DataLoader(TensorDataset(xtr), args.batch_size, shuffle=True, drop_last=True,
                    num_workers=2, pin_memory=True)
    te = DataLoader(TensorDataset(xte), 256, num_workers=2, pin_memory=True)

    model = VAE(args.img_size, args.latent_dim).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    fixed_z = torch.randn(64, args.latent_dim, device=device)
    fixed_x = (xte[:32].float() / 255.0).to(device)
    log_path = os.path.join(args.out, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_rec", "train_kl",
                                "test_loss", "test_rec", "test_kl", "sec"])

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        trm = run_epoch(model, tr, device, args.beta, opt)
        tem = run_epoch(model, te, device, args.beta)
        lr = sched.get_last_lr()[0]
        sched.step()
        dt = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, lr, *trm.values(), *tem.values(), round(dt, 1)])
        print(f"ep {ep:3d} | train {trm['loss']:.1f} (rec {trm['rec']:.1f} kl {trm['kl']:.1f}) "
              f"| test {tem['loss']:.1f} | {dt:.1f}s", flush=True)

        if ep % 10 == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                save_image(model.decode(fixed_z), os.path.join(args.out, f"samples_ep{ep:03d}.png"), nrow=8)
                recon, _, _ = model(fixed_x)
                save_image(torch.cat([fixed_x, recon]), os.path.join(args.out, f"recon_ep{ep:03d}.png"), nrow=8)
            torch.save(model.state_dict(), os.path.join(args.out, "vae.pt"))


if __name__ == "__main__":
    main()
