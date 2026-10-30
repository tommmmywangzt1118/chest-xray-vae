"""FID / Inception Score of VAE samples with pytorch-ignite, plus figures for the report.

Follows https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/: images are resized to
299x299, replicated to 3 channels and fed to ignite's FID and InceptionScore metrics.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from torchvision.utils import save_image

from vae import VAE, load_split


def to_inception(x):
    """[N,1,H,W] in [0,1] -> [N,3,299,299] in [0,1]."""
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x.clamp(0, 1).repeat(1, 3, 1, 1)


def score(fake_iter, real, device, pool3):
    """FID(fake, real) and IS(fake); fake_iter yields batches, real is a uint8 tensor.

    "FID" is ignite's default (computed on the 1000-d Inception softmax, as in the blog post);
    "FID_2048" uses the standard 2048-d pool3 features, comparable to published FID values.
    """
    fid, is_ = FID(device=device), InceptionScore(device=device)
    fid2048 = FID(num_features=2048, feature_extractor=pool3, device=device)
    i = 0
    for fake in fake_iter:
        r = real[i:i + fake.size(0)].to(device).float() / 255.0
        i += fake.size(0)
        fake, r = to_inception(fake), to_inception(r)
        fid.update((fake, r))
        fid2048.update((fake, r))
        is_.update(fake)
    return {"FID": fid.compute(), "FID_2048": fid2048.compute(), "IS": is_.compute()}


def batches(x, bs, device, fn=None):
    for i in range(0, len(x), bs):
        b = x[i:i + bs].to(device).float() / 255.0
        yield fn(b) if fn else b


def plot_curves(log_csv, out_png):
    rows = list(csv.DictReader(open(log_csv)))
    ep = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, key, title in zip(axes, ["loss", "rec", "kl"],
                              ["Total loss (neg. ELBO)", "Reconstruction (BCE)", "KL divergence"]):
        ax.plot(ep, [float(r[f"train_{key}"]) for r in rows], label="train")
        ax.plot(ep, [float(r[f"test_{key}"]) for r in rows], label="test")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("nats / image")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run", default="runs/vae")
    p.add_argument("--n", type=int, default=5000, help="number of generated images")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    model = VAE(cfg["img_size"], cfg["latent_dim"]).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run, "vae.pt"), map_location=device))
    model.eval()

    cache = os.path.join(args.data, "..", "cache")
    xtr = load_split(args.data, "train", cfg["img_size"], cache)
    xte = load_split(args.data, "test", cfg["img_size"], cache)
    # reference distribution: training images (enough to pair with every scored batch)
    real = xtr[torch.randperm(len(xtr))[:max(args.n, len(xte))]]
    bs = 100

    def gen():
        for i in range(0, args.n, bs):
            yield model.sample(min(bs, args.n - i), device)

    @torch.no_grad()
    def recon(b):
        return model.decode(model.encode(b)[0])

    pool3 = InceptionModel(return_features=True, device=device)
    results = {}
    with torch.no_grad():
        results["generated"] = {"n": args.n, **score(gen(), real, device, pool3)}
        print("generated", results["generated"], flush=True)

        results["test_reconstructions"] = {"n": len(xte), **score(batches(xte, bs, device, recon), real, device, pool3)}
        print("test reconstructions", results["test_reconstructions"], flush=True)

        results["real_test_images"] = {"n": len(xte), **score(batches(xte, bs, device), real, device, pool3)}
        print("real test images (reference)", results["real_test_images"], flush=True)

    json.dump(results, open(os.path.join(args.run, "metrics.json"), "w"), indent=2)

    # figures
    with torch.no_grad():
        save_image(model.sample(64, device), os.path.join(args.run, "fig_samples.png"), nrow=8, padding=2)
        x = xte[torch.randperm(len(xte))[:8]].to(device).float() / 255.0
        save_image(torch.cat([x, recon(x)]), os.path.join(args.run, "fig_recon.png"), nrow=8, padding=2)
        # latent interpolation between pairs of random codes
        z0, z1 = torch.randn(2, 6, cfg["latent_dim"], device=device)
        t = torch.linspace(0, 1, 8, device=device)
        z = (z0[:, None] * (1 - t[None, :, None]) + z1[:, None] * t[None, :, None]).reshape(-1, cfg["latent_dim"])
        save_image(model.decode(z), os.path.join(args.run, "fig_interp.png"), nrow=8, padding=2)
    plot_curves(os.path.join(args.run, "log.csv"), os.path.join(args.run, "fig_loss.png"))


if __name__ == "__main__":
    main()
