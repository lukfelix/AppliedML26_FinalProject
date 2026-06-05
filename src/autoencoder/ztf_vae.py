"""
VAE-based anomaly detection for ZTF tabular features.

Usage:
    python ztf_vae.py --data your_data.csv

Requirements:
    pip install torch pandas numpy scikit-learn matplotlib umap-learn
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns to drop before training (identifiers, labels, raw timestamps)
DROP_COLS = [
    "id", "objectId", "htm16", "ssnamenr", "tns_name", "tns_type",
    "host_name", "classification", "classificationReliability",
    # Raw JD timestamps — not meaningful as static features
    "jdgmax", "jdrmax", "jdmax", "jdmin", "jd_g_minus_r",
    # Raw RA/Dec — not meaningful as static features
    # "decstd", "rastd", "ramean", "decmean", "glatmean", "glonmean",
    # # after ispection of features (https://lasair-ztf.lsst.ac.uk/schema/) these also seem less useful for anomaly detection, and have high missingness
    # "z", "distpsnr1", "ncand", "ncandgp", "sgmag1", "sgmag2",
]

# Columns to keep for interpretation / post-hoc analysis
META_COLS = [
    "objectId", "classification", "tns_type", "z",
    "ramean", "decmean", "glatmean", "glonmean",
]

SEED = 4242
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.mu_head  = nn.Linear(in_dim, latent_dim)
        self.logvar_head = nn.Linear(in_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_head(h), self.logvar_head(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        # hidden_dims: list[int] = [128, 64, 32],
        hidden_dims: list[int] = [64, 32, 16],
        latent_dim: int = 8,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)

    def reparameterise(self, mu, logvar):
        """Sample z ~ N(mu, sigma^2) using the reparameterisation trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference time

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    @torch.no_grad()
    def anomaly_scores(self, x: torch.Tensor, n_samples: int = 20) -> np.ndarray:
        """
        Monte-Carlo estimate of the anomaly score for each sample.

        We use the *negative ELBO* as the score:
            score = reconstruction_loss + KL_divergence

        Higher score → more anomalous.
        Averaging over n_samples reduces variance from the stochastic encoder.
        """
        self.train()  # enable stochastic sampling
        scores = []
        for _ in range(n_samples):
            x_hat, mu, logvar = self(x)
            input_dim  = x.shape[1]
            latent_dim = mu.shape[1]
            recon = nn.functional.mse_loss(x_hat, x, reduction="none").mean(dim=1)
            kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1) / latent_dim
            # recon = nn.functional.mse_loss(x_hat, x, reduction="none").sum(dim=1)
            # kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
            scores.append((recon + kl).cpu().numpy())
        self.eval()
        return np.mean(scores, axis=0)
    
    @torch.no_grad()
    def reconstruction_scores(self, x: torch.Tensor, n_samples: int = 20) -> np.ndarray:
        """
        Monte-Carlo estimate of the reconstruction score for each sample.

        We use the *reconstruction loss* as the score:
            score = reconstruction_loss

        Lower score → more easily reconstructed.
        Averaging over n_samples reduces variance from the stochastic encoder.
        """
        self.train()  # enable stochastic sampling
        scores = []
        for _ in range(n_samples):
            x_hat, mu, logvar = self(x)
            input_dim  = x.shape[1]
            latent_dim = mu.shape[1]
            recon = nn.functional.mse_loss(x_hat, x, reduction="none").mean(dim=1)
            scores.append((recon).cpu().numpy())
        self.eval()
        return np.mean(scores, axis=0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def elbo_loss(x, x_hat, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
    # Normalise KL by input dimension too
    kl_loss = -0.5 * torch.mean(
        (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1) / x.shape[1]
    )
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(path: str):
    df = pd.read_csv(path)

    # Save metadata for post-hoc analysis
    meta = df[[c for c in META_COLS if c in df.columns]].copy()

    # Drop non-feature columns (ignore missing ones)
    drop = [c for c in DROP_COLS if c in df.columns]
    features = df.drop(columns=drop)

    # Report missingness
    missing = features.isnull().mean().sort_values(ascending=False)
    print("\n--- Top-10 missing value rates ---")
    print(missing.head(10).to_string())

    # Impute with median (robust to outliers in astro data)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(features)

    # # Scale with RobustScaler (handles heavy-tailed ZTF distributions)
    # scaler = RobustScaler()
    scaler = QuantileTransformer(output_distribution="normal", random_state=SEED)
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    feature_names = list(features.columns)
    print(f"\nFeature matrix shape: {X_scaled.shape}")
    return X_scaled, meta, feature_names, scaler, imputer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: VAE,
    X: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: torch.device = torch.device("cpu"),
    val_fraction: float = 0.1,
    warmup_epochs=50,
):
    X_tensor = torch.from_numpy(X)

    # Train / val split (stratify not needed — unsupervised)
    n_val = max(1, int(len(X_tensor) * val_fraction))
    idx = torch.randperm(len(X_tensor), generator=torch.Generator().manual_seed(SEED))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = TensorDataset(X_tensor[train_idx])
    val_ds   = TensorDataset(X_tensor[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=10, factor=0.5, verbose=True
    )

    history = {"train_loss": [], "val_loss": [], "recon": [], "kl": []}

    print(f"\nTraining VAE for {epochs} epochs on {device} …")
    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            x_hat, mu, logvar = model(batch)
            # have a warmup period where beta increases from 0 to the target value, to help with training stability
            temp_beta = min(1.0, epoch / warmup_epochs) * beta
            loss, recon, kl = elbo_loss(batch, x_hat, mu, logvar, beta=temp_beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            train_losses.append(loss.item())

        # -- Validate --
        model.eval()
        val_losses, recon_losses, kl_losses = [], [], []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                x_hat, mu, logvar = model(batch)
                loss, recon, kl = elbo_loss(batch, x_hat, mu, logvar, beta=beta)
                val_losses.append(loss.item())
                recon_losses.append(recon)
                kl_losses.append(kl)

        mean_train = np.mean(train_losses)
        mean_val   = np.mean(val_losses)
        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        history["recon"].append(np.mean(recon_losses))
        history["kl"].append(np.mean(kl_losses))

        scheduler.step(mean_val)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>4}/{epochs}  "
                f"train={mean_train:.4f}  val={mean_val:.4f}  "
                f"recon={history['recon'][-1]:.4f}  kl={history['kl'][-1]:.4f}"
            )

    return history


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training(history: dict, save_path: str = "training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("ELBO loss")
    axes[0].set_title("Total loss")
    axes[0].legend()

    axes[1].plot(history["recon"], label="reconstruction")
    axes[1].plot(history["kl"],    label="KL divergence")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Loss components (val)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curves → {save_path}")


def plot_anomaly_scores(scores: np.ndarray, threshold_pct: float, save_path: str = "anomaly_scores.png"):
    threshold = np.percentile(scores, threshold_pct)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(scores, bins=100, color="steelblue", alpha=0.7, label="All objects")
    ax.axvline(threshold, color="crimson", linestyle="--",
               label=f"{threshold_pct}th percentile threshold")
    ax.set_xlabel("Anomaly score (−ELBO)")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title("VAE anomaly score distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved score distribution → {save_path}")

def plot_reconstruction_errors(scores: np.ndarray, threshold_pct: float, save_path: str = "reconstruction_errors.png"):
    threshold = np.percentile(scores, threshold_pct)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(scores, bins=100, color="steelblue", alpha=0.7, label="All objects")
    ax.axvline(threshold, color="crimson", linestyle="--",
               label=f"{threshold_pct}th percentile threshold")
    ax.set_xlabel("Reconstruction error (MSE)")
    ax.set_ylabel("Count")
    # ax.set_yscale("log")
    ax.set_title("VAE reconstruction error distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved reconstruction error distribution → {save_path}")


def plot_latent_umap(model: VAE, X: np.ndarray, meta: pd.DataFrame,
                     scores: np.ndarray, device: torch.device,
                     save_path: str = "latent_umap.png"):
    try:
        import umap
    except ImportError:
        print("umap-learn not installed — skipping UMAP plot. pip install umap-learn")
        return

    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        mu, _ = model.encoder(X_t)
        latent = mu.cpu().numpy()

    reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
    embedding = reducer.fit_transform(latent)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=scores, cmap="RdYlBu_r", s=5, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Anomaly score")
    ax.set_title("UMAP of VAE latent space (coloured by anomaly score)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved latent UMAP → {save_path}")

def plot_feature_reconstruction(
    model: VAE,
    X: np.ndarray,
    feature_names: list[str],
    device: torch.device,
    n_cols: int = 6,
    save_path: str = "feature_reconstruction.png",
):
    """
    For each input feature, plot the per-sample MSE as a bar chart ranked by
    mean reconstruction error. Two panels:
 
      Left:  Bar chart of mean MSE per feature (sorted worst → best), with
             ±1 std error bars. Immediately shows which features the VAE
             struggles to reconstruct.
 
      Right: Heatmap of per-sample MSE for the top-N worst features, so you
             can see whether the error is spread across all objects or driven
             by a subset (the latter is a strong anomaly signal).
    """
    model.eval()
    with torch.no_grad():
        X_t   = torch.from_numpy(X).to(device)
        x_hat, _, _ = model(X_t)
        # per-sample, per-feature squared error  [n_samples, n_features]
        se = (X_t - x_hat).pow(2).cpu().numpy()
 
    mean_se = se.mean(axis=0)   # [n_features]
    std_se  = se.std(axis=0)    # [n_features]
 
    # Sort features worst → best by mean MSE
    order        = np.argsort(mean_se)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_mean  = mean_se[order]
    sorted_std   = std_se[order]
    n_features   = len(feature_names)
 
    # ------------------------------------------------------------------ #
    # Panel 1: ranked bar chart
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(14, n_features * 0.35 + 6), 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )
 
    x_pos = np.arange(n_features)
    axes[0].bar(x_pos, sorted_mean, yerr=sorted_std,
                color="steelblue", alpha=0.75, capsize=2, error_kw={"linewidth": 0.6})
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(sorted_names, rotation=90, fontsize=7)
    axes[0].set_ylabel("Mean MSE (scaled space)")
    axes[0].set_title("Per-feature reconstruction error (worst → best)")
    axes[0].set_yscale("log")
 
    # Colour the worst quartile red for easy reading
    quartile_cut = sorted_mean[n_features // 4]
    for bar, val in zip(axes[0].patches, sorted_mean):
        if val >= quartile_cut:
            bar.set_color("crimson")
            bar.set_alpha(0.8)
 
    # ------------------------------------------------------------------ #
    # Panel 2: heatmap of per-object errors for top-N worst features
    # ------------------------------------------------------------------ #
    top_n     = min(20, n_features)
    top_idx   = order[:top_n]                   # indices into original feature array
    top_names = [feature_names[i] for i in top_idx]
    top_se    = se[:, top_idx]                  # [n_samples, top_n]
 
    # Sort objects by their total error over the worst features so the
    # heatmap isn't just noise
    row_order = np.argsort(top_se.sum(axis=1))[::-1]
    top_se_sorted = top_se[row_order]
 
    # Subsample to at most 500 objects for readability
    step = max(1, len(top_se_sorted) // 500)
    top_se_plot = top_se_sorted[::step]
 
    im = axes[1].imshow(
        top_se_plot.T,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    axes[1].set_yticks(np.arange(top_n))
    axes[1].set_yticklabels(top_names, fontsize=7)
    axes[1].set_xlabel(f"Objects (sorted by total error, 1 in {step} shown)")
    axes[1].set_title(f"Per-object MSE — top {top_n} worst features")
    plt.colorbar(im, ax=axes[1], label="MSE", shrink=0.8)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved feature reconstruction plot → {save_path}")
 
    # Also print a compact ranked table to stdout
    print("\n--- Feature reconstruction error (worst → best) ---")
    print(f"{'Feature':<25}  {'Mean MSE':>10}  {'Std MSE':>10}")
    print("-" * 50)
    for name, m, s in zip(sorted_names, sorted_mean, sorted_std):
        print(f"{name:<25}  {m:>10.4f}  {s:>10.4f}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VAE anomaly detection for ZTF tabular data")
    parser.add_argument("--data",          required=True,      help="Path to CSV file")
    parser.add_argument("--latent_dim",    type=int,   default=8)
    # parser.add_argument("--hidden_dims",   type=int,   nargs="+", default=[128, 64, 32])
    parser.add_argument("--hidden_dims",   type=int,   nargs="+", default=[256, 128, 32])
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--beta",          type=float, default=1.0,
                        help="Beta for beta-VAE (1=standard VAE)")
    parser.add_argument("--anomaly_pct",   type=float, default=99.0,
                        help="Percentile threshold for flagging anomalies")
    parser.add_argument("--mc_samples",    type=int,   default=20,
                        help="Monte-Carlo samples for anomaly score estimation")
    parser.add_argument("--output",        default="ztf_anomalies.csv",
                        help="Output CSV with anomaly scores")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load data ---
    X, meta, feature_names, scaler, imputer = load_and_preprocess(args.data)

    # --- Build model ---
    model = VAE(
        input_dim=X.shape[1],
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nVAE  |  input={X.shape[1]}  hidden={args.hidden_dims}  "
          f"latent={args.latent_dim}  params={n_params:,}")

    # --- Train ---
    history = train(
        model, X,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        device=device,
    )

    # --- Anomaly scores ---
    print("\nComputing anomaly scores …")
    model.eval()
    X_tensor = torch.from_numpy(X).to(device)
    scores = model.anomaly_scores(X_tensor, n_samples=args.mc_samples)
    reconstruction_scores = model.reconstruction_scores(X_tensor, n_samples=args.mc_samples)

    threshold = np.percentile(scores, args.anomaly_pct)
    n_anomalies = (scores > threshold).sum()
    print(f"Threshold ({args.anomaly_pct}th pct): {threshold:.4f}")
    print(f"Anomalies flagged: {n_anomalies} / {len(scores)}")

    # --- Save results ---
    results = meta.copy()
    results["anomaly_score"] = scores
    results["is_anomaly"]    = scores > threshold
    results.sort_values("anomaly_score", ascending=False, inplace=True)
    results.to_csv(args.output, index=False)
    print(f"\nSaved results → {args.output}")

    print("\nTop-20 anomalies:")
    display_cols = [c for c in ["objectId", "classification", "tns_type", "anomaly_score"]
                    if c in results.columns]
    print(results[display_cols].head(20).to_string(index=False))

    # --- Plots ---
    plot_training(history)
    plot_anomaly_scores(scores, args.anomaly_pct)
    plot_reconstruction_errors(reconstruction_scores, args.anomaly_pct)
    plot_latent_umap(model, X, meta, scores, device)
    plot_feature_reconstruction(model, X, feature_names, device)

    # --- Save model ---
    torch.save({
        "model_state":  model.state_dict(),
        "model_kwargs": {
            "input_dim":   X.shape[1],
            "hidden_dims": args.hidden_dims,
            "latent_dim":  args.latent_dim,
        },
        "feature_names": feature_names,
        "scaler":        scaler,
        "imputer":       imputer,
    }, "ztf_vae_checkpoint.pt")
    print("Saved model → ztf_vae_checkpoint.pt")


if __name__ == "__main__":
    main()
    # run the following
    # python ztf_vae.py --data ../../data/AppML_ZTF_table.csv --epochs 150 --latent_dim 8 --anomaly_pct 99