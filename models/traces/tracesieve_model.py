#!/usr/bin/env python3
"""
TraceSieve: GAN noise filter + VGAE for trace anomaly detection (Zhang et al., ISSRE 2023).
Window-level 30s graphs on static 3GPP topology (no per-trace call chains available).
EWC omitted (440 training windows, no continual-learning need).
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import TraceRecord, SERVICES  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH = True
except ImportError:
    _TORCH = False

_N = len(SERVICES)
_SVC_IDX = {s: i for i, s in enumerate(SERVICES)}
_NODE_FEATS = 4             # span_count, error_rate, log_mean_dur, log_p95_dur


def _build_5g_adj() -> np.ndarray:
    """Undirected adjacency matrix from 3GPP 5G Core reference architecture."""
    A = np.zeros((_N, _N), dtype=np.float32)
    edges = [
        # AMF connectivity
        ("amf", "ausf"), ("amf", "udm"), ("amf", "smf"),
        ("amf", "nrf"),  ("amf", "nssf"), ("amf", "scp"),
        ("amf", "pcf"),  ("amf", "sepp"),
        # AUSF
        ("ausf", "udm"), ("ausf", "nrf"),
        # SMF
        ("smf", "pcf"), ("smf", "udm"), ("smf", "nrf"),
        ("smf", "bsf"), ("smf", "scp"),
        # PCF
        ("pcf", "udr"), ("pcf", "bsf"), ("pcf", "nrf"),
        # UDM
        ("udm", "udr"), ("udm", "nrf"),
        # Infrastructure NFs
        ("bsf",  "nrf"), ("nssf", "nrf"),
        ("sepp", "nrf"), ("scp",  "nrf"),
        # SCP as proxy
        ("scp", "ausf"), ("scp", "udm"), ("scp", "smf"),
        ("scp", "pcf"),  ("scp", "bsf"),
    ]
    for a, b in edges:
        if a in _SVC_IDX and b in _SVC_IDX:
            i, j = _SVC_IDX[a], _SVC_IDX[b]
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


_5G_ADJ = _build_5g_adj()


def _gcn_normalize(A: np.ndarray) -> np.ndarray:
    """Symmetric normalisation: D^{-1/2}(A+I)D^{-1/2}."""
    A_hat = A + np.eye(len(A), dtype=np.float32)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(A_hat.sum(1), 1e-8))
    return (d_inv_sqrt[:, None] * A_hat) * d_inv_sqrt[None, :]


_A_NORM = _gcn_normalize(_5G_ADJ)      # cached; topology is static


if _TORCH:

    class _GCNLayer(nn.Module):
        def __init__(self, in_feats: int, out_feats: int):
            super().__init__()
            self.W = nn.Linear(in_feats, out_feats, bias=False)

        def forward(self, X: "torch.Tensor",
                    A_norm: "torch.Tensor") -> "torch.Tensor":
            return torch.relu(A_norm @ self.W(X))

    class _GCNEncoder(nn.Module):
        def __init__(self, in_feats: int, hidden: int, latent: int):
            super().__init__()
            self.gcn1   = _GCNLayer(in_feats, hidden)
            self.w_mu   = nn.Linear(hidden, latent, bias=False)
            self.w_lv   = nn.Linear(hidden, latent, bias=False)

        def forward(self, X: "torch.Tensor",
                    A_norm: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
            H      = self.gcn1(X, A_norm)
            mu     = A_norm @ self.w_mu(H)
            logvar = A_norm @ self.w_lv(H)
            return mu, logvar

    class _FeatureDecoder(nn.Module):
        def __init__(self, latent: int, hidden: int, out_feats: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent, hidden), nn.ReLU(),
                nn.Linear(hidden, out_feats),
            )

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.net(z)

    class _VGAE(nn.Module):
        def __init__(self, in_feats: int, hidden: int, latent: int):
            super().__init__()
            self.encoder = _GCNEncoder(in_feats, hidden, latent)
            self.feat_decoder = _FeatureDecoder(latent, hidden, in_feats)

        def reparameterize(self, mu: "torch.Tensor",
                           logvar: "torch.Tensor") -> "torch.Tensor":
            if self.training:
                return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            return mu

        def forward(self, X: "torch.Tensor",
                    A_norm: "torch.Tensor",
                    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
            mu, logvar = self.encoder(X, A_norm)
            z          = self.reparameterize(mu, logvar)
            X_hat      = self.feat_decoder(z)
            A_hat      = torch.sigmoid(z @ z.t())   # inner-product decoder
            return X_hat, A_hat, mu, logvar

    class _GAN(nn.Module):
        def __init__(self, d: int, hidden: int):
            super().__init__()
            self.G = nn.Sequential(
                nn.Linear(d, hidden),   nn.ReLU(),
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, hidden),
            )
            self.D1 = nn.Linear(hidden, d)
            self.D2 = nn.Linear(hidden, d)

        def a1(self, F: "torch.Tensor") -> "torch.Tensor":
            return self.D1(self.G(F))

        def a2(self, F: "torch.Tensor") -> "torch.Tensor":
            return self.D2(self.G(F))


class TraceSieveDetector:
    """
    Window-level trace anomaly detector based on TraceSieve (ISSRE 2023).

    fit(train)           — noise-filter then train VGAE
    score(records)       — (nll_scores, labels)
    predict(records)     — (binary_preds, nll_scores, labels)
    """

    def __init__(
        self,
        hidden_dim:   int   = 64,
        latent_dim:   int   = 16,
        alpha:        float = 0.1,    # GAN noise-score weight (eq. 6)
        gan_hidden:   int   = 128,
        gan_epochs:   int   = 50,
        vgae_epochs:  int   = 100,
        batch_size:   int   = 32,
        lr:           float = 1e-3,
        noise_pct:    float = 95.0,   # percentile threshold for noise filter
        threshold_k:  float = 3.0,    # k-sigma anomaly threshold
        n_mc:         int   = 8,      # Monte Carlo samples for NLL
    ):
        self.hidden_dim  = hidden_dim
        self.latent_dim  = latent_dim
        self.alpha       = alpha
        self.gan_hidden  = gan_hidden
        self.gan_epochs  = gan_epochs
        self.vgae_epochs = vgae_epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.noise_pct   = noise_pct
        self.threshold_k = threshold_k
        self.n_mc        = n_mc

        self._mean:      Optional[np.ndarray] = None
        self._std:       Optional[np.ndarray] = None
        self._std99:     float                = 1.0
        self._vgae:      Optional["_VGAE"]    = None
        self._threshold: float                = float("inf")
        self._A_norm_t:  Optional["torch.Tensor"] = None
        self._A_t:       Optional["torch.Tensor"] = None

    @staticmethod
    def _extract_node_feats(records: list[TraceRecord]) -> np.ndarray:
        """Return (M, N, 4) node feature array from the first 44 dims of values."""
        return np.stack(
            [r.values[: _N * _NODE_FEATS].reshape(_N, _NODE_FEATS)
             for r in records],
            axis=0,
        ).astype(np.float32)

    def _normalise(self, X_raw: np.ndarray) -> np.ndarray:
        """Normalise (M, N, 4) → (M, N, 4) using training statistics."""
        M = len(X_raw)
        flat = X_raw.reshape(M, -1)
        return ((flat - self._mean) / self._std).reshape(M, _N, _NODE_FEATS).astype(np.float32)

    def fit(self, records: list[TraceRecord],
            epochs: Optional[int] = None) -> "TraceSieveDetector":
        if not _TORCH:
            raise RuntimeError("PyTorch is required for TraceSieveDetector.")
        if epochs is not None:
            self.vgae_epochs = epochs

        X_raw = self._extract_node_feats(records)
        M     = len(X_raw)
        flat  = X_raw.reshape(M, -1)

        self._mean = flat.mean(0)
        self._std  = np.where(flat.std(0) > 0, flat.std(0), 1.0)
        X_norm     = self._normalise(X_raw)

        self._A_t      = torch.tensor(_5G_ADJ)
        self._A_norm_t = torch.tensor(_A_NORM)

        X_norm = self._fit_gan_filter(X_norm, M)
        self._fit_vgae(X_norm)
        self._std99 = self._compute_std99(X_norm)

        train_scores    = self._score_batch(self._normalise(X_raw))
        self._threshold = float(
            train_scores.mean() + self.threshold_k * train_scores.std()
        )
        print(f"[TraceSieve] threshold={self._threshold:.4f}")
        return self

    def _fit_gan_filter(self, X_norm: np.ndarray, M: int) -> np.ndarray:
        """Train GAN noise filter; return denoised subset of X_norm."""
        d   = _N * _NODE_FEATS   # 44
        gan = _GAN(d, self.gan_hidden)

        # Train A1 (G + D1) and A2 (G + D2) with separate optimisers
        params_a1 = list(gan.G.parameters()) + list(gan.D1.parameters())
        params_a2 = list(gan.G.parameters()) + list(gan.D2.parameters())
        opt_a1    = optim.Adam(params_a1, lr=self.lr)
        opt_a2    = optim.Adam(params_a2, lr=self.lr)

        F_all   = torch.tensor(X_norm.reshape(M, -1))
        loader  = DataLoader(TensorDataset(F_all),
                             batch_size=max(self.batch_size, 1), shuffle=True)

        print(f"[TraceSieve] Training GAN noise filter ({M} windows, "
              f"{self.gan_epochs} epochs) ...")
        gan.train()
        for _ in range(self.gan_epochs):
            for (F,) in loader:
                opt_a1.zero_grad()
                A1_F    = gan.a1(F)
                # gradient flows into A1 so it learns to produce outputs A2 can reconstruct
                A2_A1_F = gan.a2(A1_F)
                loss_a1 = (
                    0.5 * nn.functional.mse_loss(A1_F, F) +
                    0.5 * nn.functional.mse_loss(A2_A1_F, F)
                )
                loss_a1.backward()
                opt_a1.step()

                opt_a2.zero_grad()
                A2_F     = gan.a2(F)
                A1_F_sg  = gan.a1(F).detach()   # stop gradient through A1
                A2_A1_F2 = gan.a2(A1_F_sg)
                loss_a2 = (
                    0.5 * nn.functional.mse_loss(A2_F, F) +
                    0.5 * nn.functional.mse_loss(A2_A1_F2, F)
                )
                loss_a2.backward()
                opt_a2.step()

        # Compute noise scores (paper eq. 6)
        gan.eval()
        with torch.no_grad():
            A1_all    = gan.a1(F_all)
            A2_A1_all = gan.a2(A1_all)
            noise_s   = (
                self.alpha       * ((F_all - A1_all)    ** 2).sum(1) +
                (1 - self.alpha) * ((F_all - A2_A1_all) ** 2).sum(1)
            ).numpy()

        threshold_n = float(np.percentile(noise_s, self.noise_pct))
        keep        = noise_s <= threshold_n
        n_kept      = int(keep.sum())
        print(f"[TraceSieve] Noise filter kept {n_kept}/{M} windows "
              f"(threshold={threshold_n:.4f})")
        return X_norm[keep]

    def _fit_vgae(self, X_norm: np.ndarray) -> None:
        """Train VGAE on denoised windows."""
        n = len(X_norm)
        self._vgae = _VGAE(_NODE_FEATS, self.hidden_dim, self.latent_dim)
        opt        = optim.Adam(self._vgae.parameters(), lr=self.lr)
        sched      = optim.lr_scheduler.CosineAnnealingLR(opt, self.vgae_epochs)

        Xt = torch.tensor(X_norm)
        An = self._A_norm_t
        At = self._A_t

        print(f"[TraceSieve] Training VGAE ({n} windows, "
              f"{self.vgae_epochs} epochs) ...")
        self._vgae.train()
        for ep in range(self.vgae_epochs):
            perm       = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx     = perm[i : i + self.batch_size]
                X_batch = Xt[idx]

                batch_loss = torch.tensor(0.0)
                for b in range(len(idx)):
                    X_b              = X_batch[b]
                    X_hat, A_hat, mu, logvar = self._vgae(X_b, An)

                    loss_feat = nn.functional.mse_loss(X_hat, X_b)
                    loss_adj  = nn.functional.binary_cross_entropy(A_hat, At)
                    kld       = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

                    batch_loss = batch_loss + loss_feat + loss_adj + 0.1 * kld

                batch_loss = batch_loss / max(len(idx), 1)
                opt.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self._vgae.parameters(), 1.0)
                opt.step()
                epoch_loss += batch_loss.item() * len(idx)

            sched.step()
            if ep % 25 == 0:
                print(f"[TraceSieve]   Epoch {ep:3d}/{self.vgae_epochs}  "
                      f"loss={epoch_loss/max(n, 1):.4f}")

        self._vgae.eval()

    def _compute_std99(self, X_norm: np.ndarray) -> float:
        """99.9th percentile of per-node std from encoder (paper §III-D)."""
        self._vgae.eval()
        An   = self._A_norm_t
        Xt   = torch.tensor(X_norm)
        stds: list[float] = []
        with torch.no_grad():
            for i in range(len(X_norm)):
                _, logvar = self._vgae.encoder(Xt[i], An)
                stds.append(torch.exp(0.5 * logvar).max().item())
        return float(np.percentile(stds, 99.9))

    def _score_batch(self, X_norm: np.ndarray) -> np.ndarray:
        """
        NLL via Monte Carlo with STD clipping (paper §III-D, eq. 11).
        X_norm: (M, N, 4) normalised node feature array.
        """
        self._vgae.eval()
        An  = self._A_norm_t
        At  = self._A_t
        Xt  = torch.tensor(X_norm)
        M   = len(X_norm)
        out = np.zeros(M, dtype=np.float32)

        with torch.no_grad():
            for i in range(M):
                X_i   = Xt[i]
                nlls: list[float] = []
                for _ in range(self.n_mc):
                    mu, logvar = self._vgae.encoder(X_i, An)
                    # STD clipping in latent space (paper §III-D entropy gap fix)
                    std = torch.exp(0.5 * logvar).clamp(max=self._std99)
                    z   = mu + std * torch.randn_like(std)

                    X_hat = self._vgae.feat_decoder(z)
                    A_hat = torch.sigmoid(z @ z.t())

                    # log p(X|z): unit-variance Gaussian → MSE
                    log_px = -0.5 * ((X_i - X_hat) ** 2).mean()
                    # log p(A|z): Bernoulli
                    log_pa = (
                        At * torch.log(A_hat + 1e-8)
                        + (1 - At) * torch.log(1 - A_hat + 1e-8)
                    ).mean()
                    # KL q||p (standard Gaussian prior)
                    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

                    nlls.append((-log_px - log_pa + kld).item())
                out[i] = float(np.mean(nlls))
        return out

    def score(self, records: list[TraceRecord]
              ) -> tuple[np.ndarray, np.ndarray]:
        if not records:
            return np.array([]), np.array([])
        X_norm = self._normalise(self._extract_node_feats(records))
        return self._score_batch(X_norm), np.array([r.label for r in records], dtype=int)

    def predict(self, records: list[TraceRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])
        return (scores > self._threshold).astype(int), scores, labels


if __name__ == "__main__":
    from data_loader import load_all

    data   = load_all()
    train  = data["train"]
    test   = data["test"]

    m = TraceSieveDetector(hidden_dim=64, latent_dim=16, gan_epochs=50,
                           vgae_epochs=100)
    m.fit(train)

    preds, scores, labels = m.predict(test)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    print(f"\n[TraceSieve] P={prec:.3f}  R={rec:.3f}  "
          f"F1={2*prec*rec/max(prec+rec, 1e-9):.3f}")
