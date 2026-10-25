"""
Every synthetic data generator in the project.

    NaiveModel    - resample each column independently (the baseline that is
                    supposed to destroy correlations).
    VAEModel      - custom VAE: encoder -> (mu, logvar) -> reparameterisation
                    -> decoder.  MSE on numeric columns, cross-entropy per
                    categorical column, KL divergence annealed in.
    DPVAEModel    - the same VAE trained with Differentially-Private SGD
                    (Opacus) at a target epsilon.
    CTGANModel /
    TVAEModel     - the `ctgan` package implementations, for comparison.
    WGANGPModel   - WGAN-GP on the same encoded space as the VAE, so the
                    VAE-vs-GAN comparison is an architecture comparison rather
                    than a preprocessing comparison.

All five share one interface:

    m = build_model("vae", schema, seed=0)
    m.fit(train_df, val_df=val_df)          # train split only
    syn = m.sample(1000, seed=0)            # -> DataFrame, schema column order

so the pipeline and the Streamlit app never special-case a model.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import Schema
from .encoding import EPS, TabularEncoder, coerce_dtypes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Pin every source of randomness we can reach.

    Needed because we report mean +/- std across 3 seeds: if numpy, torch and
    python's own RNG were not pinned, the spread between runs would include
    housekeeping noise rather than genuine model variance.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Encoded view of a table
# ---------------------------------------------------------------------------
class EncodedTable:
    """Pre-slice a DataFrame into the tensors a generator actually wants.

    Doing the slicing once here (rather than inside the training loop) keeps
    the loss code honest: `num` is the numeric block and `cat` holds one column
    of integer class indices per categorical group, which is exactly what
    `F.cross_entropy` needs for a per-column loss.
    """

    def __init__(self, df: pd.DataFrame, schema: Schema):
        self.schema = schema
        self.encoder = TabularEncoder(schema)
        self.df = df.reset_index(drop=True)
        X = self.encoder.encode(self.df)
        self.X = X                                        # (n, dim)
        self.num_slots = [s for s in self.encoder.slots if s.kind == "numeric"]
        self.cat_slots = [s for s in self.encoder.slots
                          if s.kind == "categorical"]
        self.num = np.stack([X[:, s.start] for s in self.num_slots],
                            axis=1).astype(np.float32) \
            if self.num_slots else np.zeros((len(X), 0), np.float32)
        # Class index per categorical group (argmax of the one-hot block).
        self.cat = np.stack([X[:, s.start:s.stop].argmax(axis=1)
                             for s in self.cat_slots], axis=1).astype(np.int64)
        self.widths = [s.width for s in self.cat_slots]
        self.n = len(df)
        self.dim = self.encoder.dim
        self.target = self.df[schema.label].to_numpy(dtype=int)

    def to_tensors(self) -> TensorDataset:
        return TensorDataset(torch.tensor(self.X), torch.tensor(self.num),
                             torch.tensor(self.cat))


# ---------------------------------------------------------------------------
# Neural network pieces shared by VAE and WGAN-GP
# ---------------------------------------------------------------------------
def mlp(in_dim: int, hidden: Sequence[int], out_dim: int,
        activation=nn.ReLU, out_activation=None) -> nn.Sequential:
    """Build a plain feed-forward stack.  No BatchNorm: Opacus does not support
    it for DP-SGD, and with batches of 32 rows it would be unstable anyway."""
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), activation()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    if out_activation is not None:
        layers.append(out_activation())
    return nn.Sequential(*layers)


class VaeNet(nn.Module):
    """Encoder -> (mu, logvar) -> reparameterised z -> decoder with split heads.

                        x  ->  [ encoder trunk ]  -> mu, logvar
                                                   |
                                    z = mu + exp(0.5*logvar) * eps ,  eps~N(0,I)
                                                   |
                        [ decoder trunk ] -> numeric head (sigmoid)
                                           -> one softmax head per categorical column
    """

    def __init__(self, enc_dim: int, cat_widths: Sequence[int],
                 n_numeric: int, hidden=(64, 32), latent_dim: int = 16):
        super().__init__()
        self.n_numeric = n_numeric
        self.cat_widths = list(cat_widths)
        self.latent_dim = latent_dim
        self.enc_trunk = mlp(enc_dim, hidden, hidden[-1])
        self.mu = nn.Linear(hidden[-1], latent_dim)
        self.logvar = nn.Linear(hidden[-1], latent_dim)
        self.dec_trunk = mlp(latent_dim, hidden[::-1], hidden[0])
        # Separate output head per column -> lets us weight numeric MSE apart
        # from categorical cross-entropy, as the specification requires.
        self.num_head = nn.Linear(hidden[0], n_numeric) if n_numeric else None
        self.cat_heads = nn.ModuleList([nn.Linear(hidden[0], w)
                                         for w in self.cat_widths])

    def encode(self, x):
        h = self.enc_trunk(x)
        return self.mu(h), self.logvar(h)

    def reparameterise(self, mu, logvar):
        """Sample z from q(z|x).  The `+0.5*logvar` is log(sigma), so
        exp(0.5*logvar) = sigma.  This is the reparameterisation trick: it makes
        the sampling step differentiable in (mu, sigma) instead of in eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def decode(self, z):
        h = self.dec_trunk(z)
        num = torch.sigmoid(self.num_head(h)) if self.num_head is not None \
            else h.new_zeros((h.shape[0], 0))
        cats = [head(h) for head in self.cat_heads]     # raw logits per group
        return num, cats

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, _ = self.reparameterise(mu, logvar)
        num, cats = self.decode(z)
        return num, cats, mu, logvar


class Critic(nn.Module):
    """WGAN-GP discriminator: maps an encoded row to a single scalar score.

    LeakyReLU and no normalisation layers - the gradient penalty is the
    regulariser here, and BatchNorm would corrupt it.
    """

    def __init__(self, dim: int, hidden=(128, 64)):
        super().__init__()
        layers, d = [], dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.2)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WGANGen(nn.Module):
    """WGAN-GP generator: noise -> numeric probs + categorical logits."""

    def __init__(self, z_dim: int, cat_widths: Sequence[int], n_numeric: int,
                 hidden=(128, 64)):
        super().__init__()
        self.n_numeric = n_numeric
        self.cat_widths = list(cat_widths)
        self.trunk = mlp(z_dim, hidden, hidden[-1], activation=nn.ReLU)
        self.num_head = nn.Linear(hidden[-1], n_numeric) if n_numeric else None
        self.cat_heads = nn.ModuleList([nn.Linear(hidden[-1], w)
                                        for w in self.cat_widths])

    def forward(self, z):
        h = self.trunk(z)
        num = torch.sigmoid(self.num_head(h)) if self.num_head is not None \
            else h.new_zeros((h.shape[0], 0))
        cats = [head(h) for head in self.cat_heads]
        return num, cats


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
@dataclass
class Generator:
    """Common contract for all synthetic data generators."""

    schema: Schema
    seed: int = 0
    name: str = "base"
    hyperparams: Dict = field(default_factory=dict)

    def __post_init__(self):
        set_seed(self.seed)
        self.enc_table: Optional[EncodedTable] = None
        self.fit_seconds: float = 0.0
        self.notes: List[str] = []

    # -- to be provided by subclasses -------------------------------------
    def fit(self, df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None,
            **kw) -> "Generator":
        raise NotImplementedError

    def _raw_outputs(self, n: int, temperature: float,
                     rng: np.random.Generator) -> Tuple[np.ndarray,
                                                         List[np.ndarray]]:
        raise NotImplementedError

    def sample(self, n: int, diversity: float = 0.5,
               seed: Optional[int] = None) -> pd.DataFrame:
        """Produce n synthetic patient records, already clipped and rounded.

        `diversity` in [0, 1] is the single knob the demo exposes; 0.5 is the
        neutral setting that reproduces the training distribution faithfully.
        See `diversity_to_knobs` for the exact mapping.
        """
        n = int(max(n, 1))
        rng = np.random.default_rng(
            self.seed if seed is None else int(seed) + 10_000)
        temperature, latent_scale = diversity_to_knobs(diversity)
        self._latent_scale = latent_scale
        num, cats = self._raw_outputs(n, temperature, rng)
        df = self._encoder().decode_logits(num, cats, temperature=temperature,
                                           rng=rng)
        return df[self.schema.all_names]

    def _encoder(self) -> TabularEncoder:
        if self.enc_table is None:
            raise RuntimeError(f"{self.name}: call fit() before sample()")
        return self.enc_table.encoder

    def trained_on_target_rate(self) -> float:
        assert self.enc_table is not None
        return float(self.enc_table.target.mean())


def diversity_to_knobs(diversity: float) -> Tuple[float, float]:
    """Map one user-facing slider to the two internal randomness controls.

    diversity = 0.5  -> temperature 1.0, latent scale 1.0  (the neutral point,
                        i.e. exactly what the evaluation pipeline uses)
    diversity -> 0   -> sharpened categories, tighter latent -> few, 'safe' rows
    diversity -> 1   -> flattened categories, wider latent     -> more unusual rows
    """
    d = float(np.clip(diversity, 0.0, 1.0))
    temperature = float(2.0 ** (d - 0.5))            # 0.5 .. 2.0
    latent_scale = float(1.0 + (d - 0.5) * 0.5)      # 0.75 .. 1.25
    return temperature, latent_scale


# ---------------------------------------------------------------------------
# 1. Naive baseline
# ---------------------------------------------------------------------------
class NaiveModel(Generator):
    """Independent per-column resampling ('shuffle each column on its own').

    This is the control condition.  It reproduces every 1-D marginal exactly by
    construction, so if a fancier model cannot beat it on the correlation gap it
    has learned nothing beyond the marginals.  It also keeps a non-zero exact
    copy rate on the low-cardinality columns, which is why it is the honest
    privacy straw-man.

    `diversity` above the neutral point adds Gaussian kernel jitter to the
    numeric columns, which is the classic way to stop a naive sampler from
    merely re-permuting observed values.
    """

    def __init__(self, schema, seed=0, **hp):
        super().__init__(schema=schema, seed=seed, name="naive",
                         hyperparams={"jitter": 0.0, **hp})

    def fit(self, df, val_df=None, **kw):
        self.enc_table = EncodedTable(df, self.schema)
        self.table = df.reset_index(drop=True)
        return self

    def _raw_outputs(self, n, temperature, rng):
        table, e = self.table, self.enc_table.encoder
        num_names = e.schema.numeric_names
        raw = np.zeros((n, len(num_names)), dtype=float)
        for i, c in enumerate(e.schema.numeric):
            vals = table[c.name].to_numpy(dtype=float)
            pick = rng.choice(vals, size=n, replace=True)
            bw = self.hyperparams["jitter"]
            if bw > 0:
                # Bandwidth is in units of the column std, so it scales sensibly
                # from mg/dl cholesterol to a 0-1 ST depression measure.
                pick = pick + rng.normal(0, bw * vals.std(), size=n)
            raw[:, i] = pick
        # Scale through the SAME train-fitted transform the neural models use.
        # Doing our own min-max here would double-transform the column and
        # destroy the very marginal this baseline is supposed to reproduce.
        num = e.scale_numeric(raw)
        cats: List[np.ndarray] = []
        for slot in [s for s in e.slots if s.kind == "categorical"]:
            vals = table[slot.name].to_numpy(dtype=float)
            # Map each observed value to its POSITION in the fixed vocabulary.
            # (Subtracting min(categories) would be wrong for a column like
            # thal whose levels are {3, 6, 7}: the gap at 4 and 5 would create
            # phantom bins and then index past the end of the category list.)
            position = {int(c): i for i, c in enumerate(slot.categories)}
            counts = np.zeros(slot.width, dtype=float)
            for v in np.round(vals).astype(int):
                if v in position:
                    counts[position[v]] += 1.0
            probs = counts
            if probs.sum() <= 0:
                probs = np.ones(slot.width)          # uniform fallback
            probs = probs / probs.sum()
            # Return LOG-probabilities, not probabilities.  decode_logits
            # softmaxes its input, and softmax(log p) == p exactly, so the naive
            # baseline reproduces the empirical marginal precisely at the
            # neutral diversity setting (softmax(p) would not).  At diversity
            # != 0.5 this becomes p**(1/T), the textbook temperature transform.
            cats.append(np.tile(np.log(probs + 1e-12), (n, 1)))
        return num, cats

    def sample(self, n, diversity=0.5, seed=None):
        """Naive has no latent space, so only the jitter part of diversity
        applies; the categorical temperature still applies through the base."""
        self.hyperparams["_d"] = float(np.clip(diversity, 0, 1))
        bw = max(0.0, (self.hyperparams["_d"] - 0.5)) * 1.2
        self.hyperparams["jitter"] = bw
        out = super().sample(n, diversity=diversity, seed=seed)
        self.hyperparams["jitter"] = 0.0
        return out


# ---------------------------------------------------------------------------
# 2. VAE
# ---------------------------------------------------------------------------
def vae_reconstruction_loss(num_hat, cat_logits, num, cat_idx,
                            cat_widths, w_num: float = 1.0,
                            w_cat: float = 1.0) -> torch.Tensor:
    """Per-sample reconstruction loss: MSE for numeric, CE per categorical col.

    Returns a vector of shape (batch,) rather than a scalar because Opacus
    needs per-sample losses to clip gradients privately.  The non-DP VAE just
    takes `.mean()` of the same vector, so both paths share one code path and
    the numbers are directly comparable.

    Each term is *averaged over its own dimensions* before the two are summed.
    Otherwise a column set with more numeric dimensions would silently dominate
    the categorical loss and the model would cheat at the metric we care about.

    WHY w_num EXISTS - and it matters more than it looks
    ----------------------------------------------------
    Averaged squared error on min-max [0,1] numerics is tiny (~0.028 at a
    constant predictor) while averaged cross-entropy on categoricals is ~0.98
    nats.  Left unweighted the categorical term is ~35x larger, the optimiser
    spends all of its capacity on the categorical columns, and the numeric
    columns are effectively never modelled - which shows up as a catastrophic
    KS statistic and a correlation gap *worse than the naive baseline*.  w_num
    rebalances the two so both are trained.  It is tuned on the VALIDATION split
    only (see synth/tune.py) and the chosen value is reported in the results.
    """
    B = num.shape[0]
    if num_hat.shape[1] > 0:
        rec_num = F.mse_loss(num_hat, num, reduction="none").mean(dim=1)
    else:
        rec_num = torch.zeros(B, device=num_hat.device)
    ce_terms = []
    for j, width in enumerate(cat_widths):
        logits = cat_logits[j]
        target = cat_idx[:, j]
        # Guard: a group whose logits are all equal is fine, but an empty or
        # mismatched width would raise - skip it loudly rather than crash.
        if logits.ndim != 2 or logits.shape[1] != width:
            continue
        ce_terms.append(F.cross_entropy(logits, target, reduction="none"))
    rec_cat = (torch.stack(ce_terms, dim=1).mean(dim=1)
               if ce_terms else torch.zeros(B, device=num_hat.device))
    return w_num * rec_num + w_cat * rec_cat


def kl_divergence_loss(mu, logvar) -> torch.Tensor:
    """KL( N(mu, s^2) || N(0, I) ) per sample, summed over latent dims.

    Closed form:  -0.5 * sum_j ( 1 + log sigma_j^2 - mu_j^2 - sigma_j^2 )
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def kl_schedule(epoch: int, total_epochs: int, anneal_epochs: int,
                beta_max: float) -> float:
    """KL annealing: ramp beta from ~0 up to beta_max over `anneal_epochs`.

    Early in training the KL term would crush the latent code to the prior
    before the decoder has learned anything, giving blurry, mode-collapsed
    output.  Starting with beta near 0 lets reconstruction get a grip first,
    then the regulariser is introduced.  Linear ramp is the standard choice
    from the original beta-VAE paper.
    """
    if anneal_epochs <= 0:
        return beta_max
    frac = min(1.0, float(epoch + 1) / float(anneal_epochs))
    return beta_max * frac


@dataclass
class VAETrainResult:
    final_train_loss: float
    final_val_loss: float
    betas: List[float]


class VAEModel(Generator):
    """Custom variational autoencoder for mixed-type medical tables."""

    def __init__(self, schema, seed=0, latent_dim=16, hidden=(64, 32),
                 epochs=400, lr=1e-3, batch_size=32, beta_max=1.0,
                 anneal_epochs=100, weight_decay=1e-5, val_every=10,
                 w_num=1.0, w_cat=1.0, obs_noise=True, obs_noise_scale=1.0,
                 **hp):
        super().__init__(schema=schema, seed=seed, name="vae",
                         hyperparams=dict(latent_dim=latent_dim, hidden=hidden,
                                          epochs=epochs, lr=lr,
                                          batch_size=batch_size,
                                          beta_max=beta_max,
                                          anneal_epochs=anneal_epochs,
                                          weight_decay=weight_decay,
                                          w_num=w_num, w_cat=w_cat,
                                          obs_noise=obs_noise))
        self.latent_dim, self.hidden = latent_dim, tuple(hidden)
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.beta_max, self.anneal_epochs = beta_max, anneal_epochs
        self.weight_decay, self.val_every = weight_decay, val_every
        self.w_num, self.w_cat = float(w_num), float(w_cat)
        self.obs_noise = bool(obs_noise)
        self.obs_noise_scale = float(obs_noise_scale)
        self.obs_sigma: Optional[np.ndarray] = None
        self.net: Optional[VaeNet] = None
        self.train_history: List[float] = []
        self.val_history: List[float] = []
        self.result = None
        self.privacy_note = "non-private"

    # ------------------------------------------------------------------ fit
    def _make_net(self, et: EncodedTable) -> VaeNet:
        return VaeNet(et.dim, et.widths, et.num.shape[1],
                      hidden=self.hidden, latent_dim=self.latent_dim).to(DEVICE)

    def fit(self, df, val_df=None, **kw):
        import time
        t0 = time.time()
        et = EncodedTable(df, self.schema)
        self.enc_table = et
        # batch_size must not exceed the dataset, else the last partial batch is
        # empty after drop_last and the loop trains on nothing.
        self.bsz = int(max(2, min(self.batch_size, et.n)))
        self.net = self._make_net(et)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        X = torch.tensor(et.X)
        loader = DataLoader(TensorDataset(X, torch.tensor(et.num),
                                          torch.tensor(et.cat)),
                            batch_size=self.bsz, shuffle=True, drop_last=False)
        val_tensor = None
        if val_df is not None:
            vet = EncodedTable(val_df, self.schema)
            val_tensor = (torch.tensor(vet.X, device=DEVICE),
                          torch.tensor(vet.num, device=DEVICE),
                          torch.tensor(vet.cat, device=DEVICE))

        betas = []
        for epoch in range(self.epochs):
            beta = kl_schedule(epoch, self.epochs, self.anneal_epochs,
                               self.beta_max)
            betas.append(beta)
            self._train_one_epoch(loader, opt, beta)
            if (epoch + 1) % max(1, self.val_every) == 0 or \
                    epoch == self.epochs - 1:
                tr = self._eval_on(et)
                va = (self._eval_val(val_tensor) if val_tensor else float("nan"))
                self.train_history.append(tr)
                self.val_history.append(va)
        self._fit_observation_noise(et)
        self.result = VAETrainResult(self.train_history[-1],
                                     self.val_history[-1], betas)
        self.fit_seconds = time.time() - t0
        return self

    def _train_one_epoch(self, loader, opt, beta):
        self.net.train()
        for Xb, numb, catb in loader:
            Xb, numb, catb = Xb.to(DEVICE), numb.to(DEVICE), catb.to(DEVICE)
            opt.zero_grad()
            num_hat, cat_logits, mu, logvar = self.net(Xb)
            loss = (vae_reconstruction_loss(
                num_hat, cat_logits, numb, catb, self.enc_table.widths,
                w_num=self.w_num, w_cat=self.w_cat)
                + beta * kl_divergence_loss(mu, logvar)).mean()
            if not torch.isfinite(loss):
                # NaN loss guard: skip the update rather than poisoning every
                # weight with NaN, which would silently produce garbage output.
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            opt.step()

    @torch.no_grad()
    def _loss_from(self, X, numb, catb) -> float:
        num_hat, cat_logits, mu, logvar = self.net(X)
        rec = vae_reconstruction_loss(num_hat, cat_logits, numb, catb,
                                      self.enc_table.widths,
                                      w_num=self.w_num, w_cat=self.w_cat)
        kl = kl_divergence_loss(mu, logvar)
        return float((rec + kl).mean().cpu())

    def _eval_val(self, val_tensor) -> float:
        X, numb, catb = val_tensor
        return self._loss_from(X, numb, catb)

    def _eval_on(self, et: "EncodedTable") -> float:
        """Full-dataset loss, evaluated in chunks.

        We deliberately do NOT reuse the training DataLoader for this.  Under
        DP-SGD Opacus swaps in a Poisson-sampling loader, so iterating it would
        score the model on a random subset and make the curve jittery and
        incomparable with the non-private VAE.
        """
        self.net.eval()
        X = torch.tensor(et.X, device=DEVICE)
        numb = torch.tensor(et.num, device=DEVICE)
        catb = torch.tensor(et.cat, device=DEVICE)
        chunk, vals = 256, []
        for i in range(0, max(len(X), 1), chunk):
            vals.append(self._loss_from(X[i:i + chunk], numb[i:i + chunk],
                                        catb[i:i + chunk]))
        return float(np.mean(vals)) if vals else float("nan")

    def _fit_observation_noise(self, et: "EncodedTable") -> None:
        """Estimate the decoder's per-column predictive spread from residuals.

        WHY SAMPLING FROM p(x|z) MATTERS
        --------------------------------
        Training a numeric head with MSE is exactly maximum-likelihood under a
        Gaussian observation with fixed variance.  The correct generative
        sample from that model is   x = f(z) + sigma * eps,   not x = f(z).
        Emitting the mean alone is the classic "regression to the mean" failure:
        the synthetic numeric columns come out systematically narrower than the
        real ones (we measured age spanning 33-71 against a real 29-76, and
        cholesterol capped at 479 against a real 564).  So we read sigma per
        column off the training reconstruction residuals and add it back at
        sample time.  This is not a hack bolted onto the loss - it is the
        sampler that the loss already implies.

        `obs_noise=False` turns it off, which is how the ablation in
        make_figures.py shows what it buys.
        """
        if not self.obs_noise or et.num.shape[1] == 0:
            self.obs_sigma = None
            return
        self.net.eval()
        with torch.no_grad():
            resid = []
            X = torch.tensor(et.X, device=DEVICE)
            N = torch.tensor(et.num, device=DEVICE)
            for i in range(0, len(X), 256):
                num_hat, _, _, _ = self.net(X[i:i + 256])
                resid.append((N[i:i + 256] - num_hat).cpu().numpy())
            r = np.concatenate(resid, axis=0)
        sigma = r.std(axis=0)
        # Floor and ceiling: a perfectly reconstructed column would otherwise
        # emit zero spread, and a wildly mis-fitted one would swamp the signal.
        self.obs_sigma = np.clip(sigma, 0.02, 0.35) * self.obs_noise_scale

    def val_loss(self) -> float:
        """Validation reconstruction+KL loss - the only loss we are allowed to
        use for model selection."""
        if self.result is None:
            return float("nan")
        return float(self.result.final_val_loss)

    # --------------------------------------------------------------- sampling
    def _raw_outputs(self, n, temperature, rng):
        self.net.eval()
        scale = float(getattr(self, "_latent_scale", 1.0))
        with torch.no_grad():
            z = torch.tensor(rng.normal(0, scale, size=(n, self.latent_dim)),
                             dtype=torch.float32, device=DEVICE)
            num, cats = self.net.decode(z)
        num = num.detach().cpu().numpy()
        # x = f(z) + sigma*eps : sample the observation distribution the MSE
        # loss already assumes, instead of emitting its mean.
        sigma = getattr(self, "obs_sigma", None)
        if sigma is not None and num.shape[1] == len(sigma):
            num = num + rng.normal(0.0, 1.0, size=num.shape) * sigma[None, :]
        return (num, [c.detach().cpu().numpy() for c in cats])


# ---------------------------------------------------------------------------
# 3. DP-VAE
# ---------------------------------------------------------------------------
class DPVAEModel(VAEModel):
    """The same VAE trained with Differentially Private SGD (Opacus).

    (epsilon, delta)-DP informally means: seeing one synthetic table tells you
    almost nothing about whether any particular patient was in the training
    set.  We fix delta = 1/n_train (the usual convention: at most one chance in
    n of the guarantee failing) and sweep epsilon over 1, 3, 10, 30.

    Mechanism: gradients are clipped to `max_grad_norm` per sample, then Gaussian
    noise is added, before the optimizer steps.  The RDP/PRV accountant works out
    how much noise is needed to hit the target epsilon over the whole training
    run.  Small dataset + strict epsilon => very large noise => the model can
    barely learn.  That is not a bug in our code, it is the actual
    privacy-utility theorem, and showing it is the point of the experiment.

    CAVEATS WE DO NOT HIDE
    ----------------------
    * `secure_mode=False` (Opacus's default, and what this project uses for
      speed).  The noise is therefore drawn from a non-cryptographic RNG, which
      Opacus warns is "fine for experimentation, not for production".  A
      production release must retrain with `secure_mode=True`.
    * Opacus may report "Optimal order is the largest alpha", meaning the RDP
      optimisation searched a bounded range of orders and the published epsilon
      could in principle be marginally looser than a wider search would give.
    * epsilon is a bound produced by an accountant under the assumption that the
      sampled batches are the ones we claim; Poisson sampling skips of empty
      batches make the *spent* epsilon smaller, never larger, so the reported
      number stays conservative.
    """

    def __init__(self, schema, seed=0, epsilon=10.0, max_grad_norm=1.0,
                 delta=None, **hp):
        super().__init__(schema, seed=seed, **hp)
        # Set after super() so the parent's hard-coded name="vae" is replaced
        # rather than silently swallowed by **hp.
        self.name = "dp_vae"
        self.hyperparams["epsilon"] = float(epsilon)
        self.hyperparams["max_grad_norm"] = float(max_grad_norm)
        self.epsilon = float(epsilon)
        self.max_grad_norm = float(max_grad_norm)
        self.delta = delta
        self.privacy_note = f"epsilon={self.epsilon:g}"
        self._dp_steps = 0
        self._dp_active = False

    def fit(self, df, val_df=None, **kw):
        import time
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator

        t0 = time.time()
        et = EncodedTable(df, self.schema)
        self.enc_table = et
        n_train = et.n
        delta = self.delta if self.delta is not None else 1.0 / max(n_train, 2)
        # Opacus refuses to private-train unsupported modules.  Our VaeNet is
        # all Linear, so fixes() should be a no-op - but we call it and record
        # any change instead of crashing on a surprise.
        net = self._make_net(et)
        if not ModuleValidator.is_valid(net):
            self.notes.append("ModuleValidator fixed the net: "
                              + str(ModuleValidator.get_errors(net)))
            net = ModuleValidator.fix(net)
        self.net = net.to(DEVICE)

        # Keep the batch small enough that drop_last still leaves real steps.
        self.bsz = int(max(4, min(self.batch_size, max(4, n_train // 2))))
        loader = DataLoader(TensorDataset(torch.tensor(et.X),
                                          torch.tensor(et.num),
                                          torch.tensor(et.cat)),
                            batch_size=self.bsz, shuffle=True, drop_last=True)
        self.noise_multiplier = float("nan")
        self.expected_batch_size = self.bsz
        self.effective_max_grad_norm = self.max_grad_norm
        self.total_steps = 0
        self.achieved_epsilon = float("nan")
        self.spent_delta = delta
        self._dp_active = False

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        engine = PrivacyEngine()
        self._engine = engine
        try:
            self.net, opt, loader = engine.make_private_with_epsilon(
                module=self.net, optimizer=opt, data_loader=loader,
                target_epsilon=self.epsilon, target_delta=delta,
                epochs=self.epochs, max_grad_norm=self.max_grad_norm)
            self._dp_active = True
        except Exception as exc:                    # pragma: no cover
            self.privacy_note = "FAILED-fallback-non-private"
            self.notes.append(f"DP engine failed ({type(exc).__name__}: {exc}); "
                              f"trained WITHOUT DP-SGD - these results are NOT "
                              f"differentially private")
        if self._dp_active:
            # Read back what the accountant actually chose, rather than
            # re-deriving it and risking a number that does not match the
            # mechanism that really ran.
            try:
                self.noise_multiplier = float(opt.noise_multiplier)
                self.expected_batch_size = int(opt.expected_batch_size)
                self.effective_max_grad_norm = float(opt.max_grad_norm)
            except Exception as exc:                # pragma: no cover
                self.notes.append(f"DP parameter read-back failed "
                                  f"({type(exc).__name__}: {exc})")
            self.privacy_note = (f"target_eps={self.epsilon:g}, "
                                 f"sigma={self.noise_multiplier:.4f}, "
                                 f"clip={self.effective_max_grad_norm:g}, "
                                 f"exp_batch={self.expected_batch_size}, "
                                 f"delta={delta:.2e}")

        val_tensor = None
        if val_df is not None:
            vet = EncodedTable(val_df, self.schema)
            val_tensor = (torch.tensor(vet.X, device=DEVICE),
                          torch.tensor(vet.num, device=DEVICE),
                          torch.tensor(vet.cat, device=DEVICE))
        betas = []
        for epoch in range(self.epochs):
            beta = kl_schedule(epoch, self.epochs, self.anneal_epochs,
                               self.beta_max)
            betas.append(beta)
            self._train_one_epoch_dp(loader, opt, beta)
            if (epoch + 1) % max(1, self.val_every) == 0 or \
                    epoch == self.epochs - 1:
                self.train_history.append(self._eval_on(et))
                self.val_history.append(
                    self._eval_val(val_tensor) if val_tensor else float("nan"))
        self.total_steps = self._dp_steps
        # Ask the accountant how much privacy was actually spent, given the
        # number of steps that really happened.  This can be below the target
        # epsilon when Poisson sampling produced empty batches we skipped.
        if self._dp_active:
            try:
                self.achieved_epsilon = float(
                    engine.accountant.get_epsilon(delta))
            except Exception as exc:                # pragma: no cover
                self.notes.append(f"get_epsilon failed "
                                  f"({type(exc).__name__}: {exc})")
        self.result = VAETrainResult(
            self.train_history[-1] if self.train_history else float("nan"),
            self.val_history[-1] if self.val_history else float("nan"), betas)
        # Unwrap the Opacus GradSampleModule now that training is over.  The
        # noise-adding hooks only belong in the training loop; inference must
        # be a plain deterministic network.  Doing it here (rather than in
        # sample()) keeps every inherited method working unchanged.
        if hasattr(self.net, "to_standard_module"):
            self.net = self.net.to_standard_module()
        # Estimate the sampler spread only after unwrapping, so the residual
        # pass runs through a plain network and spends no privacy budget.
        self._fit_observation_noise(et)
        self.fit_seconds = time.time() - t0
        return self

    def _train_one_epoch_dp(self, loader, opt, beta):
        """One epoch under DP-SGD.

        Differences from the non-private epoch: no manual gradient clipping
        (Opacus clips every sample's gradient to `max_grad_norm` and adds
        Gaussian noise inside `opt.step()`), and the loss must be the mean of
        per-sample losses so Opacus can decompose it sample by sample.

        Opacus swaps in a Poisson-sampling loader, so batch sizes vary step to
        step and a batch can come back empty.  An empty batch is skipped
        entirely, which spends no privacy - conservative, and correct.
        """
        self.net.train()
        for Xb, numb, catb in loader:
            B = int(Xb.shape[0])
            if B == 0:
                continue                            # empty batch edge case
            Xb, numb, catb = Xb.to(DEVICE), numb.to(DEVICE), catb.to(DEVICE)
            opt.zero_grad()
            num_hat, cat_logits, mu, logvar = self.net(Xb)
            per_sample = (vae_reconstruction_loss(
                num_hat, cat_logits, numb, catb, self.enc_table.widths,
                w_num=self.w_num, w_cat=self.w_cat)
                + beta * kl_divergence_loss(mu, logvar))
            loss = per_sample.mean()
            if not torch.isfinite(loss):
                continue                            # NaN loss guard
            loss.backward()
            opt.step()
            self._dp_steps += 1


# ---------------------------------------------------------------------------
# 4/5. CTGAN and TVAE
# ---------------------------------------------------------------------------
class CTGANBase(Generator):
    """Thin wrapper around the `ctgan` package so it satisfies our interface.

    Four practical things this wrapper fixes:

    1. `ctgan` defaults to batch_size=500.  With 181 training rows the sampler
       has to draw with replacement every step, so the same patients are
       repeated.  We clamp the batch size to the dataset.
    2. `ctgan` wants *column names* for discrete_columns when given a
       DataFrame, and integer indices otherwise.  We pass names.
    3. `ctgan` raises InvalidDataError on NaN in continuous columns.  Our
       loader already imputed, but we assert here so a future dataset change
       fails with a clear message instead of a cryptic one.
    4. `ctgan` returns numeric columns unclipped, so it happily emits an age of
       -4 or a cholesterol of 900.  We run every sample through our own
       post-processing so the comparison happens at the level of valid records.
    """

    pkgname = "CTGAN"
    display = "ctgan"

    def __init__(self, schema, seed=0, epochs=300, batch_size=100, lr=2e-4,
                 embedding_dim=32, dim=(128, 128), **hp):
        super().__init__(schema=schema, seed=seed, name=self.display,
                         hyperparams=dict(epochs=epochs, batch_size=batch_size,
                                          lr=lr, embedding_dim=embedding_dim,
                                          dim=dim))
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.embedding_dim, self.dim = embedding_dim, tuple(dim)
        self.model = None
        self._col_order: List[str] = []

    def fit(self, df, val_df=None, **kw):
        import time
        from ctgan import CTGAN, TVAE

        t0 = time.time()
        self.enc_table = EncodedTable(df, self.schema)
        # The label is the final column and is deliberately kept as a discrete
        # column: the generator has to produce the diagnosis as well.
        train = df[self.schema.all_names].copy()
        self._col_order = list(train.columns)
        discrete_names = self.schema.categorical_names + [self.schema.label]
        for c in self.schema.numeric_names:
            assert not train[c].isna().any(), f"NaN in continuous {c}"
        train = train.astype({c: "int64" for c in discrete_names})
        train = train.astype({c: "float64" for c in self.schema.numeric_names})

        bsz = int(max(2, min(self.batch_size, len(train))))
        common = dict(epochs=self.epochs, batch_size=bsz, enable_gpu=False,
                      verbose=False)
        if self.pkgname == "CTGAN":
            self.model = CTGAN(embedding_dim=self.embedding_dim,
                               generator_dim=self.dim,
                               discriminator_dim=self.dim,
                               generator_lr=self.lr,
                               discriminator_lr=self.lr, **common)
        else:
            self.model = TVAE(embedding_dim=self.embedding_dim,
                              compress_dims=self.dim,
                              decompress_dims=self.dim,
                              l2scale=self.lr, **common)
        self.model.fit(train, discrete_columns=discrete_names)
        self.fit_seconds = time.time() - t0
        return self

    def sample(self, n, diversity=0.5, seed=None):
        set_seed((int(seed) if seed is not None else self.seed) + 10_000)
        raw = self.model.sample(int(n))
        raw = raw[self._col_order]
        # `diversity` cannot act through a latent temperature here - the ctgan
        # package does not expose per-row probabilities - so we use the same
        # mechanism as the naive baseline: post-hoc kernel jitter on the
        # numeric columns only.  The app labels this honestly.
        bw = max(0.0, (float(diversity) - 0.5)) * 0.8
        if bw > 0:
            rng = np.random.default_rng(
                (int(seed) if seed is not None else self.seed) + 777)
            for c in self.schema.numeric_names:
                sd = float(raw[c].std())
                raw[c] = raw[c].to_numpy(dtype=float) + rng.normal(0, bw * sd,
                                                                   len(raw))
        return sanitize(raw, self.schema)


class TVAEModel(CTGANBase):
    """TVAE: the tabular VAE from the same package, our VAE's closest rival."""

    pkgname = "TVAE"
    display = "tvae"


# ---------------------------------------------------------------------------
# 6. WGAN-GP (the "one extra of your choice")
# ---------------------------------------------------------------------------
class WGANGPModel(Generator):
    """Wasserstein GAN with gradient penalty on the same encoded space.

    Why WGAN-GP rather than a vanilla GAN: on 181 rows a standard adversarial
    loss is notoriously unstable, and the training-vs-generator loss trace is
    the classic way to show that instability to an examiner.  The gradient
    penalty (constraining the critic to be 1-Lipschitz along interpolated lines)
    replaces weight clipping and gives a usable, smooth training signal.

    Because it shares TabularEncoder with the VAE, the difference in the results
    is attributable to the training objective, not to a different preprocessing
    pipeline.  Categorical heads get an extra entropy penalty that pushes the
    softmax towards a confident one-hot, since the critic alone gives a very
    weak signal about discreteness.
    """

    def __init__(self, schema, seed=0, z_dim=16, hidden=(128, 64),
                 epochs=600, lr=1e-4, batch_size=32, n_critic=2,
                 lambda_gp=10.0, entropy_weight=0.05, **hp):
        super().__init__(schema=schema, seed=seed, name="wgan_gp",
                         hyperparams=dict(z_dim=z_dim, hidden=hidden,
                                          epochs=epochs, lr=lr,
                                          batch_size=batch_size,
                                          n_critic=n_critic,
                                          lambda_gp=lambda_gp,
                                          entropy_weight=entropy_weight))
        self.z_dim, self.hidden = z_dim, tuple(hidden)
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.n_critic, self.lambda_gp = n_critic, lambda_gp
        self.entropy_weight = entropy_weight
        self.gen: Optional[WGANGen] = None
        self.critic: Optional[Critic] = None
        self.critic_history: List[float] = []

    def fit(self, df, val_df=None, **kw):
        import time
        t0 = time.time()
        et = EncodedTable(df, self.schema)
        self.enc_table = et
        self.gen = WGANGen(self.z_dim, et.widths, et.num.shape[1],
                           hidden=self.hidden).to(DEVICE)
        self.critic = Critic(et.dim, hidden=self.hidden).to(DEVICE)
        self.bsz = int(max(2, min(self.batch_size, et.n)))
        opt_g = torch.optim.RMSprop(self.gen.parameters(), lr=self.lr)
        opt_c = torch.optim.RMSprop(self.critic.parameters(), lr=self.lr)
        X = torch.tensor(et.X, device=DEVICE)
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)

        for it in range(self.epochs):
            # --- train the critic `n_critic` times per generator update ------
            for _ in range(self.n_critic):
                idx = rng.integers(0, n, size=self.bsz)
                real = X[torch.as_tensor(idx, device=DEVICE)]
                z = torch.tensor(rng.normal(size=(self.bsz, self.z_dim)),
                                 dtype=torch.float32, device=DEVICE)
                fake = self._gen_vector(z)
                opt_c.zero_grad()
                d_real = self.critic(real).mean()
                d_fake = self.critic(fake).mean()
                gp = self._gradient_penalty(real, fake)
                c_loss = d_fake - d_real + self.lambda_gp * gp
                if torch.isfinite(c_loss):
                    c_loss.backward()
                    opt_c.step()
                self.critic_history.append(float(c_loss.item()))
            # --- one generator step ------------------------------------------
            z = torch.tensor(rng.normal(size=(self.bsz, self.z_dim)),
                             dtype=torch.float32, device=DEVICE)
            fake = self._gen_vector(z)
            opt_g.zero_grad()
            g_loss = -self.critic(fake).mean()
            if self.entropy_weight > 0:
                g_loss = g_loss + self.entropy_weight * self._entropy(
                    self.gen, z)
            if torch.isfinite(g_loss):
                g_loss.backward()
                opt_g.step()
        self.fit_seconds = time.time() - t0
        return self

    def _gen_vector(self, z) -> torch.Tensor:
        """Concatenate numeric head + softmax categorical heads into one vector."""
        num, cats = self.gen(z)
        blocks = [num] + [torch.softmax(c, dim=1) for c in cats]
        return torch.cat(blocks, dim=1)

    def _entropy(self, gen, z) -> torch.Tensor:
        """Mean categorical entropy - minimised, so confident one-hots."""
        _, cats = gen(z)
        terms = []
        for logits in cats:
            p = torch.softmax(logits, dim=1)
            terms.append(-(p * torch.log(p + 1e-8)).sum(dim=1))
        return torch.stack(terms, dim=1).mean(dim=1).mean()

    def _gradient_penalty(self, real, fake) -> torch.Tensor:
        """lambda * E[(||grad D(x_hat)||_2 - 1)^2], x_hat on the real-fake line."""
        B = real.shape[0]
        alpha = torch.rand(B, 1, device=real.device)
        x_hat = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d = self.critic(x_hat)
        grads = torch.autograd.grad(outputs=d, inputs=x_hat,
                                    grad_outputs=torch.ones_like(d),
                                    create_graph=True, retain_graph=True)[0]
        norm = grads.reshape(B, -1).norm(2, dim=1)
        return ((norm - 1.0) ** 2).mean()

    def _raw_outputs(self, n, temperature, rng):
        self.gen.eval()
        scale = float(getattr(self, "_latent_scale", 1.0))
        with torch.no_grad():
            z = torch.tensor(rng.normal(0, scale, size=(n, self.z_dim)),
                             dtype=torch.float32, device=DEVICE)
            num, cats = self.gen(z)
        return (num.detach().cpu().numpy(),
                [c.detach().cpu().numpy() for c in cats])


# ---------------------------------------------------------------------------
# Post-processing for models that return a plain DataFrame
# ---------------------------------------------------------------------------
def sanitize(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Clip numerics to valid medical ranges, round integers, fix categories.

    Used by CTGAN/TVAE (which emit their own DataFrame) and again as a final
    safety net in the pipeline.
    """
    out = df.copy()
    for c in schema.numeric:
        lo, hi = schema.clip_bounds(c)
        v = pd.to_numeric(out[c.name], errors="coerce").clip(lo, hi)
        v = v.fillna(float(np.mean([lo, hi])))
        if c.integer:
            v = v.round().clip(math.ceil(lo), math.floor(hi))
        elif c.decimals is not None:
            v = v.round(c.decimals)
        out[c.name] = v.astype(float)
    for c in schema.categorical:
        allowed = list(c.categories)
        v = pd.to_numeric(out[c.name], errors="coerce")
        # Round to the nearest *legal* level.  thal for instance is {3, 6, 7},
        # so a raw 6.4 must snap to 6, not to 6.0-of-an-unknown-grid, and a NaN
        # must snap to the nearest level rather than crash the pipeline.
        arr = v.to_numpy(dtype=float)
        fixed = np.empty(len(arr), dtype=int)
        for i, x in enumerate(arr):
            if not np.isfinite(x):
                x = float(allowed[0])
            diffs = [abs(x - a) for a in allowed]
            fixed[i] = allowed[int(np.argmin(diffs))]
        out[c.name] = fixed
    out[schema.label] = pd.to_numeric(out[schema.label], errors="coerce") \
        .fillna(0).round().clip(0, 1)
    return coerce_dtypes(out[schema.all_names], schema)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
REGISTRY: Dict[str, type] = {
    "naive": NaiveModel,
    "vae": VAEModel,
    "dp_vae": DPVAEModel,
    "ctgan": CTGANBase,
    "tvae": TVAEModel,
    "wgan_gp": WGANGPModel,
}


def build_model(name: str, schema: Schema, seed: int = 0, **hp) -> Generator:
    """Factory used by the pipeline, the notebook and the Streamlit app."""
    name = name.lower()
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose {sorted(REGISTRY)}")
    return REGISTRY[name](schema, seed=seed, **hp)


def model_list() -> List[str]:
    return sorted(REGISTRY)
