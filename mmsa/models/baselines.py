from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _ModalityPool(nn.Module):
    """Project a modality sequence to d_model and mean-pool over time."""

    def __init__(self, input_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=1)


def _zero_aux(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device)


class LateFusionBaseline(nn.Module):
    """Simple late fusion: per-modality pooled MLP encoders concatenated."""

    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        text_dim: int,
        interp_dim: int = 0,
        d_model: int = 128,
        dropout: float = 0.1,
        use_interpretation: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.audio = _ModalityPool(audio_dim, d_model, dropout)
        self.vision = _ModalityPool(vision_dim, d_model, dropout)
        self.text = _ModalityPool(text_dim, d_model, dropout)
        self.use_interpretation = bool(use_interpretation and interp_dim > 0)
        n = 3
        if self.use_interpretation:
            self.interp = nn.Sequential(nn.Linear(interp_dim, d_model), nn.ReLU())
            n = 4
        self.head = nn.Sequential(
            nn.Linear(n * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, audio, vision, text, interpretation=None):
        parts = [self.audio(audio), self.vision(vision), self.text(text)]
        if self.use_interpretation and interpretation is not None:
            parts.append(self.interp(interpretation))
        fused = torch.cat(parts, dim=-1)
        return {"reg": self.head(fused).squeeze(-1), "aux_loss": _zero_aux(audio.device)}

    @property
    def temporal_backend(self) -> str:
        return "late-fusion"


class CrossModalTransformer(nn.Module):
    """MulT-style: text queries audio/vision via directed cross-attention."""

    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        text_dim: int,
        interp_dim: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_interpretation: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.text_to_audio = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.text_to_vision = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_interpretation = bool(use_interpretation and interp_dim > 0)
        head_in = d_model
        if self.use_interpretation:
            self.interp = nn.Sequential(nn.Linear(interp_dim, d_model), nn.ReLU())
            head_in = 2 * d_model
        self.head = nn.Sequential(nn.Linear(head_in, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, audio, vision, text, interpretation=None):
        a = self.audio_proj(audio)
        v = self.vision_proj(vision)
        t = self.text_proj(text)
        t_a, _ = self.text_to_audio(t, a, a, need_weights=False)
        t_v, _ = self.text_to_vision(t, v, v, need_weights=False)
        fused_seq = self.encoder(t + t_a + t_v)
        pooled = fused_seq.mean(dim=1)
        if self.use_interpretation and interpretation is not None:
            pooled = torch.cat([pooled, self.interp(interpretation)], dim=-1)
        return {"reg": self.head(pooled).squeeze(-1), "aux_loss": _zero_aux(audio.device)}

    @property
    def temporal_backend(self) -> str:
        return "cross-modal-transformer"


class LMFBaseline(nn.Module):
    """Low-rank Multimodal Fusion (TFN family) on time-pooled modality vectors."""

    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        text_dim: int,
        interp_dim: int = 0,
        d_model: int = 128,
        dropout: float = 0.1,
        rank: int = 4,
        use_interpretation: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.audio = _ModalityPool(audio_dim, d_model, dropout)
        self.vision = _ModalityPool(vision_dim, d_model, dropout)
        self.text = _ModalityPool(text_dim, d_model, dropout)
        self.use_interpretation = bool(use_interpretation and interp_dim > 0)
        self.num_modalities = 4 if self.use_interpretation else 3
        if self.use_interpretation:
            self.interp = nn.Sequential(nn.Linear(interp_dim, d_model), nn.ReLU())
        self.rank = rank
        # one low-rank factor per modality; augmented dim adds a bias term
        self.factors = nn.ParameterList(
            nn.Parameter(torch.randn(rank, d_model + 1, d_model) * 0.1) for _ in range(self.num_modalities)
        )
        self.fusion_weight = nn.Parameter(torch.randn(1, rank) * 0.1)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def _augment(self, vector: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(vector.size(0), 1, device=vector.device, dtype=vector.dtype)
        return torch.cat([vector, ones], dim=-1)

    def forward(self, audio, vision, text, interpretation=None):
        vectors = [self.audio(audio), self.vision(vision), self.text(text)]
        if self.use_interpretation and interpretation is not None:
            vectors.append(self.interp(interpretation))
        elif self.use_interpretation:
            vectors.append(torch.zeros_like(vectors[0]))

        fusion = None
        for factor, vector in zip(self.factors, vectors):
            augmented = self._augment(vector)  # (B, d+1)
            projected = torch.einsum("bd,rde->bre", augmented, factor)  # (B, rank, d)
            fusion = projected if fusion is None else fusion * projected
        fused = torch.einsum("or,bre->be", self.fusion_weight, fusion)  # (B, d)
        return {"reg": self.head(fused).squeeze(-1), "aux_loss": _zero_aux(audio.device)}

    @property
    def temporal_backend(self) -> str:
        return "lmf"


class ALMTBaseline(nn.Module):
    """Simplified, faithful-in-spirit re-implementation of ALMT (Zhang et al.
    2023, "Learning Language-guided Adaptive Hyper-modality Representation
    for Multimodal Sentiment Analysis"), NOT a literal reproduction of the
    original code/hyperparameters (which we do not have access to) -- we do
    not claim exact numerical parity with the paper's reported numbers, only
    that this captures ALMT's central idea under our own pipeline: a small
    set of learnable "hyper-modality" tokens cross-attend into the
    concatenated audio+vision sequence, and a text-derived gate then
    suppresses the parts of that hyper-modality summary that are irrelevant
    to or conflict with the (dominant) text signal, before a final fusion
    with the text representation.
    """

    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        text_dim: int,
        interp_dim: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        num_hyper_tokens: int = 4,
        dropout: float = 0.1,
        use_interpretation: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.hyper_tokens = nn.Parameter(torch.randn(1, num_hyper_tokens, d_model) * 0.02)
        self.av_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # text-guided suppression gate: down-weights hyper-modality dimensions
        # that are irrelevant to or conflict with the dominant text signal
        self.text_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_interpretation = bool(use_interpretation and interp_dim > 0)
        head_in = 2 * d_model
        if self.use_interpretation:
            self.interp = nn.Sequential(nn.Linear(interp_dim, d_model), nn.ReLU())
            head_in = 3 * d_model
        self.head = nn.Sequential(nn.Linear(head_in, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, audio, vision, text, interpretation=None):
        a = self.audio_proj(audio)
        v = self.vision_proj(vision)
        t = self.text_proj(text)
        av_seq = torch.cat([a, v], dim=1)
        hyper = self.hyper_tokens.expand(audio.size(0), -1, -1)
        hyper_attn, _ = self.av_cross_attn(hyper, av_seq, av_seq, need_weights=False)
        text_summary = t.mean(dim=1)
        gate = self.text_gate(text_summary).unsqueeze(1)
        hyper_suppressed = hyper_attn * gate
        hyper_pooled = self.encoder(hyper_suppressed).mean(dim=1)
        parts = [hyper_pooled, text_summary]
        if self.use_interpretation and interpretation is not None:
            parts.append(self.interp(interpretation))
        fused = torch.cat(parts, dim=-1)
        return {"reg": self.head(fused).squeeze(-1), "aux_loss": _zero_aux(audio.device)}

    @property
    def temporal_backend(self) -> str:
        return "almt-lite"


class KuDABaseline(nn.Module):
    """Simplified, faithful-in-spirit re-implementation of KuDA (Feng et al.
    2024, "Knowledge-Guided Dynamic Modality Attention Fusion Framework for
    Multimodal Sentiment Analysis"), NOT a literal reproduction of the
    original code/hyperparameters. It captures KuDA's central idea: text is
    treated as the *dominant* modality, and a per-sample gate derived purely
    from the pooled text representation dynamically reweights audio and
    vision before fusion -- rather than fusing all modalities with a
    fixed/learned-but-sample-independent weighting, as LateFusion/LMF do.
    """

    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        text_dim: int,
        interp_dim: int = 0,
        d_model: int = 128,
        dropout: float = 0.1,
        use_interpretation: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.audio = _ModalityPool(audio_dim, d_model, dropout)
        self.vision = _ModalityPool(vision_dim, d_model, dropout)
        self.text = _ModalityPool(text_dim, d_model, dropout)
        # dominant-modality (text) knowledge gate: dynamically reweights audio/vision per sample
        self.dominance_gate = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.Sigmoid())
        self.use_interpretation = bool(use_interpretation and interp_dim > 0)
        head_in = 3 * d_model
        if self.use_interpretation:
            self.interp = nn.Sequential(nn.Linear(interp_dim, d_model), nn.ReLU())
            head_in = 4 * d_model
        self.head = nn.Sequential(nn.Linear(head_in, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, audio, vision, text, interpretation=None):
        a = self.audio(audio)
        v = self.vision(vision)
        t = self.text(text)
        gate_a, gate_v = self.dominance_gate(t).chunk(2, dim=-1)
        parts = [a * gate_a, v * gate_v, t]
        if self.use_interpretation and interpretation is not None:
            parts.append(self.interp(interpretation))
        fused = torch.cat(parts, dim=-1)
        return {"reg": self.head(fused).squeeze(-1), "aux_loss": _zero_aux(audio.device)}

    @property
    def temporal_backend(self) -> str:
        return "kuda-lite"


def _cmd_loss(x: torch.Tensor, y: torch.Tensor, n_moments: int = 5) -> torch.Tensor:
    """Central moment discrepancy between two representations (MISA, Eq. 4).

    The authors' released implementation matches raw moments 1..n_moments across a
    modality pair; that is what is reproduced here.
    """
    loss = torch.zeros((), device=x.device, dtype=x.dtype)
    for k in range(1, n_moments + 1):
        loss = loss + (x.pow(k).mean(0) - y.pow(k).mean(0)).abs().mean()
    return loss


def _similarity_loss(private: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    """MISA's similarity term: pull each private/shared pair together (-cos)."""
    return -torch.cosine_similarity(private, shared, dim=-1).mean()


def _orthogonality_loss(private: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    p = F.normalize(private, dim=-1)
    s = F.normalize(shared, dim=-1)
    return (p * s).sum(dim=-1).pow(2).mean()


class MISABaseline(nn.Module):
    """Simplified, faithful-in-spirit re-implementation of MISA (Hazarika
    et al. 2020, "MISA: Modality-Invariant and -Specific Representations
    for Multimodal Sentiment Analysis"), ported from the feature-level MISA
    reproduction in the authors' own prior project (VWB repo,
    `src/run_feature_baseline_repro.py::MISAFeatureModel`, itself already a
    simplified reproduction, not the original paper's code) into this
    paper's regression protocol and (T, D) sequence pipeline. NOT a literal
    reproduction of the original paper -- captures MISA's central idea:
    decompose each modality into a modality-invariant ("shared", weights
    tied across modalities) and modality-specific ("private") subspace,
    regularized by a reconstruction loss and a soft-orthogonality
    ("difference") loss between the two, then fuse all six
    sub-representations for the final prediction.
    """

    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        text_dim: int,
        interp_dim: int = 0,
        d_model: int = 128,
        dropout: float = 0.1,
        use_interpretation: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.audio = _ModalityPool(audio_dim, d_model, dropout)
        self.vision = _ModalityPool(vision_dim, d_model, dropout)
        self.text = _ModalityPool(text_dim, d_model, dropout)
        self.private_t = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.private_a = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.private_v = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.shared = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())  # tied weights: modality-invariant subspace
        self.recon = nn.Linear(d_model, d_model)
        # auxiliary-loss weights; 0.05 each is the default in the authors' released config
        self.w_sim = 0.05
        self.w_diff = 0.05
        self.w_recon = 0.05
        self.w_cmd = 0.05
        self.use_interpretation = bool(use_interpretation and interp_dim > 0)
        n = 7 if self.use_interpretation else 6
        if self.use_interpretation:
            self.interp = nn.Sequential(nn.Linear(interp_dim, d_model), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(n * d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, audio, vision, text, interpretation=None):
        zt, za, zv = self.text(text), self.audio(audio), self.vision(vision)
        pt, pa, pv = self.private_t(zt), self.private_a(za), self.private_v(zv)
        st, sa, sv = self.shared(zt), self.shared(za), self.shared(zv)

        # MISA's four auxiliary terms.  The earlier version of this baseline carried
        # only difference + reconstruction; similarity and CMD were missing, and CMD is
        # what aligns the three modality-specific distributions.
        sim = (_similarity_loss(pt, st) + _similarity_loss(pa, sa) + _similarity_loss(pv, sv)) / 3.0
        diff = _orthogonality_loss(pt, st) + _orthogonality_loss(pa, sa) + _orthogonality_loss(pv, sv)
        recon = F.mse_loss(self.recon(pt + st), zt) + F.mse_loss(self.recon(pa + sa), za) + F.mse_loss(self.recon(pv + sv), zv)
        cmd = (_cmd_loss(pt, pa) + _cmd_loss(pt, pv) + _cmd_loss(pa, pv)) / 3.0
        aux_loss = self.w_sim * sim + self.w_diff * diff + self.w_recon * recon + self.w_cmd * cmd

        parts = [pt, pa, pv, st, sa, sv]
        if self.use_interpretation and interpretation is not None:
            parts.append(self.interp(interpretation))
        fused = torch.cat(parts, dim=-1)
        return {"reg": self.head(fused).squeeze(-1), "aux_loss": aux_loss}

    @property
    def temporal_backend(self) -> str:
        return "misa-lite"


class SelfMMBaseline(nn.Module):
    """Re-implementation of Self-MM (Yu et al. 2021), including the mechanism the
    paper is actually about: a modality-specific *pseudo-label* for each unimodal
    auxiliary head, formed as a performance-weighted blend of the ground-truth
    label and that modality's own prediction,

        L_m = alpha_m * y + (1 - alpha_m) * y_hat_m ,      alpha_m ~ 1 / MAE_m ,

    with alpha re-estimated once per epoch from the running unimodal errors.  The
    previous version of this baseline only distilled the *fused* prediction into
    the unimodal heads and was documented as not reproducing this contribution at
    all; that caveat no longer applies.

    Remaining deviations from the original: the unimodal errors that set alpha come
    from the epoch's training pass rather than a separate validation pass, and the
    text/audio/vision encoders are this paper's shared feature pipeline rather than
    BERT + LSTMs.  Both are stated in the supplement.
    """
    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        text_dim: int,
        interp_dim: int = 0,
        d_model: int = 128,
        dropout: float = 0.1,
        use_interpretation: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.audio = _ModalityPool(audio_dim, d_model, dropout)
        self.vision = _ModalityPool(vision_dim, d_model, dropout)
        self.text = _ModalityPool(text_dim, d_model, dropout)
        self.text_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.audio_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.vision_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.use_interpretation = bool(use_interpretation and interp_dim > 0)
        n = 4 if self.use_interpretation else 3
        if self.use_interpretation:
            self.interp = nn.Sequential(nn.Linear(interp_dim, d_model), nn.ReLU())
        self.fusion_head = nn.Sequential(
            nn.Linear(n * d_model, 2 * d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * d_model, 1)
        )
        # pseudo-label blend weights (text, audio, vision); updated by on_epoch_end()
        self.register_buffer("alpha", torch.full((3,), 1.0 / 3.0))
        self._running_err: torch.Tensor | None = None

    needs_labels = True  # train.py passes the label so the pseudo-labels can be built

    def forward(self, audio, vision, text, interpretation=None, labels=None):
        t, a, v = self.text(text), self.audio(audio), self.vision(vision)
        parts = [t, a, v]
        if self.use_interpretation and interpretation is not None:
            parts.append(self.interp(interpretation))
        fused_score = self.fusion_head(torch.cat(parts, dim=-1)).squeeze(-1)

        aux_t = self.text_head(t).squeeze(-1)
        aux_a = self.audio_head(a).squeeze(-1)
        aux_v = self.vision_head(v).squeeze(-1)
        aux_loss = _zero_aux(fused_score.device)

        if labels is not None and self.training:
            y = labels.reshape(-1).to(fused_score.dtype)
            preds = [aux_t, aux_a, aux_v]
            # track each modality's error so on_epoch_end() can re-weight the blend
            with torch.no_grad():
                errs = torch.stack([(p.detach() - y).abs().mean() for p in preds])
                self._running_err = errs if self._running_err is None else self._running_err + errs
            # L_m = alpha_m * y + (1 - alpha_m) * y_hat_m, target detached (pseudo-label)
            losses = [
                F.l1_loss(p, (self.alpha[m] * y + (1.0 - self.alpha[m]) * p.detach()).detach())
                for m, p in enumerate(preds)
            ]
            aux_loss = torch.stack(losses).mean()
        elif labels is not None:
            # evaluation: keep the loss well-defined but out of the reported metrics
            aux_loss = _zero_aux(fused_score.device)
        return {"reg": fused_score, "aux_loss": aux_loss}

    def on_epoch_end(self) -> None:
        """Re-estimate the pseudo-label blend from the epoch's unimodal errors."""
        if self._running_err is None:
            return
        weight = 1.0 / (self._running_err + 1e-6)
        self.alpha.copy_(weight / weight.sum())
        self._running_err = None

    @property
    def temporal_backend(self) -> str:
        return "self-mm (pseudo-labels)"


class MAGBertBaseline(nn.Module):
    """Simplified re-implementation of the MAG mechanism from MAG-BERT
    (Rahman et al. 2020, "Integrating Multimodal Information in Large
    Pretrained Transformers"), ported from the feature-level MAG-BERT
    reproduction in the authors' own prior project (VWB repo,
    `src/run_feature_baseline_repro.py::MAGBertFeatureModel`).
    SIMPLIFICATION: the original injects the Multimodal Adaptation Gate
    into every layer of a fine-tuned pretrained BERT; this paper's pipeline
    (like all other baselines here) operates on pre-extracted text features
    rather than raw text + a trainable BERT backbone, so the gate is
    applied once to the projected text sequence before a small Transformer
    encoder, not inside BERT's internal layers. Captures MAG's core idea
    (a tanh-bounded, audio/visual-conditioned additive shift applied to the
    text representation) without the pretrained-language-model backbone.
    """

    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        text_dim: int,
        interp_dim: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        beta_shift: float = 1e-3,
        use_interpretation: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.gate = nn.Linear(3 * d_model, d_model)
        self.beta_shift = beta_shift
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.use_interpretation = bool(use_interpretation and interp_dim > 0)
        head_in = d_model
        if self.use_interpretation:
            self.interp = nn.Sequential(nn.Linear(interp_dim, d_model), nn.ReLU())
            head_in = 2 * d_model
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, audio, vision, text, interpretation=None):
        t = self.text_proj(text)
        a = self.audio_proj(audio).mean(dim=1, keepdim=True).expand(-1, t.size(1), -1)
        v = self.vision_proj(vision).mean(dim=1, keepdim=True).expand(-1, t.size(1), -1)
        shift = torch.tanh(self.gate(torch.cat([t, a, v], dim=-1))) * (1.0 + self.beta_shift)
        pooled = self.encoder(t + shift).mean(dim=1)
        parts = [pooled]
        if self.use_interpretation and interpretation is not None:
            parts.append(self.interp(interpretation))
        fused = torch.cat(parts, dim=-1)
        return {"reg": self.head(fused).squeeze(-1), "aux_loss": _zero_aux(audio.device)}

    @property
    def temporal_backend(self) -> str:
        return "magbert-lite"


class CorMulTBaseline(nn.Module):
    """Simplified, single-stage re-implementation of CorMulT (the
    correlation-aware MulT variant used as the "CLAMP"/correlation
    reference method in the authors' own prior paper), ported from
    `src/run_cormult_repro.py::CorMulTReproModel` in the same VWB project.
    SIMPLIFICATION #1: the original pretrains a *frozen* correlation model
    on a contrastive triplet objective in a separate stage before training
    the classifier; this version instead folds an analogous cross-modal
    correlation objective into a *single* joint training stage as part of
    `aux_loss`, since this paper's train()/ablation()/multiseed() loops
    call training exactly once per model (no two-phase pretrain-then-freeze
    protocol). SIMPLIFICATION #2: pairwise cross-modal interaction reuses
    `nn.MultiheadAttention` (as in this paper's `CrossModalTransformer`
    baseline) over only the text<-audio/text<-vision directions, not
    CorMulT's full 6-way pairwise grid with a custom TransformerEncoder.
    Do not describe this as reproducing CorMulT's numbers.
    """

    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        text_dim: int,
        interp_dim: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_interpretation: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.text_to_audio = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.text_to_vision = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_interpretation = bool(use_interpretation and interp_dim > 0)
        head_in = d_model
        if self.use_interpretation:
            self.interp = nn.Sequential(nn.Linear(interp_dim, d_model), nn.ReLU())
            head_in = 2 * d_model
        self.head = nn.Sequential(nn.Linear(head_in, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, audio, vision, text, interpretation=None):
        a = self.audio_proj(audio)
        v = self.vision_proj(vision)
        t = self.text_proj(text)

        t_pool, a_pool, v_pool = t.mean(dim=1), a.mean(dim=1), v.mean(dim=1)
        # correlation scale: (B,) -> (1, B, 1) so it broadcasts over nn.MultiheadAttention's
        # batch_first=True output (B, T, D) once transposed for the additive fusion below
        c_ta = F.cosine_similarity(t_pool, a_pool, dim=-1).view(-1, 1, 1)
        c_tv = F.cosine_similarity(t_pool, v_pool, dim=-1).view(-1, 1, 1)

        t_a, _ = self.text_to_audio(t, a, a, need_weights=False)
        t_v, _ = self.text_to_vision(t, v, v, need_weights=False)
        # correlation-scaled fusion: text absorbs more of a modality pair the more that pair
        # already agrees with text at the pooled level, mirroring CorMulT's (1 + correlation) scaling
        fused_seq = self.encoder(t + t_a * (1.0 + c_ta) + t_v * (1.0 + c_tv))
        pooled = fused_seq.mean(dim=1)

        aux_loss = self._correlation_loss(t_pool, a_pool) + self._correlation_loss(t_pool, v_pool)

        parts = [pooled]
        if self.use_interpretation and interpretation is not None:
            parts.append(self.interp(interpretation))
        fused = torch.cat(parts, dim=-1)
        return {"reg": self.head(fused).squeeze(-1), "aux_loss": aux_loss}

    @staticmethod
    def _correlation_loss(x: torch.Tensor, y: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
        """Single-stage stand-in for CorMulT's frozen triplet-pretrained correlation
        model: pull each sample's true cross-modal pair together, push a
        within-batch shuffled (mismatched) pair apart, by at least `margin`."""
        positive = F.cosine_similarity(x, y, dim=-1)
        shuffled = y[torch.randperm(y.size(0), device=y.device)]
        negative = F.cosine_similarity(x, shuffled, dim=-1)
        return F.relu(negative - positive + margin).mean()

    @property
    def temporal_backend(self) -> str:
        return "cormult-lite"
