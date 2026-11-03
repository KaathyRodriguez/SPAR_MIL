# coding=utf-8
"""
SPAR-MIL model.
(Semantic Prototype-guided Attention-Refined Multiple Instance Learning)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import math
import warnings
from os.path import join as pjoin
from .model_utils import *
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import pandas as pd

_tokenizer = _Tokenizer()


# =============================================================
# Utilities
# =============================================================
def _encode_token_length(tokenizer, name):
    """Encode class name length; compatible with both CLIP and CONCH tokenizers."""
    try:
        return len(tokenizer.encode(name, max_length=127, truncation=True))
    except TypeError:
        return len(tokenizer.encode(name))


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# =============================================================
# Bipartite matching utilities (for cross-scale contrastive)
# =============================================================
def _greedy_bipartite_match(cost: torch.Tensor):
    """Greedy bipartite matching; approximates Hungarian for small P."""
    P_a, P_b = cost.shape
    P_min = min(P_a, P_b)
    INF = float("inf")
    row_ind_list, col_ind_list = [], []
    cost_work = cost.clone()
    for _ in range(P_min):
        flat_idx = cost_work.argmin()
        r = (flat_idx // P_b).item()
        c = (flat_idx %  P_b).item()
        row_ind_list.append(r)
        col_ind_list.append(c)
        cost_work[r, :] = INF
        cost_work[:, c] = INF
    row_ind = torch.tensor(row_ind_list, device=cost.device)
    col_ind = torch.tensor(col_ind_list, device=cost.device)
    return row_ind, col_ind


def _bipartite_match_smart(cost: torch.Tensor):
    """Use scipy (Hungarian) when available; fall back to greedy otherwise."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost.detach().cpu().numpy())
        return torch.tensor(r, device=cost.device), torch.tensor(c, device=cost.device)
    except Exception:
        return _greedy_bipartite_match(cost)


# =============================================================
# Auxiliary losses
# =============================================================
def prototype_contrastive_loss_v2(z_low: torch.Tensor,
                                  z_high: torch.Tensor,
                                  temperature: float = 0.1) -> torch.Tensor:
    """Cross-scale contrastive loss (bipartite-matched InfoNCE)."""
    if z_low.size(0) == 0 or z_high.size(0) == 0:
        return torch.tensor(0.0, device=z_low.device)

    a = F.normalize(z_low,  dim=-1)
    b = F.normalize(z_high, dim=-1)
    sim = a @ b.T
    P_low, P_high = sim.shape

    if P_low == P_high:
        row_ind = torch.arange(P_low, device=sim.device)
        col_ind = torch.arange(P_high, device=sim.device)
    else:
        row_ind, col_ind = _bipartite_match_smart(-sim)

    pos_low  = a[row_ind]
    pos_high = b[col_ind]
    P_min = pos_low.size(0)
    if P_min <= 1:
        return torch.tensor(0.0, device=sim.device)

    sim_paired = pos_low @ pos_high.T / temperature
    labels = torch.arange(P_min, device=sim.device)
    return (F.cross_entropy(sim_paired, labels) +
            F.cross_entropy(sim_paired.T, labels)) / 2.0


def prototype_diversity_loss(proto_feats: torch.Tensor) -> torch.Tensor:
    """Penalize pairwise cosine similarity within a single scale to avoid collapse."""
    if proto_feats.size(0) <= 1:
        return torch.tensor(0.0, device=proto_feats.device)
    p = F.normalize(proto_feats, dim=-1)
    sim = p @ p.T
    P = sim.size(0)
    mask = ~torch.eye(P, dtype=torch.bool, device=sim.device)
    return (sim[mask] ** 2).mean()


def prototype_margin_loss(image_feat: torch.Tensor,
                          text_feat: torch.Tensor,
                          label: torch.Tensor,
                          margin: float = 0.3) -> torch.Tensor:
    """Encourage sim(image, pos_text) - sim(image, neg_text) >= margin."""
    if image_feat.size(0) == 0:
        return torch.tensor(0.0, device=image_feat.device)

    image_norm = F.normalize(image_feat, dim=-1)
    text_norm  = F.normalize(text_feat,  dim=-1)

    pos_text = text_norm[label]
    neg_text = text_norm[1 - label]

    pos_sim = (image_norm * pos_text).sum(dim=-1)
    neg_sim = (image_norm * neg_text).sum(dim=-1)

    return F.relu(margin - (pos_sim - neg_sim)).mean()


# =============================================================
# Text encoder wrapper
# =============================================================
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        text_module = getattr(clip_model, "text", clip_model)
        self.transformer = text_module.transformer
        self.positional_embedding = text_module.positional_embedding
        self.ln_final = text_module.ln_final
        self.text_projection = text_module.text_projection
        self.dtype = next(clip_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


# =============================================================
# Learnable prompt learner
# =============================================================
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, tokenizer=None, tokenize_fn=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = next(clip_model.parameters()).dtype
        text_module = getattr(clip_model, "text", clip_model)
        ctx_dim = text_module.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.image_size
        self.tokenizer = tokenizer if tokenizer is not None else _tokenizer
        self.tokenize_fn = tokenize_fn if tokenize_fn is not None else clip.tokenize

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = self.tokenize_fn([ctx_init])
            with torch.no_grad():
                embedding = text_module.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [_encode_token_length(self.tokenizer, name) for name in classnames]
        prompts = [name for name in classnames]
        tokenized_prompts = self.tokenize_fn(prompts)
        with torch.no_grad():
            embedding = text_module.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1:-n_ctx, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts


# =============================================================
# Per-sample scale router (Gumbel-Softmax)
# =============================================================
class ScaleRouter(nn.Module):
    """Per-sample dynamic alpha router; returns fusion weights [B, 2]."""

    def __init__(self,
                 dim: int,
                 temperature: float = 1.0,
                 use_gumbel: bool = True,
                 hard_gumbel: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2)  # [w_low, w_high]
        )
        self.temperature = temperature
        self.use_gumbel  = use_gumbel
        self.hard_gumbel = hard_gumbel

    def forward(self,
                image_feat_low: torch.Tensor,
                image_feat_high: torch.Tensor,
                training: bool = True):
        ctx = torch.cat([image_feat_low, image_feat_high], dim=-1)
        logits_alpha = self.context_proj(ctx)

        if self.training and self.use_gumbel:
            weights = F.gumbel_softmax(
                logits_alpha, tau=self.temperature,
                hard=self.hard_gumbel, dim=-1
            )
        else:
            weights = F.softmax(logits_alpha / self.temperature, dim=-1)
        return weights


# =============================================================
# Prototype GNN (relation modeling among prototypes)
# =============================================================
class PrototypeGNN(nn.Module):
    """TransformerEncoder acting as a GNN over prototypes."""

    def __init__(self, dim, num_layers=2, num_heads=4, dropout=0.1, ffn_factor=2):
        super().__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * ffn_factor,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, proto_feats: torch.Tensor) -> torch.Tensor:
        x = proto_feats.unsqueeze(0)
        x = self.encoder(x)
        x = self.norm_out(x)
        return x.squeeze(0)


# =============================================================
# Main model: SPAR-MIL
# =============================================================
class SPAR_MIL_Model(nn.Module):
    """SPAR-MIL for WSI binary classification.

    Total loss = CE + margin + cross-scale contrastive + diversity.
    """

    def __init__(self, config, num_classes=2):
        super(SPAR_MIL_Model, self).__init__()

        assert num_classes == 2, (
            f"SPAR-MIL targets binary WSI classification; "
            f"num_classes must be 2, got {num_classes}"
        )

        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.L = config.input_size
        self.raw_L = getattr(config, 'raw_input_size', self.L)
        self.D = config.hidden_size
        self.K = 1

        if self.raw_L != self.L:
            self.input_proj = nn.Linear(self.raw_L, self.L)
        else:
            self.input_proj = None

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        # Text backend
        self.text_backend = getattr(config, "text_backend", "open_clip")
        if self.text_backend == "conch":
            from CONCH.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
            conch_model_cfg = getattr(config, "conch_model_cfg", "conch_ViT-B-16")
            conch_checkpoint_path = getattr(config, "conch_checkpoint_path", None)
            if not conch_checkpoint_path:
                raise ValueError("conch backend requires config.conch_checkpoint_path")
            clip_model, _ = create_model_from_pretrained(conch_model_cfg, conch_checkpoint_path)
            tokenizer = get_tokenizer()
            tokenize_fn = lambda texts, _tokenizer=tokenizer: tokenize(_tokenizer, texts)
        else:
            from open_clip import create_model_and_transforms
            model_name = 'ViT-B-32'
            pretrained_dataset = 'laion2b_s34b_b79k'
            clip_model, _, _ = create_model_and_transforms(
                model_name, pretrained=pretrained_dataset, device=torch.device('cpu')
            )
            tokenizer = _tokenizer
            tokenize_fn = clip.tokenize

        self.clip_model = clip_model.float()
        self.prompt_learner = PromptLearner(
            config.text_prompt, self.clip_model,
            tokenizer=tokenizer, tokenize_fn=tokenize_fn,
        )
        self.text_encoder = TextEncoder(self.clip_model)

        # Validate text_prompt length
        expected_prompt_len = num_classes * 2
        if len(config.text_prompt) != expected_prompt_len:
            raise ValueError(
                f"Binary classification requires {expected_prompt_len} prompts "
                f"(low + high, {num_classes} each); got {len(config.text_prompt)}: "
                f"{config.text_prompt}"
            )

        # Prototype prompts
        prototype_prompt = (
            getattr(config, "prototype_prompt", None)
            or getattr(config, "prototype_prompt_csv", None)
        )
        if not prototype_prompt:
            raise ValueError("SPAR_MIL_Model requires config.prototype_prompt")

        prototype_df = pd.read_csv(prototype_prompt)
        low_proto_prompts  = prototype_df[prototype_df["scale"] == "low"]["prompt"].tolist()
        high_proto_prompts = prototype_df[prototype_df["scale"] == "high"]["prompt"].tolist()
        self.low_proto_num  = len(low_proto_prompts)
        self.high_proto_num = len(high_proto_prompts)
        self.low_proto_tokens  = tokenize_fn(low_proto_prompts)
        self.high_proto_tokens = tokenize_fn(high_proto_prompts)
        self.register_buffer("low_proto_tokenized_prompts",  self.low_proto_tokens)
        self.register_buffer("high_proto_tokenized_prompts", self.high_proto_tokens)

        self.norm = nn.LayerNorm(config.input_size)
        # Must use project-local MultiheadAttention (from model_utils):
        # it accepts 2D key/value (M: [N, L]); nn.MultiheadAttention requires 3D.
        self.cross_attention_1 = MultiheadAttention(embed_dim=config.input_size, num_heads=1)
        self.cross_attention_2 = MultiheadAttention(embed_dim=config.input_size, num_heads=1)

        # Gated residual fusion
        self.gate_low_linear  = nn.Linear(self.L * 2, 1)
        self.gate_high_linear = nn.Linear(self.L * 2, 1)

        # Lazily-built learnable prototype queries
        self.proto_perturb_scale = getattr(config, "proto_perturb_scale", 1.0)
        self.use_gated_residual = getattr(config, "use_gated_residual", True)
        self.use_norm_anchor     = getattr(config, "use_norm_anchor", True)
        self.learnable_low_query  = None
        self.learnable_high_query = None

        # Prototype GNN
        gnn_layers  = getattr(config, "gnn_num_layers", 0)
        gnn_heads   = getattr(config, "gnn_num_heads", 4)
        gnn_ffn     = getattr(config, "gnn_ffn_factor", 2)
        gnn_dropout = getattr(config, "gnn_dropout", 0.1)

        if not isinstance(gnn_layers, int) or gnn_layers < 0:
            raise ValueError(f"gnn_num_layers must be a non-negative integer, got {gnn_layers}")

        self.gnn_num_layers  = gnn_layers
        self.gnn_num_heads   = gnn_heads
        self.gnn_ffn_factor  = gnn_ffn
        self.gnn_dropout     = gnn_dropout

        self.use_prototype_gnn = (
            getattr(config, "use_prototype_gnn", True) and (gnn_layers > 0)
        )

        if self.use_prototype_gnn:
            self.prototype_gnn_low  = PrototypeGNN(
                self.L, gnn_layers, gnn_heads, gnn_dropout, gnn_ffn
            )
            self.prototype_gnn_high = PrototypeGNN(
                self.L, gnn_layers, gnn_heads, gnn_dropout, gnn_ffn
            )
            logger.info(
                f"[PrototypeGNN] enabled: layers={gnn_layers}, heads={gnn_heads}, "
                f"ffn_factor={gnn_ffn}, dropout={gnn_dropout}"
            )
        else:
            self.prototype_gnn_low  = None
            self.prototype_gnn_high = None
            logger.info(
                f"[PrototypeGNN] disabled: gnn_num_layers={gnn_layers}, "
                f"use_prototype_gnn={getattr(config, 'use_prototype_gnn', True)}"
            )

        # Auxiliary loss weights
        self.loss_weight_contrastive = getattr(config, "loss_weight_contrastive", 0.1)
        self.loss_weight_diversity   = getattr(config, "loss_weight_diversity",   0.05)
        self.contrastive_temperature = getattr(config, "contrastive_temperature", 0.1)
        self.margin_loss_weight      = getattr(config, "margin_loss_weight",      0.1)
        self.margin_value            = getattr(config, "margin_value",            0.3)

        # Top-k aggregation
        self.use_topk_aggregation = getattr(config, "use_topk_aggregation", True)
        self.topk_ratio           = getattr(config, "topk_ratio", 0.7)

        # Scale router
        router_temp    = getattr(config, "router_temperature", 1.0)
        use_gumbel     = getattr(config, "use_gumbel",         True)
        hard_gumbel    = getattr(config, "hard_gumbel",        False)
        router_dropout = getattr(config, "router_dropout",     0.1)
        self.scale_router = ScaleRouter(
            dim=self.L,
            temperature=router_temp,
            use_gumbel=use_gumbel,
            hard_gumbel=hard_gumbel,
            dropout=router_dropout,
        )

        # Ablation switches
        self.use_scale_router = getattr(config, "use_scale_router", True)
        self.router_mode      = getattr(config, "router_mode", "gumbel")
        self.static_w_low     = getattr(config, "static_w_low", 0.5)

        self.use_contrastive_loss = getattr(config, "use_contrastive_loss", True)
        self.use_diversity_loss   = getattr(config, "use_diversity_loss",   True)
        self.use_margin_loss      = getattr(config, "use_margin_loss",      True)

        self._ablation_keys = {
            "use_scale_router":     self.use_scale_router,
            "router_mode":          self.router_mode,
            "use_prototype_gnn":    self.use_prototype_gnn,
            "use_contrastive_loss": self.use_contrastive_loss,
            "use_diversity_loss":   self.use_diversity_loss,
            "use_margin_loss":      self.use_margin_loss,
        }

    def encode_text(self, tokenized_prompts):
        text_module = self.text_encoder
        clip_text_module = getattr(self.clip_model, "text", self.clip_model)
        with torch.no_grad():
            text_embedding = clip_text_module.token_embedding(tokenized_prompts).type(text_module.dtype)
        x = text_embedding + text_module.positional_embedding.type(text_module.dtype)
        x = x.permute(1, 0, 2)
        x = text_module.transformer(x)
        x = x.permute(1, 0, 2)
        x = text_module.ln_final(x).type(text_module.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ text_module.text_projection
        return x

    def _build_learnable_queries(self, device, dtype):
        if self.learnable_low_query is None:
            low_q = torch.empty(self.low_proto_num, 1, self.L, device=device, dtype=dtype)
            trunc_normal_(low_q, std=0.02)
            self.learnable_low_query = nn.Parameter(low_q)
        if self.learnable_high_query is None:
            high_q = torch.empty(self.high_proto_num, 1, self.L, device=device, dtype=dtype)
            trunc_normal_(high_q, std=0.02)
            self.learnable_high_query = nn.Parameter(high_q)

    def _norm_anchor(self, learnable_query, text_proto):
        """Rescale learnable_query norm to match text_proto to prevent semantic drift."""
        if not self.use_norm_anchor:
            return learnable_query
        text_norm  = text_proto.norm(dim=-1, keepdim=True)
        query_norm = learnable_query.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return learnable_query * (text_norm / query_norm)

    def _gated_mix(self, text_proto, learnable_query, gate_linear):
        """Sigmoid-gated fusion of text prototype and learnable query."""
        if not self.use_gated_residual:
            return text_proto + learnable_query
        gate_input = torch.cat([text_proto, learnable_query], dim=-1)
        gate = torch.sigmoid(gate_linear(gate_input))
        return gate * text_proto + (1.0 - gate) * learnable_query

    def _topk_aggregation(self, H, A_logits):
        """Top-k weighted aggregation over prototypes."""
        A = A_logits.squeeze(-1) if A_logits.dim() > 1 else A_logits
        A = F.softmax(A, dim=-1)

        P = A.size(0)
        top_k = max(1, int(self.topk_ratio * P))

        if top_k >= P:
            return (A.unsqueeze(-1) * H).sum(0).unsqueeze(0)

        topk_vals, topk_idx = torch.topk(A, top_k)
        mask = torch.zeros_like(A)
        mask[topk_idx] = 1.0
        A_masked = A * mask
        A_masked = A_masked / (A_masked.sum() + 1e-6)

        return (A_masked.unsqueeze(-1) * H).sum(0).unsqueeze(0)

    def forward(self, x_s, coord_s, x_l, coords_l, label):
        # 1. Encode text prompts
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # 2. Encode semantic prototypes
        low_semantic_prototypes  = self.encode_text(self.low_proto_tokenized_prompts).unsqueeze(1)
        high_semantic_prototypes = self.encode_text(self.high_proto_tokenized_prompts).unsqueeze(1)

        self._build_learnable_queries(
            device=low_semantic_prototypes.device,
            dtype=low_semantic_prototypes.dtype,
        )

        # 3. Low-scale prototype-guided aggregation
        M = x_s.float()
        if self.input_proj is not None:
            M = self.input_proj(M)

        learnable_low = self.learnable_low_query.to(dtype=M.dtype) * self.proto_perturb_scale
        learnable_low = self._norm_anchor(learnable_low, low_semantic_prototypes)

        mixed_query_low = self._gated_mix(
            low_semantic_prototypes, learnable_low, self.gate_low_linear
        )
        compents, _ = self.cross_attention_1(mixed_query_low, M, M)
        compents = self.norm(compents + mixed_query_low)

        compents_low_flat = compents.squeeze(1)
        if self.use_prototype_gnn:
            compents_low_rel = self.prototype_gnn_low(compents_low_flat)
        else:
            compents_low_rel = compents_low_flat
        compents = compents_low_rel.unsqueeze(1)

        # 4. High-scale prototype-guided aggregation
        M_high = x_l.float()
        if self.input_proj is not None:
            M_high = self.input_proj(M_high)

        learnable_high = self.learnable_high_query.to(dtype=M_high.dtype) * self.proto_perturb_scale
        learnable_high = self._norm_anchor(learnable_high, high_semantic_prototypes)

        mixed_query_high = self._gated_mix(
            high_semantic_prototypes, learnable_high, self.gate_high_linear
        )
        compents_high, _ = self.cross_attention_1(mixed_query_high, M_high, M_high)
        compents_high = self.norm(compents_high + mixed_query_high)

        compents_high_flat = compents_high.squeeze(1)
        if self.use_prototype_gnn:
            compents_high_rel = self.prototype_gnn_high(compents_high_flat)
        else:
            compents_high_rel = compents_high_flat
        compents_high = compents_high_rel.unsqueeze(1)

        # 5. Low-scale ABMIL (top-k)
        H = compents.squeeze(1).float()
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A   = self.attention_weights(A_V * A_U)

        if self.use_topk_aggregation:
            image_features_low = self._topk_aggregation(H, A)
        else:
            A_t = F.softmax(A.squeeze(-1), dim=-1)
            image_features_low = (A_t.unsqueeze(-1) * H).sum(0)

        # 6. High-scale ABMIL (top-k)
        H_high = compents_high.squeeze(1).float()
        A_V_high = self.attention_V(H_high)
        A_U_high = self.attention_U(H_high)
        A_high   = self.attention_weights(A_V_high * A_U_high)

        if self.use_topk_aggregation:
            image_features_high = self._topk_aggregation(H_high, A_high)
        else:
            A_high_t = F.softmax(A_high.squeeze(-1), dim=-1)
            image_features_high = (A_high_t.unsqueeze(-1) * H_high).sum(0)

        # 7. Context-guided text enhancement (low-scale)
        text_features_low = text_features[:self.num_classes]
        image_context = torch.cat((compents.squeeze(1), M), dim=0)
        text_context_features, _ = self.cross_attention_2(
            text_features_low.unsqueeze(1), image_context, image_context
        )
        text_features_low = text_context_features.squeeze(1) + text_features_low

        # 8. Context-guided text enhancement (high-scale)
        text_features_high = text_features[self.num_classes:]
        image_context_high = torch.cat((compents_high.squeeze(1), M_high), dim=0)
        text_context_features_high, _ = self.cross_attention_2(
            text_features_high.unsqueeze(1), image_context_high, image_context_high
        )
        text_features_high = text_context_features_high.squeeze(1) + text_features_high

        # 9. Dual-scale logits
        logits_low  = image_features_low  @ text_features_low.T.to(image_features_low.device)
        logits_high = image_features_high @ text_features_high.T.to(image_features_high.device)

        # 10. Per-sample dynamic alpha routing
        device = logits_low.device
        B = image_features_low.size(0)

        if (not self.use_scale_router) or self.router_mode == "mean":
            router_weights = torch.full(
                (B, 2), 0.5, device=device, dtype=logits_low.dtype
            )
            logits = 0.5 * logits_low + 0.5 * logits_high
        elif self.router_mode == "static":
            w_l = float(self.static_w_low)
            w_h = 1.0 - w_l
            router_weights = torch.tensor(
                [[w_l, w_h]] * B, device=device, dtype=logits_low.dtype
            )
            logits = w_l * logits_low + w_h * logits_high
        else:
            router_weights = self.scale_router(
                image_features_low, image_features_high,
                training=self.training,
            )
            if self.router_mode == "softmax":
                ctx = torch.cat([image_features_low, image_features_high], dim=-1)
                logits_alpha = self.scale_router.context_proj(ctx)
                router_weights = F.softmax(
                    logits_alpha / self.scale_router.temperature, dim=-1
                )
            logits = (
                router_weights[:, 0:1] * logits_low +
                router_weights[:, 1:2] * logits_high
            )

        # 11. Auxiliary losses
        device = logits.device
        label = label.to(device)
        B = image_features_low.size(0)

        # 11.1 Cross-scale contrastive (bipartite InfoNCE)
        if (self.loss_weight_contrastive > 0
                and self.low_proto_num > 0
                and self.high_proto_num > 0
                and self.use_contrastive_loss):
            contrast_loss = prototype_contrastive_loss_v2(
                compents_low_flat.float(), compents_high_flat.float(),
                temperature=self.contrastive_temperature,
            )
        else:
            contrast_loss = torch.tensor(0.0, device=device)

        # 11.2 Prototype diversity
        if self.loss_weight_diversity > 0 and self.use_diversity_loss:
            diversity_loss = (
                prototype_diversity_loss(compents_low_flat.float()) +
                prototype_diversity_loss(compents_high_flat.float())
            )
        else:
            diversity_loss = torch.tensor(0.0, device=device)

        # 11.3 Prototype margin
        margin_loss_low  = torch.tensor(0.0, device=device)
        margin_loss_high = torch.tensor(0.0, device=device)
        if self.margin_loss_weight > 0 and B > 0 and self.use_margin_loss:
            margin_loss_low = prototype_margin_loss(
                image_features_low.float(), text_features_low.float(),
                label, self.margin_value
            )
            margin_loss_high = prototype_margin_loss(
                image_features_high.float(), text_features_high.float(),
                label, self.margin_value
            )

        # 12. Total loss
        ce_loss = self.loss_ce(logits, label)

        loss = (
            ce_loss
            + self.loss_weight_contrastive * contrast_loss
            + self.loss_weight_diversity   * diversity_loss
            + self.margin_loss_weight      * (margin_loss_low + margin_loss_high)
        )

        # 13. Prediction
        Y_prob = F.softmax(logits, dim=1)
        Y_hat  = torch.topk(Y_prob, 1, dim=1)[1]

        return Y_prob, Y_hat, loss

    def get_trainable_info(self):
        with torch.no_grad():
            info = {
                "router_temp":             self.scale_router.temperature,
                "use_gumbel":              self.scale_router.use_gumbel,
                "hard_gumbel":             self.scale_router.hard_gumbel,
                "topk_ratio":              self.topk_ratio,
                "use_topk_agg":            self.use_topk_aggregation,
                "use_norm_anchor":         self.use_norm_anchor,
                "loss_weight_contrastive": self.loss_weight_contrastive,
                "loss_weight_diversity":   self.loss_weight_diversity,
                "margin_loss_weight":      self.margin_loss_weight,
                "margin_value":            self.margin_value,
                "low_proto_num":           self.low_proto_num,
                "high_proto_num":          self.high_proto_num,
            }
            info.update(self._ablation_keys)
            if self.learnable_low_query is not None:
                info["learnable_low_norm"] = self.learnable_low_query.norm(dim=-1).mean().item()
            if self.learnable_high_query is not None:
                info["learnable_high_norm"] = self.learnable_high_query.norm(dim=-1).mean().item()
        return info