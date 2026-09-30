"""Model definition and Matryoshka Representation Learning (MRL) loss module.

Supports google/embeddinggemma-300m and google/gemma-2-2b with LoRA adaptation,
mean pooling, and nested dimensional loss wrapping MultipleNegativesRankingLoss.
"""

import logging
import os
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
try:
    from sentence_transformers.sentence_transformer.losses.matryoshka import ForwardDecorator
except ImportError:
    from sentence_transformers.losses.matryoshka import ForwardDecorator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mrl.model")


class DetailedMatryoshkaLoss(MatryoshkaLoss):
    """Subclass of MatryoshkaLoss that records and exposes per-dimension losses."""

    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module,
        matryoshka_dims: list[int],
        matryoshka_weights: list[float] | None = None,
        n_dims_per_step: int = -1,
    ):
        super().__init__(
            model=model,
            loss=loss,
            matryoshka_dims=matryoshka_dims,
            matryoshka_weights=matryoshka_weights,
            n_dims_per_step=n_dims_per_step,
        )
        self.latest_dim_losses: dict[int, float] = {}

    def forward(
        self,
        sentence_features: list[dict[str, torch.Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_forward = self.model.forward
        try:
            decorated_forward = ForwardDecorator(original_forward)
            self.model.forward = decorated_forward

            total_loss = 0.0
            for idx in range(len(self.matryoshka_dims)):
                dim = self.matryoshka_dims[idx]
                weight = self.matryoshka_weights[idx]
                decorated_forward.set_dim(dim)

                dim_labels = labels
                if (
                    isinstance(labels, torch.Tensor)
                    and labels.ndim == 2
                    and labels.size(-1) == self.model.get_embedding_dimension()
                ):
                    dim_labels = labels[:, :dim]

                dim_loss = self.loss(sentence_features, dim_labels)
                self.latest_dim_losses[dim] = dim_loss.detach().item()
                total_loss += weight * dim_loss

            return total_loss
        finally:
            self.model.forward = original_forward


def get_default_nested_dims(hidden_size: int) -> list[int]:
    """Determine nested Matryoshka dimensions matching model capacity."""
    if hidden_size >= 1024:
        return [1024, 512, 256, 128, 64]
    elif hidden_size >= 768:
        return [768, 512, 256, 128, 64]
    elif hidden_size >= 512:
        return [512, 256, 128, 64]
    else:
        return [hidden_size, 128, 64]


def build_bi_encoder(
    model_name_or_path: str = "google/embeddinggemma-300m",
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_length: int = 256,
    device: str = "mps",
) -> SentenceTransformer:
    """Build a bi-encoder SentenceTransformer with optional LoRA and MPS acceleration."""
    logger.info(f"Initializing bi-encoder from: {model_name_or_path} (use_lora={use_lora})")
    token = os.environ.get("HF_TOKEN")

    # Check if gated and fallback if needed
    if model_name_or_path == "google/embeddinggemma-300m":
        candidate_models = ["unsloth/embeddinggemma-300m", "google/embeddinggemma-300m"]
    else:
        candidate_models = [model_name_or_path]

    model = None
    last_err = None
    for m_id in candidate_models:
        # First try loading from local cache to avoid network retries in offline/sandbox mode
        try:
            model = SentenceTransformer(
                m_id,
                token=token,
                device=device,
                trust_remote_code=True,
                local_files_only=True,
            )
            logger.info(f"Loaded bi-encoder backbone from local cache: {m_id}")
            break
        except Exception:
            pass

        try:
            model = SentenceTransformer(
                m_id,
                token=token,
                device=device,
                trust_remote_code=True,
            )
            logger.info(f"Loaded bi-encoder backbone from {m_id}")
            break
        except Exception as e:
            last_err = e
            logger.warning(f"Could not load SentenceTransformer from {m_id}: {e}")

    if model is None:
        logger.info(f"Direct load failed. Assembling via Transformer + Mean Pooling for {candidate_models[-1]}...")
        chosen_backbone = candidate_models[-1]
        word_embedding_model = models.Transformer(
            chosen_backbone,
            model_args={"token": token, "trust_remote_code": True},
        )
        pooling_model = models.Pooling(
            word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model],
            device=device,
        )

    # Apply LoRA if requested or if using a large generative backbone (e.g. 2B+)
    if use_lora or "2b" in model_name_or_path.lower():
        logger.info("Configuring PEFT LoRA adapter for backbone...")
        transformer_module = model[0]
        # Target modules per spec: ["q_proj", "v_proj", "k_proj", "o_proj"]
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )
        transformer_module.auto_model = get_peft_model(transformer_module.auto_model, lora_config)
        transformer_module.auto_model.print_trainable_parameters()

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
        logger.info(f"Configured model max_seq_length to {max_seq_length}")

    return model


def build_mrl_loss(
    model: SentenceTransformer,
    nested_dims: list[int] | None = None,
) -> DetailedMatryoshkaLoss:
    """Construct DetailedMatryoshkaLoss wrapping MultipleNegativesRankingLoss with uniform weights."""
    hidden_size = model.get_sentence_embedding_dimension()
    if nested_dims is None:
        nested_dims = get_default_nested_dims(hidden_size)

    # Filter out dimensions exceeding base hidden size
    valid_dims = [int(d) for d in nested_dims if d <= hidden_size]
    if not valid_dims:
        valid_dims = [hidden_size]

    logger.info(f"Configuring MRL loss with nested dimensions: {valid_dims} (Base hidden size: {hidden_size})")

    # In-batch negatives InfoNCE loss
    base_loss = MultipleNegativesRankingLoss(model)

    # Uniform weights: L_total = sum_{d in D} L_InfoNCE(d)
    uniform_weights = [1.0] * len(valid_dims)

    mrl_loss = DetailedMatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=valid_dims,
        matryoshka_weights=uniform_weights,
    )
    return mrl_loss


def encode_mrl(
    model: SentenceTransformer,
    sentences: list[str] | str,
    dim: int = 128,
    batch_size: int = 64,
    show_progress_bar: bool = False,
) -> torch.Tensor:
    """Encode sentences and dynamically truncate and L2-normalize to target dimension."""
    # Obtain raw embeddings (untruncated)
    raw_embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_tensor=True,
        normalize_embeddings=False,
    )
    # Truncate to desired Matryoshka dimension and L2-normalize
    sliced = raw_embeddings[:, :dim]
    normalized = F.normalize(sliced, p=2, dim=1)
    return normalized
