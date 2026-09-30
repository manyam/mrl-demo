"""Training pipeline for Matryoshka bi-encoder on Apple Silicon MPS.

Trains the model using DetailedMatryoshkaLoss wrapping MultipleNegativesRankingLoss,
tracks per-dimension losses, and retains the top 2 checkpoints.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from transformers import TrainerCallback
from src.models.mrl_model import build_bi_encoder, build_mrl_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mrl.train")


def verify_mps_device():
    """Verify Apple Silicon Metal Performance Shaders (MPS) availability."""
    if not torch.backends.mps.is_available():
        error_msg = (
            "FATAL: Apple Silicon MPS acceleration is not available! "
            "PyTorch must be built with MPS support (torch.backends.mps.is_available() == True)."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    logger.info("Device verified: Apple Silicon Metal Performance Shaders (MPS) is active.")


def load_pairs_dataset(jsonl_path: Path, max_samples: int | None = None) -> Dataset:
    """Load anchor-positive pairs from JSON Lines into Hugging Face Dataset."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    anchors = []
    positives = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            anchors.append(record["anchor"])
            positives.append(record["positive"])
            if max_samples and len(anchors) >= max_samples:
                break

    logger.info(f"Loaded {len(anchors)} pairs from {jsonl_path.name}")
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


class MatryoshkaLossLoggingCallback(TrainerCallback):
    """Callback to log individual Matryoshka dimension losses during training and eval."""

    def __init__(self, mrl_loss_fn):
        super().__init__()
        self.mrl_loss_fn = mrl_loss_fn

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and hasattr(self.mrl_loss_fn, "latest_dim_losses"):
            dim_losses = self.mrl_loss_fn.latest_dim_losses
            if dim_losses:
                dim_str = ", ".join([f"dim_{d}: {loss:.4f}" for d, loss in sorted(dim_losses.items(), reverse=True)])
                logger.info(f"Step {state.global_step} | Per-dimension losses: [{dim_str}]")
                for d, loss in dim_losses.items():
                    logs[f"loss_dim_{d}"] = loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MRL bi-encoder on Apple Silicon MPS.")
    parser.add_argument("--model_name_or_path", type=str, default="google/embeddinggemma-300m", help="Base model backbone")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory with processed jsonl data")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 2e-5 for full, 1e-4 for LoRA)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Linear warmup ratio")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Cap training samples for quick runs")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--use_lora", action="store_true", help="Apply PEFT LoRA adapter")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision on MPS if supported")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # 1. Device Verification (Strict Acceptance Criterion)
    verify_mps_device()
    device = torch.device("mps")

    # 2. Determine Learning Rate based on specification
    is_lora = args.use_lora or "2b" in args.model_name_or_path.lower()
    lr = args.lr if args.lr is not None else (1e-4 if is_lora else 2e-5)
    logger.info(f"Using learning rate: {lr} (is_lora={is_lora})")

    # 3. Load Data
    data_dir = Path(args.data_dir)
    train_dataset = load_pairs_dataset(data_dir / "train.jsonl", max_samples=args.max_train_samples)
    val_dataset = load_pairs_dataset(data_dir / "val.jsonl", max_samples=args.max_train_samples // 10 if args.max_train_samples else None)

    # 4. Build Model & MRL Loss
    model = build_bi_encoder(
        model_name_or_path=args.model_name_or_path,
        use_lora=is_lora,
        max_seq_length=args.max_seq_length,
        device=str(device),
    )
    mrl_loss = build_mrl_loss(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = int(args.max_steps * args.warmup_ratio) if args.max_steps > 0 else 50

    step_interval = max(25, args.max_steps // 3) if args.max_steps > 0 else 50

    # 5. Training Arguments configured per spec
    # Retain top 2 checkpoints based on eval loss
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        fp16=args.fp16,
        max_steps=args.max_steps,
        logging_steps=max(5, step_interval // 5),
        eval_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=step_interval if args.max_steps > 0 else None,
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=step_interval if args.max_steps > 0 else None,
        save_total_limit=2,  # Retain only top 2 performing checkpoints
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        dataloader_pin_memory=False,
        seed=args.seed,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=mrl_loss,
        callbacks=[MatryoshkaLossLoggingCallback(mrl_loss)],
    )

    logger.info("Starting MRL fine-tuning on Apple Silicon MPS...")
    train_result = trainer.train()
    logger.info(f"Training completed. Global steps: {train_result.global_step}, Training loss: {train_result.training_loss:.4f}")

    # 6. Save final best model
    best_model_dir = output_dir / "best_model"
    model.save_pretrained(str(best_model_dir))
    logger.info(f"Top-performing model checkpoint saved to: {best_model_dir}")


if __name__ == "__main__":
    main()
