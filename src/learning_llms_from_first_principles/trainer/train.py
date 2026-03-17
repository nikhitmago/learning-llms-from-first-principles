import logging
from pathlib import Path

import hydra
import tiktoken
import torch
from omegaconf import DictConfig, OmegaConf

from learning_llms_from_first_principles.config import GPT_CONFIG_124M
from learning_llms_from_first_principles.data.dataloader import create_dataloader_v1
from learning_llms_from_first_principles.modules.gpt import GPTModel
from learning_llms_from_first_principles.utils.data_utils import split_data
from learning_llms_from_first_principles.utils.gpu_utils import get_device
from learning_llms_from_first_principles.utils.model_utils import print_model_parameters
from learning_llms_from_first_principles.utils.train_utils import train_model_v1

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> tuple[GPTModel, list[float], list[float], list[float]]:

    logger.info("\n" + "=" * 50)
    logger.info("🚀 INITIALIZING PRE-TRAINING PIPELINE")
    logger.info("=" * 50)

    logger.info("\n[1/5] Loading Configuration...")
    logger.info("-" * 30)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    logger.info("\n[2/5] Initializing Model...")
    logger.info("-" * 30)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    logger.info(f"Model Name: {cfg.model.name}")
    print_model_parameters(model)

    # Load data
    logger.info("\n[3/5] Processing Dataset...")
    logger.info("-" * 30)
    with open(cfg.data.file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # Split data
    train_data, val_data, _ = split_data(
        text_data, train_ratio=cfg.data.train_ratio, val_ratio=cfg.data.val_ratio
    )
    logger.info(f"Source file: {cfg.data.file_path}")
    logger.info(f"Training data size:   {len(train_data)} chars")
    logger.info(f"Validation data size: {len(val_data)} chars")

    # Create loaders
    logger.info("\n[4/5] Creating Dataloaders...")
    logger.info("-" * 30)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=cfg.training.batch_size,
        max_length=cfg.training.max_length,
        stride=cfg.training.stride,
        drop_last=True,
        shuffle=True,
        num_workers=0,
        tokenizer=tokenizer,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=cfg.training.batch_size,
        max_length=cfg.training.max_length,
        stride=cfg.training.stride,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        tokenizer=tokenizer,
    )
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches:   {len(val_loader)}")

    # Device setup
    logger.info("\n[5/5] Finalizing Environment...")
    logger.info("-" * 30)
    device = get_device()
    logger.info(f"Device detected: {device}")
    model.to(device)
    logger.info("Model moved to device.")

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    logger.info(
        f"Optimizer: AdamW (lr={cfg.training.lr}, weight_decay={cfg.training.weight_decay})"
    )

    logger.info("\n" + "=" * 50)
    logger.info("✨ STARTING TRAINING LOOP")
    logger.info("=" * 50 + "\n")

    # Train
    model, train_losses, val_losses, lrs = train_model_v1(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=cfg.training.num_epochs,
        eval_freq=cfg.training.eval_freq,
        tokenizer=tokenizer,
        warmup_ratio=cfg.training.warmup_ratio,
        warmup_min_lr=cfg.training.warmup_min_lr,
        decay_floor_lr=cfg.training.decay_floor_lr,
        max_norm=cfg.training.max_norm,
    )

    logger.info("\n" + "=" * 50)
    logger.info("✅ Training complete.")
    logger.info("=" * 50 + "\n")

    if cfg.model.save_model_path:
        save_path = Path(cfg.model.save_model_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"💾 Model state dict saved to: {save_path}")

    return model, train_losses, val_losses, lrs


if __name__ == "__main__":
    main()
