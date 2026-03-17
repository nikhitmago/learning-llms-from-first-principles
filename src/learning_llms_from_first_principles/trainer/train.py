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


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> tuple[GPTModel, list[float], list[float], list[float]]:

    print("\n" + "=" * 50)
    print("🚀 INITIALIZING PRE-TRAINING PIPELINE")
    print("=" * 50)

    print("\n[1/5] Loading Configuration...")
    print("-" * 30)
    print(OmegaConf.to_yaml(cfg))

    print("\n[2/5] Initializing Model...")
    print("-" * 30)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    print(f"Model Name: {cfg.model.name}")
    print_model_parameters(model)

    # Load data
    print("\n[3/5] Processing Dataset...")
    print("-" * 30)
    with open(cfg.data.file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # Split data
    train_data, val_data, _ = split_data(
        text_data, train_ratio=cfg.data.train_ratio, val_ratio=cfg.data.val_ratio
    )
    print(f"Source file: {cfg.data.file_path}")
    print(f"Training data size:   {len(train_data)} chars")
    print(f"Validation data size: {len(val_data)} chars")

    # Create loaders
    print("\n[4/5] Creating Dataloaders...")
    print("-" * 30)
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
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Device setup
    print("\n[5/5] Finalizing Environment...")
    print("-" * 30)
    device = get_device()
    print(f"Device detected: {device}")
    model.to(device)
    print("Model moved to device.")

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    print(f"Optimizer: AdamW (lr={cfg.training.lr}, weight_decay={cfg.training.weight_decay})")

    print("\n" + "=" * 50)
    print("✨ STARTING TRAINING LOOP")
    print("=" * 50 + "\n")

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
    )

    print("\n" + "=" * 50)
    print("✅ Training complete.")
    print("=" * 50 + "\n")

    return model, train_losses, val_losses, lrs


if __name__ == "__main__":
    main()
