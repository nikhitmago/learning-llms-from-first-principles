import logging
from pathlib import Path

import hydra
import tiktoken
import torch
from omegaconf import DictConfig, OmegaConf

from learning_llms_from_first_principles.config import GPT_CONFIG_124M
from learning_llms_from_first_principles.data.dataloader import create_classify_dataloader
from learning_llms_from_first_principles.data.datasets import Split
from learning_llms_from_first_principles.modules.gpt import GPTModel
from learning_llms_from_first_principles.utils.classify_utils import (
    calc_accuracy_loader,
    train_classifier,
)
from learning_llms_from_first_principles.utils.gpu_utils import get_device
from learning_llms_from_first_principles.utils.model_utils import print_model_parameters
from learning_llms_from_first_principles.utils.peft_utils import (
    merge_lora_weights,
    replace_linear_with_lora,
    save_lora_adapters,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="classify")
def main(cfg: DictConfig) -> GPTModel:
    logger.info("\n" + "=" * 50)
    logger.info("🚀 INITIALIZING CLASSIFICATION FINE-TUNING PIPELINE")
    logger.info("=" * 50)

    logger.info("\n[1/5] Loading Configuration...")
    logger.info("-" * 30)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # ------------------------------------------------------------------ #
    # [2/5] Model                                                          #
    # ------------------------------------------------------------------ #
    logger.info("\n[2/5] Initializing Model...")
    logger.info("-" * 30)

    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()
    logger.info(f"Device detected: {device}")

    model = GPTModel(GPT_CONFIG_124M)

    # Load pretrained weights — required for meaningful fine-tuning
    pretrained_path = Path(cfg.model.pretrained_path).expanduser()
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Pretrained weights not found at: {pretrained_path}. "
            "Classification fine-tuning requires pretrained weights."
        )
    logger.info(f"Loading pretrained weights from: {pretrained_path}")
    model.load_state_dict(torch.load(pretrained_path, weights_only=True))

    for param in model.parameters():
        param.requires_grad = False

    use_lora = cfg.lora.enabled

    if use_lora:
        logger.info(f"\n🔧 LoRA enabled (rank={cfg.lora.rank}, alpha={cfg.lora.alpha})")
        logger.info("Before LoRA:")
        print_model_parameters(model)

        replace_linear_with_lora(model, rank=cfg.lora.rank, alpha=cfg.lora.alpha)

    num_classes: int = cfg.model.num_classes
    model.out_head = torch.nn.Linear(
        in_features=int(GPT_CONFIG_124M["emb_dim"]), out_features=num_classes
    )

    if not use_lora:
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True

    if use_lora:
        logger.info("\nAfter LoRA + classification head:")

    model.to(device)
    logger.info(f"Model Name: {cfg.model.name} | Classification head: {num_classes} classes")
    print_model_parameters(model)

    # ------------------------------------------------------------------ #
    # [3/5] Dataloaders                                                    #
    # ------------------------------------------------------------------ #
    logger.info("\n[3/5] Creating Dataloaders...")
    logger.info("-" * 30)

    csv_file = str(Path(cfg.data.file_path).expanduser())

    train_loader, train_dataset = create_classify_dataloader(
        csv_file=csv_file,
        split_name=Split.TRAIN,
        tokenizer=tokenizer,
        max_length=None,  # auto-detect from training split
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
    )

    val_loader, _ = create_classify_dataloader(
        csv_file=csv_file,
        split_name=Split.VAL,
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,  # keep same padding as train
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
    )

    test_loader, _ = create_classify_dataloader(
        csv_file=csv_file,
        split_name=Split.TEST,
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
    )

    logger.info(f"Train batches: {len(train_loader)} | Max seq length: {train_dataset.max_length}")
    logger.info(f"Val batches:   {len(val_loader)}")
    logger.info(f"Test batches:  {len(test_loader)}")

    # ------------------------------------------------------------------ #
    # [4/5] Training                                                       #
    # ------------------------------------------------------------------ #
    logger.info("\n[4/5] Fine-Tuning Classifier...")
    logger.info("-" * 30)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    logger.info(
        f"Optimizer: AdamW (lr={cfg.training.lr}, weight_decay={cfg.training.weight_decay})"
    )

    logger.info("\n" + "=" * 50)
    logger.info("✨ STARTING TRAINING LOOP")
    logger.info("=" * 50 + "\n")

    model, train_losses, val_losses, train_accs, val_accs = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=cfg.training.num_epochs,
        eval_freq=cfg.training.eval_freq,
        eval_iter=cfg.training.eval_iter,
    )

    # ------------------------------------------------------------------ #
    # [5/5] Evaluation & Save                                              #
    # ------------------------------------------------------------------ #
    logger.info("\n[5/5] Final Evaluation on Test Set...")
    logger.info("-" * 30)

    test_acc = calc_accuracy_loader(test_loader, model, device)
    logger.info(f"Test accuracy: {test_acc * 100:.2f}%")

    logger.info("\n" + "=" * 50)
    logger.info("✅ Fine-tuning complete.")
    logger.info("=" * 50 + "\n")

    if use_lora:
        if cfg.lora.save_adapter_path:
            adapter_path = Path(cfg.lora.save_adapter_path).expanduser()
            adapter_path.parent.mkdir(parents=True, exist_ok=True)
            save_lora_adapters(model, str(adapter_path))
            logger.info(f"💾 LoRA adapters saved to: {adapter_path}")

        if cfg.lora.merge_after_training:
            merge_lora_weights(model)
            logger.info("🔀 LoRA weights merged into base model")
            print_model_parameters(model)

    if cfg.model.save_model_path:
        save_path = Path(cfg.model.save_model_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"💾 Model state dict saved to: {save_path}")

    return model


if __name__ == "__main__":
    main()
