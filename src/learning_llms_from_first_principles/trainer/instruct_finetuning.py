import json
import logging
from pathlib import Path

import hydra
import tiktoken
import torch
from omegaconf import DictConfig, OmegaConf

from learning_llms_from_first_principles.config import GPT_CONFIG_124M
from learning_llms_from_first_principles.data.dataloader import create_instruct_dataloader
from learning_llms_from_first_principles.inference.generate import generate_text
from learning_llms_from_first_principles.modules.gpt import GPTModel
from learning_llms_from_first_principles.utils.data_utils import format_instruct_prompt
from learning_llms_from_first_principles.utils.gpu_utils import get_device
from learning_llms_from_first_principles.utils.model_utils import print_model_parameters
from learning_llms_from_first_principles.utils.train_utils import (
    train_model_v1,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="instruct_finetuning")
def main(cfg: DictConfig) -> GPTModel:
    logger.info("\n" + "=" * 50)
    logger.info("🚀 INITIALIZING INSTRUCTION FINE-TUNING PIPELINE")
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

    pretrained_path = Path(cfg.model.pretrained_path).expanduser()
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Pretrained weights not found at: {pretrained_path}. "
            "Instruction fine-tuning requires pretrained weights."
        )
    logger.info(f"Loading pretrained weights from: {pretrained_path}")
    model.load_state_dict(torch.load(pretrained_path, weights_only=True))

    model.to(device)
    logger.info(f"Model Name: {cfg.model.name}")
    print_model_parameters(model)

    # ------------------------------------------------------------------ #
    # [3/5] Dataloaders                                                    #
    # ------------------------------------------------------------------ #
    logger.info("\n[3/5] Creating Dataloaders...")
    logger.info("-" * 30)

    data_path = Path(cfg.data.file_path).expanduser()
    with open(data_path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} instruction entries from {data_path}")

    train_portion = int(len(data) * cfg.data.train_ratio)
    test_portion = int(len(data) * cfg.data.test_ratio)

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    logger.info(f"Split: {len(train_data)} train | {len(val_data)} val | {len(test_data)} test")

    device_str = str(device)
    allowed_max_length = cfg.data.allowed_max_length

    train_loader = create_instruct_dataloader(
        data=train_data,
        tokenizer=tokenizer,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
        allowed_max_length=allowed_max_length,
        device=device_str,
    )

    val_loader = create_instruct_dataloader(
        data=val_data,
        tokenizer=tokenizer,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
        allowed_max_length=allowed_max_length,
        device=device_str,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches:   {len(val_loader)}")

    # ------------------------------------------------------------------ #
    # [4/5] Training                                                       #
    # ------------------------------------------------------------------ #
    logger.info("\n[4/5] Instruction Fine-Tuning...")
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

    model, train_losses, val_losses, lrs = train_model_v1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=cfg.training.num_epochs,
        eval_freq=cfg.training.eval_freq,
        tokenizer=tokenizer,
        instruction_fine_tuning_samples=[
            format_instruct_prompt(entry) + "\n\n### Response:\n" for entry in val_data[:2]
        ],
    )

    # ------------------------------------------------------------------ #
    # [5/5] Save                                                           #
    # ------------------------------------------------------------------ #
    logger.info("\n[5/5] Saving Model...")
    logger.info("-" * 30)

    logger.info("\n" + "=" * 50)
    logger.info("✅ Instruction fine-tuning complete.")
    logger.info("=" * 50 + "\n")

    if cfg.model.save_model_path:
        save_path = Path(cfg.model.save_model_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"💾 Model state dict saved to: {save_path}")

    # ------------------------------------------------------------------ #
    # Sample Inference                                                     #
    # ------------------------------------------------------------------ #
    logger.info("\n📝 Sample Inference:")
    logger.info("-" * 30)

    sample_entries = [
        {"instruction": "What is the capital of France?", "input": ""},
        {
            "instruction": "Summarize the following text.",
            "input": "Machine learning is a subset of AI.",
        },
        {"instruction": "Translate to Spanish.", "input": "Good morning"},
    ]

    for entry in sample_entries:
        prompt = format_instruct_prompt(entry) + "\n\n### Response:\n"
        response = generate_text(
            text=prompt,
            model=model,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            max_new_tokens=128,
            context_size=int(GPT_CONFIG_124M["context_len"]),
            temperature=0.0,
            device=device,
        )
        # Strip the prompt to show only the generated response
        generated = response[len(prompt) :]
        logger.info(f"\nInstruction: {entry['instruction']}")
        if entry["input"]:
            logger.info(f"Input: {entry['input']}")
        logger.info(f"Response: {generated.strip()}")

    return model


if __name__ == "__main__":
    main()
