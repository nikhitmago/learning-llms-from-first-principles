# Learning LLMs from First Principles

This repository contains code and experiments for understanding LLMs from first principles. Huge thanks to Sebastian Raschka for the original book and codebase.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nikhitmago/learning-llms-from-first-principles.git
cd learning-llms-from-first-principles
```

### 2. Python Environment

I recommend using a Conda environment with Python 3.10.

```bash
# Create the environment
conda create -n llm-first-principles python=3.10

# Activate the environment
conda activate llm-first-principles
```

### 3. Install Project

Install the project in **editable mode** to enable the command-line tools and allows imports from anywhere in the codebase.

```bash
pip install -e .
```

### 4. Development & Quality Check

You can verify your environment and run core health checks using `hatch`:

```bash
# Auto-format and sort imports
hatch run format

# Run full health check (linting + tests)
hatch run all

# Run only tests
hatch run test
```

---

## Pre-Training

The project provides a production-ready entry point for training. You can run pre-training directly from your terminal using the `pretrain` command.

### Usage

The trainer uses **Hydra** for configuration management. All parameters in [**`src/.../config/train.yaml`**](file:///Users/nikmag/Documents/learning-llms-from-first-principles/src/learning_llms_from_first_principles/config/train.yaml) can be overridden from the CLI.

```bash
# Run with default settings
pretrain

# Use a specific training config file
pretrain --config-name train

# Override model or training parameters
pretrain model.name=gpt2-large training.lr=0.001 training.num_epochs=10
```

---

## Classification Fine-Tuning

Fine-tune a pretrained GPT model for spam/ham text classification (based on Chapter 6 of the book). This requires a pretrained model checkpoint from the pre-training step above.

### Usage

Configuration lives in [`src/.../config/classify.yaml`](src/learning_llms_from_first_principles/config/classify.yaml). The spam dataset (SMS Spam Collection, balanced to 1494 samples) is shipped with the package.

```bash
# Run with default settings (LoRA enabled by default)
classify

# Disable LoRA and use standard fine-tuning
classify lora.enabled=false

# Customize LoRA rank and alpha
classify lora.rank=8 lora.alpha=8

# Override training parameters
classify training.num_epochs=10 training.lr=1e-4
```

LoRA (Low-Rank Adaptation) is enabled by default. Instead of updating all 124M parameters, it freezes the pretrained weights and injects small trainable adapter matrices (A and B) into every Linear layer — typically training only ~2-3% of the total parameters. After training, the adapters are saved separately and then merged back into the base weights for zero-overhead inference.

When LoRA is disabled, the pipeline falls back to the standard approach: freezing all layers except the last transformer block, final layer norm, and the classification head.

---

## Project Structure

- **`src/`**: Core logic and model architecture.
    - **`artifacts/`**: Bundled data files (text corpora, CSV datasets).
    - **`config/`**: Centralized YAML configurations (Hydra).
    - **`data/`**: Dataset classes and dataloaders.
    - **`trainer/`**: Training engines and CLI entry points (`pretrain`, `classify`).
    - **`utils/`**: General helper functions (loss, device setup, etc.).
    - **`modules/`**: Neural network components (Attention, Transformers, etc.).
- **`tests/`**: Unit and end-to-end tests.
- **`notebooks/`**: Exploratory scratchpads and tutorials.

---