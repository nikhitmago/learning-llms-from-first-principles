# Learning LLMs from First Principles

This repository contains code and experiments for understanding LLMs from first principles. Huge thanks to Sebastian Raschka for the original book and codebase.

## Setup Instructions

To set up your development environment, follow these steps:

### 1. Python Environment

I recommend using a Conda environment with Python 3.10.

```bash
# Create the environment
conda create -n llm-first-principles python=3.10

# Activate the environment
conda activate llm-first-principles
```

### 2. Install Project in Editable Mode

Install the required libraries and the project itself in "editable" mode. This allows you to import from project subdirectories (like `data/` or `models/`) from anywhere.

```bash
pip install -e .
```

> **Note:** Because this is an editable install, any changes you make to the code are immediately available without needing to re-install!

### 3. Verify & Run

You can verify your environment and run the core quality checks:

```bash
# Auto-format and sort imports
hatch run format

# Run full health check (linting + tests)
hatch run all

# Run only tests
hatch run test
```

## Project Structure

- **`src/`**: Contains the core logic and datasets.
- **`tests/`**: Contains unit and end-to-end tests, mirrored to the source structure.
- **`setup/`**: Documentation and environment verification scripts.

---