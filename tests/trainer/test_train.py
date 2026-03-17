from typing import Any
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from learning_llms_from_first_principles.trainer.train import main


@patch("learning_llms_from_first_principles.trainer.train.GPTModel")
@patch("learning_llms_from_first_principles.trainer.train.create_dataloader_v1")
@patch("learning_llms_from_first_principles.trainer.train.get_device")
@patch("learning_llms_from_first_principles.trainer.train.train_model_v1")
@patch("learning_llms_from_first_principles.trainer.train.split_data")
def test_train_main(
    mock_split: MagicMock,
    mock_train_loop: MagicMock,
    mock_get_device: MagicMock,
    mock_create_loader: MagicMock,
    mock_gpt_model: MagicMock,
) -> None:
    # Setup mock config
    cfg = OmegaConf.create(
        {
            "training": {
                "num_epochs": 1,
                "eval_freq": 1,
                "lr": 0.001,
                "weight_decay": 0.1,
                "batch_size": 2,
                "max_length": 64,
                "stride": 32,
                "warmup_ratio": 0.1,
                "warmup_min_lr": 3e-05,
                "decay_floor_lr": 1e-06,
                "max_norm": 1.0,
            },
            "data": {"file_path": "dummy.txt", "train_ratio": 0.9, "val_ratio": 0.1},
            "model": {"name": "gpt2-small", "save_model_path": None},
        }
    )

    # Setup mocks
    mock_split.return_value = ("train", "val", "test")
    mock_get_device.return_value = "cpu"
    mock_train_loop.return_value = (MagicMock(), [0.1], [0.1], [0.1])

    # Fix: mock parameters() to return an iterable for the optimizer
    dummy_param = torch.nn.Parameter(torch.randn(1, 1))
    mock_gpt_model.return_value.parameters.return_value = [dummy_param]

    # Use a side_effect for open to only handle our dummy file
    original_open = open

    def side_effect(file: str | int, *args: Any, **kwargs: Any) -> Any:
        if file == "dummy.txt":
            m = MagicMock()
            m.__enter__.return_value.read.return_value = "dummy text content"
            return m
        return original_open(file, *args, **kwargs)

    with patch("builtins.open", side_effect=side_effect):
        # Run main
        main(cfg)

    # Verify calls
    mock_create_loader.assert_called()
    mock_train_loop.assert_called_once()
    mock_gpt_model.assert_called_once()
