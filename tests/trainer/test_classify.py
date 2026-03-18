from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from learning_llms_from_first_principles.trainer.classify import main


@patch("learning_llms_from_first_principles.trainer.classify.GPTModel")
@patch("learning_llms_from_first_principles.trainer.classify.create_classify_dataloader")
@patch("learning_llms_from_first_principles.trainer.classify.get_device")
@patch("learning_llms_from_first_principles.trainer.classify.train_classifier")
@patch("learning_llms_from_first_principles.trainer.classify.calc_accuracy_loader")
@patch("learning_llms_from_first_principles.trainer.classify.torch.load")
@patch("pathlib.Path.exists", return_value=True)
def test_classify_main_without_lora(
    mock_path_exists: MagicMock,
    mock_torch_load: MagicMock,
    mock_accuracy: MagicMock,
    mock_train_loop: MagicMock,
    mock_get_device: MagicMock,
    mock_create_loader: MagicMock,
    mock_gpt_model: MagicMock,
) -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "num_epochs": 1,
                "eval_freq": 1,
                "eval_iter": 1,
                "lr": 5e-5,
                "weight_decay": 0.1,
                "batch_size": 2,
                "num_workers": 0,
            },
            "data": {"file_path": "dummy_spam.csv"},
            "model": {
                "name": "gpt2-124m",
                "num_classes": 2,
                "pretrained_path": "/tmp/fake_pretrained.pth",
                "save_model_path": None,
            },
            "lora": {
                "enabled": False,
                "rank": 16,
                "alpha": 16,
                "save_adapter_path": None,
                "merge_after_training": False,
            },
        }
    )

    mock_get_device.return_value = torch.device("cpu")

    dummy_param = torch.nn.Parameter(torch.randn(1, 1))
    mock_model_instance = MagicMock()
    mock_model_instance.parameters.side_effect = lambda: iter([dummy_param])
    mock_model_instance.trf_blocks = [MagicMock()]
    mock_model_instance.final_norm = MagicMock()
    mock_gpt_model.return_value = mock_model_instance

    mock_dataset = MagicMock()
    mock_dataset.max_length = 64
    mock_loader = MagicMock()
    mock_loader.__len__ = MagicMock(return_value=4)
    mock_create_loader.return_value = (mock_loader, mock_dataset)

    mock_train_loop.return_value = (mock_model_instance, [0.5], [0.6], [0.8], [0.7])

    mock_accuracy.return_value = 0.9

    main(cfg)

    mock_create_loader.assert_called()
    mock_train_loop.assert_called_once()
    mock_accuracy.assert_called_once()


@patch("learning_llms_from_first_principles.trainer.classify.GPTModel")
@patch("learning_llms_from_first_principles.trainer.classify.create_classify_dataloader")
@patch("learning_llms_from_first_principles.trainer.classify.get_device")
@patch("learning_llms_from_first_principles.trainer.classify.train_classifier")
@patch("learning_llms_from_first_principles.trainer.classify.calc_accuracy_loader")
@patch("learning_llms_from_first_principles.trainer.classify.torch.load")
@patch("pathlib.Path.exists", return_value=True)
@patch("learning_llms_from_first_principles.trainer.classify.replace_linear_with_lora")
@patch("learning_llms_from_first_principles.trainer.classify.save_lora_adapters")
@patch("learning_llms_from_first_principles.trainer.classify.merge_lora_weights")
def test_classify_main_with_lora(
    mock_merge: MagicMock,
    mock_save_adapters: MagicMock,
    mock_replace_lora: MagicMock,
    mock_path_exists: MagicMock,
    mock_torch_load: MagicMock,
    mock_accuracy: MagicMock,
    mock_train_loop: MagicMock,
    mock_get_device: MagicMock,
    mock_create_loader: MagicMock,
    mock_gpt_model: MagicMock,
) -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "num_epochs": 1,
                "eval_freq": 1,
                "eval_iter": 1,
                "lr": 5e-5,
                "weight_decay": 0.1,
                "batch_size": 2,
                "num_workers": 0,
            },
            "data": {"file_path": "dummy_spam.csv"},
            "model": {
                "name": "gpt2-124m",
                "num_classes": 2,
                "pretrained_path": "/tmp/fake_pretrained.pth",
                "save_model_path": None,
            },
            "lora": {
                "enabled": True,
                "rank": 8,
                "alpha": 8,
                "save_adapter_path": "/tmp/fake_adapters.pth",
                "merge_after_training": True,
            },
        }
    )

    mock_get_device.return_value = torch.device("cpu")

    dummy_param = torch.nn.Parameter(torch.randn(1, 1))
    mock_model_instance = MagicMock()
    mock_model_instance.parameters.side_effect = lambda: iter([dummy_param])
    mock_model_instance.trf_blocks = [MagicMock()]
    mock_model_instance.final_norm = MagicMock()
    mock_gpt_model.return_value = mock_model_instance

    mock_dataset = MagicMock()
    mock_dataset.max_length = 64
    mock_loader = MagicMock()
    mock_loader.__len__ = MagicMock(return_value=4)
    mock_create_loader.return_value = (mock_loader, mock_dataset)

    mock_train_loop.return_value = (mock_model_instance, [0.5], [0.6], [0.8], [0.7])
    mock_accuracy.return_value = 0.9

    main(cfg)

    mock_replace_lora.assert_called_once_with(mock_model_instance, rank=8, alpha=8)
    mock_save_adapters.assert_called_once()
    mock_merge.assert_called_once()
    mock_train_loop.assert_called_once()
