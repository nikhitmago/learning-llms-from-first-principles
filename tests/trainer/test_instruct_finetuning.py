from unittest.mock import MagicMock, mock_open, patch

import torch
from omegaconf import OmegaConf

from learning_llms_from_first_principles.trainer.instruct_finetuning import main

SAMPLE_JSON = '[{"instruction":"Test","input":"","output":"Response"},{"instruction":"Test2","input":"Hello","output":"World"},{"instruction":"Test3","input":"","output":"Done"}]'


@patch("learning_llms_from_first_principles.trainer.instruct_finetuning.generate_text")
@patch("learning_llms_from_first_principles.trainer.instruct_finetuning.GPTModel")
@patch("learning_llms_from_first_principles.trainer.instruct_finetuning.create_instruct_dataloader")
@patch("learning_llms_from_first_principles.trainer.instruct_finetuning.get_device")
@patch("learning_llms_from_first_principles.trainer.instruct_finetuning.train_model_v1")
@patch("learning_llms_from_first_principles.trainer.instruct_finetuning.torch.load")
@patch("pathlib.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=SAMPLE_JSON)
def test_instruct_finetuning_main(
    mock_file: MagicMock,
    mock_path_exists: MagicMock,
    mock_torch_load: MagicMock,
    mock_train_loop: MagicMock,
    mock_get_device: MagicMock,
    mock_create_loader: MagicMock,
    mock_gpt_model: MagicMock,
    mock_generate_text: MagicMock,
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
            "data": {
                "file_path": "dummy_instruct.json",
                "train_ratio": 0.85,
                "test_ratio": 0.1,
                "allowed_max_length": 512,
            },
            "model": {
                "name": "gpt2-124m",
                "pretrained_path": "/tmp/fake_pretrained.pth",
                "save_model_path": None,
            },
        }
    )

    mock_get_device.return_value = torch.device("cpu")

    dummy_param = torch.nn.Parameter(torch.randn(1, 1))
    mock_model_instance = MagicMock()
    mock_model_instance.parameters.side_effect = lambda: iter([dummy_param])
    mock_gpt_model.return_value = mock_model_instance

    mock_loader = MagicMock()
    mock_loader.__len__ = MagicMock(return_value=4)
    mock_create_loader.return_value = mock_loader

    mock_train_loop.return_value = (mock_model_instance, [0.5], [0.6], [0.001])

    mock_generate_text.return_value = "prompt text\n\n### Response:\nSample response"

    main(cfg)

    mock_create_loader.assert_called()
    mock_train_loop.assert_called_once()
