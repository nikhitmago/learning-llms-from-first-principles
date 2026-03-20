import pytest
import tiktoken

from learning_llms_from_first_principles.data.datasets import InstructionDataset
from learning_llms_from_first_principles.utils.data_utils import (
    format_instruct_prompt,
    instruct_collate_fn,
)


@pytest.fixture
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def sample_data() -> list[dict[str, str]]:
    return [
        {
            "instruction": "Translate to French",
            "input": "Hello world",
            "output": "Bonjour le monde",
        },
        {
            "instruction": "Summarize the text",
            "input": "",
            "output": "This is a summary.",
        },
    ]


def test_format_instruct_prompt_with_input() -> None:
    entry = {"instruction": "Translate", "input": "Hello", "output": "Bonjour"}
    result = format_instruct_prompt(entry)
    assert "### Instruction:" in result
    assert "Translate" in result
    assert "### Input:" in result
    assert "Hello" in result


def test_format_instruct_prompt_without_input() -> None:
    entry = {"instruction": "Tell a joke", "input": "", "output": "Why did..."}
    result = format_instruct_prompt(entry)
    assert "### Instruction:" in result
    assert "### Input:" not in result


def test_instruction_dataset_length(
    sample_data: list[dict[str, str]], tokenizer: tiktoken.Encoding
) -> None:
    dataset = InstructionDataset(sample_data, tokenizer)
    assert len(dataset) == 2


def test_instruction_dataset_returns_token_list(
    sample_data: list[dict[str, str]], tokenizer: tiktoken.Encoding
) -> None:
    dataset = InstructionDataset(sample_data, tokenizer)
    item = dataset[0]
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "target_ids" in item
    assert all(isinstance(t, int) for t in item["input_ids"])
    assert all(isinstance(t, int) for t in item["target_ids"])


def test_instruction_dataset_prompt_is_masked(
    sample_data: list[dict[str, str]], tokenizer: tiktoken.Encoding
) -> None:
    dataset = InstructionDataset(sample_data, tokenizer)
    item = dataset[0]
    # The beginning of target_ids should be -100 (prompt masked)
    assert item["target_ids"][0] == -100
    # The end should have real token IDs (response)
    assert item["target_ids"][-1] != -100


def test_instruct_collate_fn_pads_and_stacks() -> None:
    batch = [
        {"input_ids": [1, 2, 3], "target_ids": [-100, -100, 4]},
        {"input_ids": [5, 6], "target_ids": [-100, 7]},
    ]
    inputs, targets = instruct_collate_fn(batch)
    assert inputs.shape == (2, 3)
    assert targets.shape == (2, 3)
    # Second item should be padded
    assert inputs[1, 2].item() == 50256  # pad_token_id
    assert targets[1, 2].item() == -100  # ignore_index


def test_instruct_collate_fn_truncates() -> None:
    batch = [
        {"input_ids": [1, 2, 3, 4, 5], "target_ids": [-100, -100, -100, 4, 5]},
    ]
    inputs, targets = instruct_collate_fn(batch, allowed_max_length=3)
    assert inputs.shape == (1, 3)
    assert targets.shape == (1, 3)
