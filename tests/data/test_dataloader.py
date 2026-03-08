import torch

from data.dataloader import create_dataloader_v1


def test_create_dataloader_v1() -> None:
    txt = "This is a simple test text to verify the dataloader creation and batching."
    batch_size = 2
    max_length = 4
    stride = 2

    dataloader = create_dataloader_v1(
        txt, batch_size=batch_size, max_length=max_length, stride=stride, shuffle=False
    )

    # Get first batch
    batch = next(iter(dataloader))
    x, y = batch

    assert x.shape == (batch_size, max_length)
    assert y.shape == (batch_size, max_length)

    # Check if target is shifted version of input in the batch
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_dataloader_shuffle_reproducibility() -> None:
    # Simple check to see if it runs with shuffle=True
    txt = "Some text for shuffling test." * 10
    dataloader = create_dataloader_v1(txt, batch_size=1, max_length=2, stride=1, shuffle=True)
    assert len(dataloader) > 0
