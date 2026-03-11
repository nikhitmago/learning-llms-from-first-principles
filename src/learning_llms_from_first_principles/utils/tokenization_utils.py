from typing import Any

import torch


def text_to_token_ids(text: str, tokenizer: Any) -> torch.Tensor:
    encoded_text = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded_text).unsqueeze(0)  # add a batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Any) -> str:
    return tokenizer.decode(token_ids.squeeze(0).tolist())  # remove batch dimension
