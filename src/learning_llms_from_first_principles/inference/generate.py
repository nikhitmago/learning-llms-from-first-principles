import torch
import torch.nn as nn


def generate_text_simple(
    model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int
) -> torch.Tensor:
    """
    Generates text sequentially by repeatedly predicting the next token.
    """
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # idx: (bs, num_tokens)
        idx_trunc = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_trunc)  # (bs, num_tokens, vocab_size)

        # Focus only on the last time stamp/token
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
