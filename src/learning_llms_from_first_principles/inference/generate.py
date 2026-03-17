import torch
import torch.nn as nn


def generate_text_simple(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Generates text sequentially by repeatedly predicting the next token.
    """

    # Intuition: Temperature is a "Volume Control" for the gap between token scores.
    # - T = 0.0 (Greedy): Deterministic. The model always picks the #1 winner.
    # - T < 1.0 (Sharp): Higher confidence, less creative. Everest becomes higher,
    #                   valleys deeper. Risk: Getting stuck in repetitive loops.
    # - T = 1.0 (Natural): Original distribution from training. Balanced creativity.
    # - T > 1.0 (Flat): Low confidence, high creativity/randomness. Peaks flatten out.
    #                   Results in "Word Salad" (nonsense) as T increases.
    temperature = max(0, temperature)  # negative temp doesn't make sense, treat as greedy decoding

    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # idx: (bs, num_tokens)
        idx_trunc = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_trunc)  # (bs, num_tokens, vocab_size)

        # Focus only on the last time stamp/token (next token over all vocab)
        logits = logits[:, -1, :]

        # Add topK to make higher temp values less non-sensical so that fewer tokens are chosen from the vocab
        if top_k is not None:
            top_k_logits, top_k_ind = torch.topk(logits, top_k)  # bs, vocab_size
            mask = torch.zeros_like(logits)
            mask = ~mask.scatter(dim=-1, index=top_k_ind, value=1).bool()
            logits.masked_fill_(mask, -torch.inf)

        # Apply temperature
        if temperature > 0:
            # Stochastic Sampling: Scale logits and roll the dice (multinomial)
            logits_scaled = logits / temperature
            probs = torch.softmax(logits_scaled, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            # Deterministic Greedy: No math needed, just pick the #1 highest score
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
