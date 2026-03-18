from typing import Any

import torch
import torch.nn as nn

from learning_llms_from_first_principles.modules.gpt import GPTModel


def generate_tokens(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_kv_cache: bool = False,
) -> torch.Tensor:
    """
    Generates text sequentially by repeatedly predicting the next token.
    """

    model.eval()

    # Intuition: Temperature is a "Volume Control" for the gap between token scores.
    # - T = 0.0 (Greedy): Deterministic. The model always picks the #1 winner.
    # - T < 1.0 (Sharp): Higher confidence, less creative. Everest becomes higher,
    #                   valleys deeper. Risk: Getting stuck in repetitive loops.
    # - T = 1.0 (Natural): Original distribution from training. Balanced creativity.
    # - T > 1.0 (Flat): Low confidence, high creativity/randomness. Peaks flatten out.
    #                   Results in "Word Salad" (nonsense) as T increases.
    temperature = max(0, temperature)  # negative temp doesn't make sense, treat as greedy decoding

    full_sequence = idx  # track the full output for kv_cache mode

    # KV cache note: the first iteration implicitly acts as PREFILL — idx is the full
    # prompt, so the model processes all tokens and populates the cache. After that,
    # idx = idx_next (1 token), so subsequent iterations are DECODE — the model only
    # processes the new token while the cache provides the history.

    for _ in range(max_new_tokens):
        # KV cache: full prompt on first iter (prefill), single token after (decode)
        idx_trunc = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_trunc, use_kv_cache=use_kv_cache)  # (bs, num_tokens, vocab_size)

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
            logits = logits / temperature

            # Top-P (Nucleus Sampling): Dynamic filtering. We keep the smallest set of tokens that
            # sum to P% probability. This narrows the pool when the model is confident (high
            # precision) and expands it when it's uncertain (high diversity).
            if top_p is not None:
                # Sort probabilities and calculate cumulative sum
                probs_logits = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_ind = torch.sort(probs_logits, dim=-1, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

                # Create mask
                mask = cumsum_probs < top_p
                # Rule: Keep the smallest set of tokens that sum to >= top_p.
                # Example: top_p = 0.5, sorted_probs = [0.4, 0.3, 0.2, 0.1]
                #   1. cumsum = [0.4, 0.7, 0.9, 1.0]
                #   2. mask = (cumsum < top_p) -> [True, False, False, False]
                #      Note: 0.4 isn't enough to reach 0.5. We NEED the 0.3 token too.
                #
                #   3. Right shift by 1: mask = [True, True, False, False]
                #      Now the 0.3 token (which pushed us over 0.5) is correctly KEPT.
                mask[:, 1:] = mask[:, :-1].clone()
                # Always keep at least the #1 most likely token, so that atleast 1 token is chosen
                mask[:, 0] = True

                # Apply mask to un-sorted probs/logits (mask re-arranging)
                new_mask = torch.zeros_like(logits, dtype=torch.bool)
                new_mask.scatter_(dim=-1, index=sorted_ind, src=mask)

                # Flip the mask for masked fill and add -torch.inf where value is "True"
                logits.masked_fill_(~new_mask, -torch.inf)

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            # Deterministic Greedy: No math needed, just pick the #1 highest score
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if use_kv_cache:
            full_sequence = torch.cat((full_sequence, idx_next), dim=1)
            idx = idx_next  # KV cache decode: feed only the new token, cache has the rest
        else:
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

    return full_sequence if use_kv_cache else idx


def generate_text(
    text: str,
    model: nn.Module,
    tokenizer: Any,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: torch.device | str = "cpu",
) -> str:
    """Generate text from a prompt string and return the full output as a string."""
    from learning_llms_from_first_principles.utils.tokenization_utils import (
        text_to_token_ids,
        token_ids_to_text,
    )

    model.eval()
    idx = text_to_token_ids(text, tokenizer).to(device)

    with torch.no_grad():
        out_ids = generate_tokens(
            model=model,  # type: ignore[arg-type]
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    return token_ids_to_text(out_ids, tokenizer)


def classify_text(
    text: str,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device | str = "cpu",
    label_map: dict[int, str] | None = None,
) -> str:
    """Classify a single text string using a fine-tuned GPT classifier.

    Args:
        text: The input text to classify.
        model: A GPT model with a classification ``out_head``.
        tokenizer: A tiktoken-compatible tokenizer.
        device: Device to run inference on.
        label_map: Optional mapping from class index to label string.
            Defaults to ``{0: "ham", 1: "spam"}``.

    Returns:
        The predicted label string.
    """
    if label_map is None:
        label_map = {0: "ham", 1: "spam"}

    model.eval()
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(token_tensor)[:, -1, :]

    pred = int(torch.argmax(logits, dim=-1).item())
    return label_map[pred]
