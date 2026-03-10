import torch
import torch.nn as nn

from learning_llms_from_first_principles.models.norm import LayerNorm
from learning_llms_from_first_principles.models.transformer import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"]
        )  # embedding dimension used as a look up table for token_ids
        self.pos_emb = nn.Embedding(cfg["context_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Layer Norm
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Final layer
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        bs, seq_len = token_ids.shape
        token_embeds = self.tok_emb(token_ids)
        pos_embeds = self.pos_emb(torch.arange(seq_len))

        x = token_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
