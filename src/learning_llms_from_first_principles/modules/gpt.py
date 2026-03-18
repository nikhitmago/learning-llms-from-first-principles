import torch
import torch.nn as nn

from learning_llms_from_first_principles.modules.norm import LayerNorm
from learning_llms_from_first_principles.modules.transformer import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"]
        )  # embedding dimension used as a look up table for token_ids
        self.pos_emb = nn.Embedding(cfg["context_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # cur pos for pos embeddings and kv cache
        self.cur_pos_gpt = 0

        # Layer Norm
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Final layer
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, token_ids: torch.Tensor, use_kv_cache: bool = False) -> torch.Tensor:
        bs, seq_len = token_ids.shape
        token_embeds = self.tok_emb(token_ids)
        if use_kv_cache:
            pos_embeds = self.pos_emb(
                torch.arange(self.cur_pos_gpt, self.cur_pos_gpt + seq_len, device=token_ids.device)
            )
            self.cur_pos_gpt += seq_len
        else:
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=token_ids.device))

        x = token_embeds + pos_embeds
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x, use_kv_cache=use_kv_cache)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache_gpt(self) -> None:
        for block in self.trf_blocks:
            block.multi_head_attention.reset_kv_cache()  # type: ignore[union-attr, operator]
        self.cur_pos_gpt = 0
