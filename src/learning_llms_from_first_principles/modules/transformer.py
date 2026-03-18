import torch
import torch.nn as nn

from learning_llms_from_first_principles.modules.attention import MultiHeadAttentionWeightSplits
from learning_llms_from_first_principles.modules.feedforward import Feedforward
from learning_llms_from_first_principles.modules.norm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        self.ffn = Feedforward(cfg)
        self.layer_norm_1 = LayerNorm(
            emb_dim=cfg["emb_dim"]
        )  # need 2 copies of layer norms becasue it has learnable params scale and shift
        self.layer_norm_2 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.multi_head_attention = MultiHeadAttentionWeightSplits(
            d_emb=cfg["emb_dim"],
            d_attn=cfg["emb_dim"],
            context_len=cfg["context_len"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
        )
        self.dropout_layer = nn.Dropout(
            cfg["drop_rate"]
        )  # dropout doesn't need 2 copies because it doesn't have learnable params

    def forward(self, x: torch.Tensor, use_kv_cache: bool = False) -> torch.Tensor:
        # attention
        shortcut = x
        x = self.layer_norm_1(x)
        x = self.multi_head_attention(x, use_kv_cache=use_kv_cache)
        x = self.dropout_layer(x)
        x = x + shortcut

        # ffn
        shortcut = x
        x = self.layer_norm_2(x)
        x = self.ffn(x)
        x = self.dropout_layer(x)
        x = x + shortcut

        return x
