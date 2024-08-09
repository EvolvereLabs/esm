# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union #, List, Dict, Optional, Tuple
import torch
import torch.nn as nn

import esm
from esm.modules_mech_intepret import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer


class ESM2(nn.Module):
    """
    ESM2 model class for protein sequence modeling.
    """

    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
    ):
        """
        Initialize the ESM2 model.

        Args:
            num_layers (int): Number of transformer layers.
            embed_dim (int): Embedding dimension.
            attention_heads (int): Number of attention heads.
            alphabet (Union[esm.data.Alphabet, str]): Alphabet for tokenization.
            token_dropout (bool): Whether to apply token dropout.
        """
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout

        self._init_submodules()

    def _init_submodules(self):
        """
        Initialize submodules for the ESM2 model.
        """
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2, "Tokens tensor must be 2-dimensional"
        padding_mask = tokens.eq(self.padding_idx)

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        layer_inputs = {}
        wide_Xs = {}
        pre_res_Xs = {}
        after_attents ={}
        before_f1s={}

        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            #print(f"prcoessing layer {layer_idx}")
            x, attn, layer_input, after_attent, wide_X, pre_res_x, before_f1 = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
              # Debug prints
           # print(f"Layer {layer_idx + 1}")
           # print(f"x shape: {x.shape}")
           # print(f"attn shape: {attn.shape}")
           # print(f"layer_input shape: {layer_input.shape}")
           # print(f"after_attent shape: {after_attent.shape}")
           # print(f"wide_X shape: {wide_X.shape}")
           # print(f"pre_res_x shape: {pre_res_x.shape}")
           # print(f"before_f1 shape: {before_f1.shape}")
        
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
                layer_inputs[layer_idx + 1] = layer_input.transpose(0, 1)
                wide_Xs[layer_idx + 1] = wide_X.transpose(0, 1)
                pre_res_Xs[layer_idx + 1] = pre_res_x.transpose(0, 1)
                after_attents[layer_idx + 1] = after_attent.transpose(0, 1)
                before_f1s[layer_idx + 1] = before_f1.transpose(0, 1)
            if need_head_weights:
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {
            "logits": x,
            "representations": hidden_representations,
            "layer_inputs": layer_inputs,
            "wide_Xs": wide_Xs,
            "pre_res_Xs": pre_res_Xs,
            "after_atten": after_attents,
            "before_f1": before_f1s
        }
        if need_head_weights:
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        """
        Predict contacts for the given tokens.

        Args:
            tokens (torch.Tensor): Input token tensor of shape (B, T).

        Returns:
            torch.Tensor: Contact predictions.
        """
        return self(tokens, return_contacts=True)["contacts"]
