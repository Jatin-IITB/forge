"""Qwen2 token-classification model with optional bidirectional attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Qwen2ForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput


class ForgeQwen2ForTokenClassification(Qwen2ForTokenClassification):
    """Qwen2 label head with full-attention and class-balanced loss support."""

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer(
            "forge_class_weights",
            torch.ones(config.num_labels),
            persistent=False,
        )

    def set_class_weights(self, weights: torch.Tensor) -> None:
        self.forge_class_weights.copy_(weights.to(self.forge_class_weights.device))

    def _attention_mapping(self, attention_mask: torch.Tensor | dict | None):
        if not getattr(self.config, "forge_full_attention", False):
            return attention_mask
        if isinstance(attention_mask, dict):
            return attention_mask
        if attention_mask is None:
            return None
        dtype = self.model.embed_tokens.weight.dtype
        padding = attention_mask[:, None, None, :].to(dtype=dtype)
        additive_mask = (1.0 - padding) * torch.finfo(dtype).min
        return {"full_attention": additive_mask}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=self._attention_mapping(attention_mask),
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=False,
            **kwargs,
        )
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                outputs.logits.reshape(-1, self.config.num_labels),
                labels.reshape(-1),
                weight=self.forge_class_weights.to(outputs.logits.device),
                ignore_index=-100,
            )
        return TokenClassifierOutput(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
