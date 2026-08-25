from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class BackboneOutput:
    text: torch.Tensor
    vision: torch.Tensor | None


class FrozenSiglip2(nn.Module):
    """Frozen Hugging Face SigLIP2 encoder with pooled image/text features.

    Optional dependency. For empty-room language-to-motion training, call only
    `encode_text`; add `encode_images` later when visual information is useful.
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224", device: str = "cuda"):
        super().__init__()
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError("Install with: pip install -e '.[vlm]'") from exc
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device_name = device

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}
        if hasattr(self.model, "get_text_features"):
            return self.model.get_text_features(**inputs)
        out = self.model.text_model(**inputs)
        return getattr(out, "pooler_output", out.last_hidden_state.mean(dim=1))

    @torch.inference_mode()
    def encode_images(self, images: list[Any]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}
        if hasattr(self.model, "get_image_features"):
            return self.model.get_image_features(**inputs)
        out = self.model.vision_model(**inputs)
        return getattr(out, "pooler_output", out.last_hidden_state.mean(dim=1))
