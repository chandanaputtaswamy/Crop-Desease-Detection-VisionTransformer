"""
Multi-task Swin-Tiny model definition.

Reconstructed from the checkpoint's state_dict structure:
    encoder.embeddings.patch_embeddings.projection   -> Conv2d(3, 96, kernel=4, stride=4)
    encoder.encoder.layers.{0..3}.blocks.{...}       -> depths [2, 2, 6, 2]
    encoder.encoder.layers.*.attention.q/k/v/o_proj  -> HF SwinModel attention
    encoder.layernorm                                -> final backbone norm
    crop_head      -> Linear(768, NUM_CROP_CLASSES)
    disease_head   -> Linear(768, NUM_DISEASE_CLASSES)
    part_head      -> Linear(768, 1)

This matches microsoft/swin-tiny-patch4-window7-224's architecture
(embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=7,
patch_size=4, image_size=224), wrapped with three classification heads.
"""

import torch
import torch.nn as nn
from transformers import SwinConfig, SwinModel

NUM_CROP_CLASSES = 14
NUM_DISEASE_CLASSES = 22
NUM_PART_OUTPUTS = 1

IMAGE_SIZE = 224


class MultiTaskSwin(nn.Module):
    def __init__(
        self,
        num_crop_classes: int = NUM_CROP_CLASSES,
        num_disease_classes: int = NUM_DISEASE_CLASSES,
        num_part_outputs: int = NUM_PART_OUTPUTS,
    ):
        super().__init__()

        config = SwinConfig(
            image_size=IMAGE_SIZE,
            patch_size=4,
            num_channels=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
        )
        # NOTE: we build a fresh (randomly initialized) SwinModel here and then
        # overwrite every weight with the ones from best_model_swin.pth in
        # load_model() below. We deliberately do NOT call SwinModel.from_pretrained
        # since we want our own trained weights, not ImageNet pretrained ones.
        self.encoder = SwinModel(config)

        hidden_size = self.encoder.config.hidden_size  # 768 for swin-tiny
        self.crop_head = nn.Linear(hidden_size, num_crop_classes)
        self.disease_head = nn.Linear(hidden_size, num_disease_classes)
        self.part_head = nn.Linear(hidden_size, num_part_outputs)

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.encoder(pixel_values=pixel_values)
        pooled = outputs.pooler_output  # (B, 768)

        return {
            "crop": self.crop_head(pooled),
            "disease": self.disease_head(pooled),
            "part": self.part_head(pooled),
        }


def load_model(checkpoint_path: str, device: str = "cpu") -> MultiTaskSwin:
    """Load MultiTaskSwin and populate it with weights from best_model_swin.pth."""
    model = MultiTaskSwin()

    state_dict = torch.load(checkpoint_path, map_location=device)
    # Some training scripts save {"model_state_dict": ...} or {"state_dict": ...}
    # instead of the raw state_dict. Handle both.
    if isinstance(state_dict, dict) and "state_dict" in state_dict and not any(
        k.startswith("encoder.") for k in state_dict.keys()
    ):
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict and not any(
        k.startswith("encoder.") for k in state_dict.keys()
    ):
        state_dict = state_dict["model_state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[model.py] WARNING - missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[model.py] WARNING - unexpected keys when loading checkpoint: {unexpected}")
    if not missing and not unexpected:
        print("[model.py] Checkpoint loaded with a perfect key match.")

    model.to(device)
    model.eval()
    return model
