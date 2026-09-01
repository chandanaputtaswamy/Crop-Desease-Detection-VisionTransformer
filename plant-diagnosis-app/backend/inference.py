"""
Preprocessing + inference helpers.
Uses standard ImageNet normalization stats, matching how Swin models
(including HF's SwinImageProcessor defaults) are normally trained.
"""

import io
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels.json")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_labels():
    with open(LABELS_PATH, "r") as f:
        return json.load(f)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(image)
    return tensor.unsqueeze(0)  # add batch dim -> (1, 3, 224, 224)


@torch.no_grad()
def predict(model, image_bytes: bytes, device: str = "cpu", top_k: int = 3):
    labels = load_labels()
    crop_labels = labels.get("crop", [])
    disease_labels = labels.get("disease", [])

    pixel_values = preprocess_image(image_bytes).to(device)
    outputs = model(pixel_values)

    crop_probs = F.softmax(outputs["crop"], dim=1).squeeze(0)
    disease_probs = F.softmax(outputs["disease"], dim=1).squeeze(0)
    part_score = torch.sigmoid(outputs["part"]).squeeze(0)

    def top_k_result(probs, label_list, k):
        k = min(k, probs.shape[0])
        top_probs, top_idxs = torch.topk(probs, k)
        results = []
        for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
            name = label_list[idx] if idx < len(label_list) else f"class_{idx}"
            results.append({"label": name, "confidence": round(prob, 4)})
        return results

    return {
        "crop": {
            "top_prediction": top_k_result(crop_probs, crop_labels, 1)[0],
            "top_k": top_k_result(crop_probs, crop_labels, top_k),
        },
        "disease": {
            "top_prediction": top_k_result(disease_probs, disease_labels, 1)[0],
            "top_k": top_k_result(disease_probs, disease_labels, top_k),
        },
        "part_score": round(part_score.item(), 4),
    }
