# Plant Diagnosis App (Swin Transformer)

A minimal full-stack app around your `best_model_swin.pth` checkpoint:
FastAPI backend that loads the real model and runs inference, plus a
simple HTML/CSS/JS frontend to upload an image and view predictions.

## What's in the checkpoint

I inspected `best_model_swin.pth` directly (its state_dict keys and tensor
shapes) and reconstructed the architecture from that:

- **Backbone**: Swin-Tiny (`microsoft/swin-tiny-patch4-window7-224` config —
  embed_dim=96, depths=[2,2,6,2], heads=[3,6,12,24], window=7, patch=4,
  image size 224), implemented via Hugging Face `transformers.SwinModel`.
- **crop_head**: `Linear(768 → 14)` — crop type classification
- **disease_head**: `Linear(768 → 22)` — disease classification
- **part_head**: `Linear(768 → 1)` — a single score (binary/regression —
  you know your training setup better than I can infer from weights alone,
  so it's exposed as a raw sigmoid score; adjust `inference.py` if it's
  actually meant differently, e.g. a regression value without sigmoid)

`backend/model.py` builds this exact architecture and loads the checkpoint
with `strict=False`, printing any missing/unexpected keys on startup so
you can immediately see if anything doesn't line up in your environment.

## Folder structure

```
plant-diagnosis-app/
├── backend/
│   ├── app.py                # FastAPI app (routes: /health, /labels, /predict)
│   ├── model.py               # MultiTaskSwin architecture + checkpoint loader
│   ├── inference.py           # image preprocessing + prediction logic
│   ├── labels.json            # EDIT ME: real class names for crop/disease
│   ├── requirements.txt
│   └── best_model_swin.pth    # your uploaded checkpoint
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

## 1. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

On startup, watch the console:
- `"Checkpoint loaded with a perfect key match."` → architecture matches exactly.
- Any `missing keys` / `unexpected keys` warnings → your transformers version's
  internal Swin naming might differ slightly from what I inferred. Share those
  printed key names with me and I'll adjust `model.py` to match exactly.

Test it directly:
```bash
curl -X POST http://localhost:8000/predict -F "file=@/path/to/leaf.jpg"
```

## 2. IMPORTANT: fill in real labels

`backend/labels.json` currently has placeholder names (`crop_0`, `disease_0`, ...).
Replace them with your actual class names **in the same order** your training
script used (usually `ImageFolder.classes` or your `class_to_idx` mapping).
Predictions will still work with placeholders — only the display names are affected.

## 3. Frontend setup

No build step needed — it's plain HTML/CSS/JS.

```bash
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500` in your browser. The page has a
"Backend URL" field at the bottom (defaults to `http://localhost:8000`) —
update it if your backend runs elsewhere.

## API reference

| Method | Path       | Description                                  |
|--------|-----------|-----------------------------------------------|
| GET    | `/health`  | Check server + model load status             |
| GET    | `/labels`  | Returns current crop/disease label lists      |
| POST   | `/predict` | Upload an image (`file` field), get predictions |

Example response from `/predict`:
```json
{
  "crop": {
    "top_prediction": {"label": "tomato", "confidence": 0.94},
    "top_k": [ ... ]
  },
  "disease": {
    "top_prediction": {"label": "early_blight", "confidence": 0.81},
    "top_k": [ ... ]
  },
  "part_score": 0.73
}
```

## Notes / things to double check on your end

1. **Preprocessing**: I used standard ImageNet mean/std normalization at
   224×224, which is the default for Swin models. If your training script
   used different normalization or image size, update `inference.py`.
2. **`part_head`**: shape is `(768 → 1)`. I exposed it as a sigmoid score.
   If it's actually a regression target (e.g. severity 0–100) rather than a
   probability, remove the `torch.sigmoid()` call in `inference.py`.
3. No GPU is required — the backend auto-detects CUDA and falls back to CPU.
