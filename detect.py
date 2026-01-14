import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_labels(model_dir: Path):
    label_path = model_dir / "label_classes.json"
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def predict(text: str, model_dir: str = "bert-lyrics-emotion", top_k: int = 3):
    model_dir = Path(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    labels = load_labels(model_dir)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    top_k = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k=top_k)

    results = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        name = labels[idx] if labels and idx < len(labels) else f"LABEL_{idx}"
        results.append({"label": name, "score": score})

    return results


if __name__ == "__main__":
    # chạy kiểu: python infer_local.py "your lyric here"
    if len(sys.argv) < 2:
        print('Usage: python infer_local.py "your lyric here"')
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    out = predict(text, model_dir="bert-lyrics-emotion", top_k=3)

    print("\nInput:", text)
    print("Predictions:")
    for r in out:
        print(f"- {r['label']}: {r['score']:.4f}")