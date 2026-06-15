import json
from pathlib import Path

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from .preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model_artifacts"
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained(str(MODEL_DIR))
model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()

label_map_path = MODEL_DIR / "label_map.json"

if label_map_path.exists():
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
else:
    label_map = {
        "0": "Negative",
        "1": "Positive"
    }

def predict_sentiment(text: str) -> dict:
    cleaned_text = preprocess_text(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class_id = int(torch.argmax(probabilities, dim=1).cpu().item())
        confidence = float(probabilities[0][predicted_class_id].cpu().item()) * 100

    return {
        "input_text": text,
        "cleaned_text": cleaned_text,
        "predicted_class_id": predicted_class_id,
        "label": label_map[str(predicted_class_id)],
        "confidence": round(confidence, 2)
    }