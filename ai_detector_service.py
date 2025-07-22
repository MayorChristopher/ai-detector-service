from fastapi import FastAPI, Request
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()
model_name = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.post("/detect")
async def detect(request: Request):
    data = await request.json()
    text = data.get("text", "")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
    return {
        "human_score": scores[0],
        "ai_score": scores[1],
        "result": "AI-generated" if scores[1] > scores[0] else "Human-written"
    }
