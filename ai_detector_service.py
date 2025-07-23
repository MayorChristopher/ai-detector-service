from fastapi import FastAPI, Request
import requests
import os

app = FastAPI()

API_URL = "https://api-inference.huggingface.co/models/roberta-base-openai-detector"
HF_TOKEN = os.environ.get("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.post("/detect")
async def detect(request: Request):
    data = await request.json()
    text = data.get("text", "")
    result = query({"inputs": text})
    # The result is a list of dicts with 'label' and 'score'
    # Example: [{'label': 'LABEL_0', 'score': 0.98}, {'label': 'LABEL_1', 'score': 0.02}]
    # Let's map LABEL_0 to 'Human-written', LABEL_1 to 'AI-generated'
    if isinstance(result, list) and all('label' in r and 'score' in r for r in result):
        scores = {r['label']: r['score'] for r in result}
        human_score = scores.get('LABEL_0', 0)
        ai_score = scores.get('LABEL_1', 0)
        return {
            "human_score": human_score,
            "ai_score": ai_score,
            "result": "AI-generated" if ai_score > human_score else "Human-written"
        }
    # If the model is loading or returns an error
    return {"error": result}
