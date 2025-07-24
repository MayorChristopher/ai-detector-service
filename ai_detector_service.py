from fastapi import FastAPI, Request
import requests
import os

app = FastAPI()

API_URL = "https://api-inference.huggingface.co/models/openai-community/roberta-base-openai-detector"
HF_TOKEN = os.environ.get("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        result = response.json()
        # Flatten double list if present
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            result = result[0]
        return result
    except Exception as e:
        print("HuggingFace API error:", response.status_code, response.text)
        if response.status_code == 429:
            return {"error": "Too many requests to HuggingFace API. Please try again later.", "status_code": 429}
        return {"error": f"Failed to parse HuggingFace response: {e}", "status_code": response.status_code, "raw": response.text}

@app.post("/detect")
async def detect(request: Request):
    data = await request.json()
    text = data.get("text", "")
    result = query({"inputs": text})
    if isinstance(result, list) and all('label' in r and 'score' in r for r in result):
        scores = {r['label'].lower(): r['score'] for r in result}
        ai_score = scores.get('fake', 0)
        human_score = scores.get('real', 0)
        return {
            "human_score": human_score,
            "ai_score": ai_score,
            "result": "AI-generated" if ai_score > human_score else "Human-written"
        }
    return {"error": result}