from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .predictor import predict_sentiment

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Tweet Sentiment Analysis")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "result": None,
            "error": None
        }
    )

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, tweet: str = Form(...)):
    tweet = tweet.strip()

    if not tweet:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "result": None,
                "error": "Please enter a tweet."
            }
        )

    result = predict_sentiment(tweet)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "result": result,
            "error": None
        }
    )

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/predict")
def api_predict(payload: dict):
    text = payload.get("tweet", "").strip()

    if not text:
        return {"error": "Please provide a non-empty tweet."}

    return predict_sentiment(text)