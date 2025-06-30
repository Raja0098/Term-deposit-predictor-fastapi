from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.model_utils import predict
import os


app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    age: int = Form(...),
    job: str = Form(...),
    marital: str = Form(...),
    education: str = Form(...),
    default: str = Form(...),
    balance: float = Form(...),
    housing: str = Form(...),
    loan: str = Form(...),
    contact: str = Form(...),
    duration: float = Form(...),
    campaign: int = Form(...),
    pdays: int = Form(...),
    previous: int = Form(...),
    poutcome: str = Form(...)
):
    features = [age, job, marital, education, default, balance, housing,
                loan, contact, duration, campaign, pdays, previous, poutcome]
    
    result = predict(features)
    message = "✅ Client **will** subscribe to term deposit." if result == 1 else "❌ Client **will NOT** subscribe to term deposit."
    return templates.TemplateResponse("index.html", {"request": request, "prediction": message})
