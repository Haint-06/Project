import io
import torch
import gdown
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from PIL import Image

from app.services.ingredient_detector import IngredientDetector
from app.services.Calories import CalorieCLIP

_HERE = Path(__file__).resolve().parent
_WEIGHTS = _HERE / "app" / "services" / "weights" / "calorie_clip.pt"
MODEL_URL = "https://drive.google.com/uc?id=1JxC7f7nu41MtrqWgrkk-VgQBqhGF77Wk"

app = FastAPI(title="CalorieCLIP API", version="1.0.0")

model = None
detector = None


def get_model():
    global model

    if model is None:
        if not _WEIGHTS.exists():
            print("[CalorieCLIP] Chưa có model, đang tải từ Google Drive...")
            _WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

            gdown.download(
                MODEL_URL,
                str(_WEIGHTS),
                quiet=False
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CalorieCLIP] Đang tải mô hình trên {device}...")
        model = CalorieCLIP.from_pretrained(
            model_path=_WEIGHTS,
            device=device
        )
        print("[CalorieCLIP] Model sẵn sàng!")

    return model


def get_detector():
    global detector

    if detector is None:
        print("[Detector] Đang tải ingredient detector...")
        detector = IngredientDetector()
        print("[Detector] Sẵn sàng!")

    return detector


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_static = _HERE / "app" / "static"
_static.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/")
async def root():
    return {"status": "online"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    detector = get_detector()

    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    return {"ingredients": detector.detect(image)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()
    detector = get_detector()

    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    food = model.get_food_name(image)
    calories = model.predict(image, use_tta=True)

    ingredients = detector.detect(image)

    return {
        "food": food,
        "ingredients": ingredients,
        "calories": round(calories, 1),
        "unit": "kcal"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}