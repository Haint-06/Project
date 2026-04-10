import io
import torch
import gdown
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from app.services.ingredient_detector import IngredientDetector
from pathlib import Path
from PIL import Image

from app.services.Calories import CalorieCLIP

# Đường dẫn tuyệt đối tính từ vị trí file này — .resolve() đảm bảo không bao giờ là relative path
_HERE    = Path(__file__).resolve().parent
_WEIGHTS = _HERE / "app" / "services" / "weights" / "calorie_clip.pt"
MODEL_URL = "https://drive.google.com/uc?id=1JxC7f7nu41MtrqWgrkk-VgQBqhGF77Wk"

# ── Tải mô hình một lần khi server khởi động
model = None

#khởi tạo model
detector = IngredientDetector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not _WEIGHTS.exists():
        print(f"\n[CalorieCLIP] Chưa có model, đang tải từ Google Drive...")
        _WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

        gdown.download(
            MODEL_URL, 
            str(_WEIGHTS), 
            quiet=False
        )
        print("Đang tải model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[CalorieCLIP] Đang tải mô hình trên {device}...")
    model = CalorieCLIP.from_pretrained(model_path=_WEIGHTS, device=device)
    print("[CalorieCLIP] Sẵn sàng! Truy cập http://localhost:8000\n")
    yield

# ── Khởi tạo app
app = FastAPI(title="CalorieCLIP API", version="1.0.0", lifespan=lifespan)

# ── CORS: cho phép mọi origin (localhost file://, port 7654, v.v.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Serve frontend tĩnh từ app/static/
_static = _HERE / "app" / "static"
_static.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static)), name="static")

@app.get("/")
async def root():
    index = _static / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "CalorieCLIP API đang chạy. Gửi POST /predict với field 'file'."}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Cần UploadFile")

    data = await file.read()

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Không đọc được file ảnh.")

    raw_ingredients = detector.detect(image)
    return {"ingredients": raw_ingredients}

# ── POST /predict
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Nhận ảnh thực phẩm, trả về tên món và calo ước tính.
    Response: { "food": "Greek Salad", "calories": 494.8, "unit": "kcal" }
    """
    if model is None:
        raise HTTPException(503, "Model chưa sẵn sàng")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Cần file ảnh, nhận được: {file.content_type}")

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Không đọc được file ảnh.")

    food     = model.get_food_name(image)
    calories = model.predict(image, use_tta=True)

    ingredient = [
        item for item in detector.detect(image)
    ]

    return {
        "food": food,
        "ingredients": ingredient,
        "calories": round(calories, 1),
        "unit": "kcal"
    }

# ── Health check
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
