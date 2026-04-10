
import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

# Bảo vệ import: nếu open_clip chưa cài thì hướng dẫn rõ ràng thay vì crash
try:
    import open_clip
except ImportError:
    raise ImportError("Thiếu thư viện open_clip. Cài bằng lệnh: pip install open-clip-torch")

import torchvision.transforms.functional as TF


# ─── Khối hồi quy (Regression Head)

class RegressionHead(nn.Module):

    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            # Tầng 1: giảm chiều từ 512 → 512, chuẩn hóa, kích hoạt
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),          # Dropout 40% để chống overfit

            # Tầng 2: giảm xuống 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),          # Dropout 30%

            # Tầng 3: giảm xuống 64
            nn.Linear(256, 64),
            nn.ReLU(),

            # Tầng cuối: đầu ra 1 giá trị (số calo dự đoán)
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ─── Mô hình chính

class CalorieCLIP(nn.Module):
    """
    Chỉ số hiệu năng:
      - MAE: ~54.3 calo
      - 60.7% dự đoán sai lệch dưới 50 calo
      - 81.5% dự đoán sai lệch dưới 100 calo

    """

    def __init__(self, clip_model, preprocess, regression_head, device="cpu"):
        super().__init__()
        # Lưu bộ mã hóa CLIP (ViT-B/32)
        self.clip_model = clip_model
        # Hàm tiền xử lý ảnh chuẩn của CLIP (resize, normalize...)
        self.preprocess = preprocess
        # Đầu hồi quy dự đoán calo
        self.head = regression_head
        # Thiết bị tính toán: "cpu" hoặc "cuda"
        self.device = device

        # Tải danh sách nhãn món ăn từ file JSON cùng thư mục
        current_dir = os.path.dirname(os.path.abspath(__file__))
        label_path = Path(current_dir) / "food_labels.json"

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                self.food_labels = json.load(f)
            print(f"Loaded {len(self.food_labels)} food labels")
        else:
            # Dự phòng: nếu không có file nhãn thì dùng nhãn chung
            self.food_labels = ["food"]
            print("Cảnh báo: không tìm thấy food_labels.json — dùng nhãn mặc định")

    # ── Tải mô hình từ file checkpoint

    @classmethod
    def from_pretrained(cls, model_path=None, device="cpu"):
        """
        Tải CalorieCLIP từ file checkpoint .pt.

        Tham số:
            model_path: đường dẫn đến file .pt, hoặc thư mục chứa file.
                        Nếu None → tự động tìm trong thư mục weights/ cùng cấp.
            device:     "cpu" hoặc "cuda"
        """
        # ── Xác định đường dẫn checkpoint
        if model_path is None:
            # Mặc định: tìm file trong thư mục weights/ cùng thư mục với file này
            services_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
            model_path = services_dir / "weights" / "calorie_clip.pt"
        else:
            model_path = Path(model_path).resolve()

        print(f"[CalorieCLIP] Đường dẫn checkpoint: {model_path}")

        # Nếu đường dẫn trỏ vào thư mục (không phải file),
        # tìm file calorie_clip.pt hoặc best_model.pt bên trong
        if model_path.is_dir():
            # Đọc config nếu tồn tại (theo định dạng HuggingFace)
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            else:
                config = {"base_model": "ViT-B-32", "pretrained": "openai"}

            # Ưu tiên calorie_clip.pt, fallback sang best_model.pt
            weights_file = model_path / "calorie_clip.pt"
            if not weights_file.exists():
                weights_file = model_path / "best_model.pt"
        else:
            # Đường dẫn trực tiếp đến file .pt (dùng phổ biến trong local dev)
            weights_file = model_path
            config = {"base_model": "ViT-B-32", "pretrained": "openai"}

        # ── Tải CLIP backbone
        # create_model_and_transforms trả về: (model, train_transform, val_transform)
        # Chỉ cần val_transform (preprocess) cho inference
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            config.get("base_model", "ViT-B-32"),
            pretrained=config.get("pretrained", "openai")
        )

        # ── Khởi tạo đầu hồi quy ────────────────────────────────────────────
        head = RegressionHead(input_dim=512)

        # ── Nạp trọng số từ checkpoint ───────────────────────────────────────
        if weights_file.exists():
            checkpoint = torch.load(weights_file, map_location=device, weights_only=False)

            # [FIX từ reference] Nạp lại trọng số CLIP nếu được fine-tune
            # Trong Calories.py gốc, bước này bị bỏ qua → CLIP chạy bằng
            # trọng số OpenAI gốc thay vì bản đã được tinh chỉnh
            if "clip_state" in checkpoint:
                clip_model.load_state_dict(checkpoint["clip_state"], strict=False)
                print("Đã nạp clip_state (CLIP fine-tuned weights)")

            # Nạp trọng số đầu hồi quy — hỗ trợ cả 2 tên key khác nhau
            if "regressor_state" in checkpoint:
                head.load_state_dict(checkpoint["regressor_state"])
                print(f"Đã nạp regressor_state từ: {weights_file}")
            elif "head_state" in checkpoint:
                head.load_state_dict(checkpoint["head_state"])
                print(f"Đã nạp head_state từ: {weights_file}")
            else:
                # Fallback: giả sử toàn bộ checkpoint là state_dict của head
                head.load_state_dict(checkpoint)
                print(f"Đã nạp trọng số từ: {weights_file}")

            # In sai số trung bình nếu được lưu trong checkpoint
            if "mae" in checkpoint:
                print(f"Sai số trung bình: {checkpoint['mae']:.2f} kcal")
        else:
            print(f"Cảnh báo: không tìm thấy checkpoint tại {weights_file}")
            print("Mô hình sẽ chạy với trọng số ngẫu nhiên — kết quả sẽ sai!")

        # ── Lắp ráp và trả về mô hình hoàn chỉnh ────────────────────────────
        model = cls(clip_model, preprocess, head, device=device)
        model.to(device)
        model.eval()  # Chuyển sang chế độ inference (tắt BatchNorm/Dropout stochastic)
        return model

    # ── Mã hóa ảnh thành vector đặc trưng ───────────────────────────────────

    def encode_image(self, image_tensor):

        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensor).float()
        return features

    # ── Phân loại tên món ăn (zero-shot) ────────────────────────────────────

    def get_food_name(self, image_path):
        # Tải và tiền xử lý ảnh
        # Đảm bảo image luôn là PIL Image RGB trước khi đưa vào preprocess
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Xây dựng prompt văn bản cho từng nhãn
        # Dùng tiền tố "food:" để định ngữ cảnh cho CLIP
        text_inputs = [f"food: {label.replace('_', ' ')}" for label in self.food_labels]
        text_tokens = open_clip.tokenize(text_inputs).to(self.device)

        with torch.no_grad():
            # Mã hóa ảnh và văn bản
            image_features = self.clip_model.encode_image(tensor)
            text_features = self.clip_model.encode_text(text_tokens)

            # Chuẩn hóa L2 để tính cosine similarity
            # (chỉ dùng trong phân loại zero-shot, KHÔNG dùng khi dự đoán calo)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Tính điểm tương đồng và chuyển sang xác suất (softmax)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Lấy nhãn có điểm cao nhất
            _, top_index = similarity[0].topk(1)

        detected_label = self.food_labels[top_index[0]]
        # Trả về tên dễ đọc: gạch dưới → khoảng trắng, viết hoa chữ đầu
        return detected_label.replace("_", " ").title()

    # ── Dự đoán calo (single image + TTA) ───────────────────────────────────

    def predict(self, image_path, use_tta=True):

        # Tải ảnh
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

        if use_tta:
            # Tạo 4 biến thể ảnh để TTA
            variants = [
                image,                    # ảnh gốc
                TF.hflip(image),          # lật ngang (mirror)
                TF.rotate(image, 10),     # xoay +10°
                TF.rotate(image, -10),    # xoay −10°
            ]

            # Dự đoán riêng cho từng biến thể, rồi lấy trung bình
            preds = []
            with torch.no_grad():
                for v in variants:
                    tensor = self.preprocess(v).unsqueeze(0).to(self.device)
                    features = self.encode_image(tensor)
                    preds.append(self.head(features).item())

            return sum(preds) / len(preds)   # trung bình cộng 4 dự đoán

        else:
            # Chế độ nhanh: chỉ dự đoán một lần (theo reference gốc)
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.encode_image(tensor)
                calories = self.head(features).item()
            return calories

    # ── Dự đoán calo theo batch ──────────────────────────────────────────────

    def predict_batch(self, image_paths):

        tensors = []
        for img in image_paths:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            tensors.append(self.preprocess(img))

        # Stack thành một batch duy nhất rồi đưa lên device
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self.encode_image(batch)
            calories = self.head(features).squeeze(-1)

        return calories.cpu().numpy()

    def forward(self, image_tensor):
        """
        Forward pass chuẩn PyTorch: tensor ảnh → calo dự đoán.
        Dùng khi tích hợp mô hình vào pipeline huấn luyện.
        """
        features = self.encode_image(image_tensor)
        return self.head(features).squeeze(-1)


# ─── Hàm tiện ích ───────────────────────────────────────────────────────────

def load_model(device="cpu"):

    # Xây dựng đường dẫn tuyệt đối để không phụ thuộc vào thư mục làm việc hiện tại
    weights_path = Path(os.path.dirname(os.path.abspath(__file__))) / "weights" / "calorie_clip.pt"
    return CalorieCLIP.from_pretrained(model_path=weights_path, device=device)
