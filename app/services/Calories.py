import os
import torch
from pathlib import Path
import open_clip
from PIL import Image
import json

class RegressionHead(torch.nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),

            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),

            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class CalorieCLIP(torch.nn.Module):
    def __init__(self, clip_model, preprocess, regression_head, device="cpu"):
        super().__init__()
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.head = regression_head
        self.device = device

        current_dir = os.path.dirname(os.path.abspath(__file__))
        label_path = Path(current_dir, "food_labels.json")

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                self.food_labels = json.load(f)
        else:
            self.food_labels = ["food"] # mặc định nếu bị lỗi

    def get_food_name(self, image_path):
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        text_inputs = [f"food:   {label.replace('_', ' ')}" for label in self.food_labels]
        text_tokens = open_clip.tokenize(text_inputs).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(tensor)
            text_features = self.clip_model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            values, indices = similarity[0].topk(1)

            dectected_label = self.food_labels[indices[0]]
            return dectected_label.replace("_", " ").title()

    @classmethod
    def from_pretrained(cls, model_path="weights/calorie_clip.pt", device="cpu"):
        #load CLIP
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai"
        )

        head = RegressionHead(input_dim=512)
        checkpoint_path = Path(model_path)

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if "regressor_state" in checkpoint:
                head.load_state_dict(checkpoint["regressor_state"])
            else:
                head.load_state_dict(checkpoint)
            

        else:
            print(f"not found checkpoint as {checkpoint_path}")

        model = cls(clip_model, preprocess, head, device=device)
        model.device = device
        model.to(device)
        model.eval()
        return model

    def encode_image(self, image):
        with torch.no_grad():
            return self.clip_model.encode_image(image).float()
        
    def predict(self, image_path):
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
        
        import torchvision.transforms.functional as F
        variants = [
            image,
            F.hflip(image), #lật ngang
            F.rotate(image, 10), #xoay 10 độ
            F.rotate(image, -10) #xoay ngược 10 độ
        ]

        preds = []
        with torch.no_grad():
            for v in variants:
                tensor = self.preprocess(v).unsqueeze(0).to(self.device)
                features = self.encode_image(tensor)
                preds.append(self.head(features).item())
        final_cal = sum(preds)/len(preds)
        return final_cal
    
    def predict_batch(self, image_paths):
        tensors = []
        for img in image_paths:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            tensors.append(self.preprocess(img))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.encode_image(batch).float()
            calories = self.head(features).squeeze(-1)
        return calories.cpu().numpy()

def load_model(device="cpu"):
    return CalorieCLIP.from_pretrained(device=device)
