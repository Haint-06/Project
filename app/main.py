"""
file model của em quá lớn
em lưu model ở link drive này ạ:
https://drive.google.com/drive/folders/1C5SYsmEFit5A3PrjlTT4Bz9KQ4bH_Gag?usp=sharing

tải xong thầy cô để vào file weights trong app/services/weights ạ
"""



import torch
from pathlib import Path
from app.services.Calories import load_model

def main():
    estimator = load_model(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    image_path = Path("app/assets/salad.jpg")  # đổi ảnh test

    if image_path.exists():
        food_name = estimator.get_food_name(image_path)
        calories = estimator.predict(image_path)

        print(f"\n" + "="*30)
        print(f"Image: {image_path.name}")
        print(f"Food: {food_name}")
        print(f"Calories: {calories:.1f} kcal")
        print("="*30 + "\n")
    else:
        print(f"Image not found: {image_path}")

if __name__ == "__main__":
    main()
