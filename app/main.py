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

        print(f"Image: {image_path.name}")
        print(f"Food: {food_name}")
        print(f"Calories: {calories:.1f} kcal")
    else:
        print(f"Image not found: {image_path}")

if __name__ == "__main__":
    main()
