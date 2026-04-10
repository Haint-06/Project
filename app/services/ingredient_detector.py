import json
from pathlib import Path
from transformers import pipeline


class IngredientDetector:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-image-classification",
            model="google/siglip-base-patch16-224"
        )

        label_path = Path(__file__).parent / "ingredient_labels.json"

        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.candidate_labels = data["ingredients"]

    def detect(self, image):
        results = self.classifier(
            image,
            candidate_labels=self.candidate_labels,
            hypothesis_template="This is a photo of {}."
        )

        ingredients = []

        for item in results[:5]:
            ingredients.append({
                "ingredient": item["label"],
                "score": round(float(item["score"]), 4)
            })
        if not ingredients:
            ingredients.append({
                "ingredient": "Unknown",
                "score": 0.0
            })

        return ingredients