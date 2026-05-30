from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


DEFAULT_CHECKPOINT = Path("model/best_resnet18_faces.pth")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class FaceRaceClassifier:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or get_device()

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.class_names = checkpoint["class_names"]
        self.image_size = checkpoint.get("image_size", 224)
        mean = checkpoint.get("imagenet_mean", [0.485, 0.456, 0.406])
        std = checkpoint.get("imagenet_std", [0.229, 0.224, 0.225])

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.model = build_model(len(self.class_names)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

        results = [
            {"class_name": class_name, "probability": float(probabilities[index].item())}
            for index, class_name in enumerate(self.class_names)
        ]
        results.sort(key=lambda row: row["probability"], reverse=True)
        return results
