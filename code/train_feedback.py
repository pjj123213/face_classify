import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms

from inference import DEFAULT_CHECKPOINT, build_model, get_device


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the model with tester feedback samples.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--feedback-dir", type=Path, default=Path("feedback/images"))
    parser.add_argument("--output", type=Path, default=Path("model/best_resnet18_faces_feedback.pth"))
    parser.add_argument("--metrics-output", type=Path, default=Path("model/feedback_training_metrics.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feedback-repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--promote", action="store_true", help="Overwrite the main checkpoint after training.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


class FeedbackDataset(Dataset):
    def __init__(self, feedback_dir, class_names, transform=None, repeat=1):
        self.feedback_dir = Path(feedback_dir)
        self.class_names = class_names
        self.class_to_idx = {name: index for index, name in enumerate(class_names)}
        self.transform = transform
        self.samples = []
        self.repeat = max(1, repeat)

        for class_name in class_names:
            class_dir = self.feedback_dir / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, index):
        path, label = self.samples[index % len(self.samples)]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def freeze_for_feedback(model):
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith("layer4.") or name.startswith("fc."):
            param.requires_grad = True


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def evaluate_predictions(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_true.extend(labels.numpy().tolist())
    return y_true, y_pred


def main():
    args = parse_args()
    set_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)
    train_transform, eval_transform = build_transforms(image_size)

    feedback_dataset = FeedbackDataset(
        args.feedback_dir,
        class_names,
        transform=train_transform,
        repeat=args.feedback_repeat,
    )
    if len(feedback_dataset.samples) == 0:
        raise SystemExit("No feedback samples found. Save feedback from the desktop app first.")

    train_dataset = datasets.ImageFolder(args.data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(args.data_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(args.data_dir / "test", transform=eval_transform)

    if train_dataset.classes != class_names:
        raise ValueError(f"Class mismatch: {train_dataset.classes} != {class_names}")

    combined_train = ConcatDataset([train_dataset, feedback_dataset])
    train_loader = DataLoader(combined_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = get_device()
    model = build_model(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    freeze_for_feedback(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    history = []

    print(f"Using device: {device}")
    print(f"Base train images: {len(train_dataset)}")
    print(f"Feedback images: {len(feedback_dataset.samples)} x repeat {args.feedback_repeat}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "model_state_dict": model.state_dict(),
            "feedback_training": {
                "feedback_images": len(feedback_dataset.samples),
                "feedback_repeat": args.feedback_repeat,
                "epochs": args.epochs,
                "lr": args.lr,
                "best_val_acc": best_val_acc,
            },
        },
        args.output,
    )

    y_true, y_pred = evaluate_predictions(model, test_loader, device)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "best_val_acc": best_val_acc,
        "test_accuracy": report["accuracy"],
        "class_names": class_names,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
        "history": history,
        "args": vars(args) | {
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "feedback_dir": str(args.feedback_dir),
            "output": str(args.output),
            "metrics_output": str(args.metrics_output),
        },
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.promote:
        torch.save(torch.load(args.output, map_location="cpu"), DEFAULT_CHECKPOINT)
        print(f"Promoted feedback model to {DEFAULT_CHECKPOINT}")

    print(f"Saved feedback checkpoint: {args.output}")
    print(f"Saved metrics: {args.metrics_output}")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test acc: {report['accuracy']:.4f}")


if __name__ == "__main__":
    main()
