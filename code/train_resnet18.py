import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage ResNet18 transfer learning for 7-class face classification."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    parser.add_argument(
        "--torch-cache",
        type=Path,
        default=None,
        help="Where torchvision should cache pretrained weights. Defaults to output-dir/torch_cache.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--head-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unfreeze",
        choices=["layer4", "layer3_layer4", "all"],
        default="layer4",
        help="Backbone scope to unfreeze in stage 2. layer4 is safest for tiny datasets.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use random initialization instead of ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--freeze-bn-stats",
        action="store_true",
        help="Keep BatchNorm running statistics fixed during training.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.10),
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


def build_loaders(data_dir, image_size, batch_size, num_workers):
    train_transform, eval_transform = build_transforms(image_size)

    datasets_by_split = {
        "train": datasets.ImageFolder(data_dir / "train", transform=train_transform),
        "val": datasets.ImageFolder(data_dir / "val", transform=eval_transform),
        "test": datasets.ImageFolder(data_dir / "test", transform=eval_transform),
    }

    class_to_idx = datasets_by_split["train"].class_to_idx
    for split, split_dataset in datasets_by_split.items():
        if split_dataset.class_to_idx != class_to_idx:
            raise ValueError(
                f"{split} class folders do not match train folders: "
                f"{split_dataset.class_to_idx} != {class_to_idx}"
            )

    loaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    return datasets_by_split, loaders


def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_for_finetune(model, scope):
    for param in model.parameters():
        param.requires_grad = False

    trainable_prefixes = ["fc."]
    if scope == "layer4":
        trainable_prefixes.append("layer4.")
    elif scope == "layer3_layer4":
        trainable_prefixes.extend(["layer3.", "layer4."])
    elif scope == "all":
        for param in model.parameters():
            param.requires_grad = True
        return

    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            param.requires_grad = True


def make_optimizer(model, lr, weight_decay):
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def keep_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def run_one_epoch(model, loader, criterion, device, optimizer=None, update_bn_stats=False):
    is_train = optimizer is not None
    model.train(is_train)
    if is_train and not update_bn_stats:
        keep_batchnorm_eval(model)

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
            predictions = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.numpy().tolist())

    return y_true, y_pred


def fit_stage(
    model,
    loaders,
    criterion,
    device,
    optimizer,
    epochs,
    patience,
    stage_name,
    best_state,
    best_val_acc,
    history,
    update_bn_stats,
):
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            update_bn_stats=update_bn_stats,
        )
        val_loss, val_acc = run_one_epoch(model, loaders["val"], criterion, device)

        record = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(record)

        print(
            f"[{stage_name}] epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    f"[{stage_name}] early stopping: val_acc did not improve "
                    f"for {patience} epochs."
                )
                break

    return best_state, best_val_acc


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def serializable_args(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch_cache = args.torch_cache or (args.output_dir / "torch_cache")
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_cache.resolve()))

    device = get_device()
    print(f"Using device: {device}")

    split_datasets, loaders = build_loaders(
        args.data_dir, args.image_size, args.batch_size, args.num_workers
    )
    class_names = split_datasets["train"].classes
    class_to_idx = split_datasets["train"].class_to_idx

    print("Dataset sizes:")
    for split, split_dataset in split_datasets.items():
        print(f"  {split}: {len(split_dataset)} images")
    print(f"Classes: {class_names}")

    model = build_model(len(class_names), pretrained=not args.no_pretrained)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    print("\nStage 1: freeze backbone, train classifier head")
    freeze_backbone(model)
    optimizer = make_optimizer(model, args.head_lr, args.weight_decay)
    best_state, best_val_acc = fit_stage(
        model,
        loaders,
        criterion,
        device,
        optimizer,
        args.head_epochs,
        args.patience,
        "head",
        best_state,
        best_val_acc,
        history,
        update_bn_stats=not args.freeze_bn_stats,
    )

    print(f"\nStage 2: fine-tune {args.unfreeze} with low learning rate")
    model.load_state_dict(best_state)
    unfreeze_for_finetune(model, args.unfreeze)
    optimizer = make_optimizer(model, args.finetune_lr, args.weight_decay)
    best_state, best_val_acc = fit_stage(
        model,
        loaders,
        criterion,
        device,
        optimizer,
        args.finetune_epochs,
        args.patience,
        "finetune",
        best_state,
        best_val_acc,
        history,
        update_bn_stats=not args.freeze_bn_stats,
    )

    model.load_state_dict(best_state)
    checkpoint_path = args.output_dir / "best_resnet18_faces.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "image_size": args.image_size,
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
            "best_val_acc": best_val_acc,
            "args": serializable_args(args),
        },
        checkpoint_path,
    )

    y_true, y_pred = evaluate_predictions(model, loaders["test"], device)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    ).tolist()

    metrics = {
        "best_val_acc": best_val_acc,
        "test_accuracy": report["accuracy"],
        "class_names": class_names,
        "classification_report": report,
        "confusion_matrix": matrix,
        "args": serializable_args(args),
        "history": history,
    }
    save_json(args.output_dir / "training_metrics.json", metrics)

    print("\nBest validation accuracy:", f"{best_val_acc:.4f}")
    print("Test classification report:")
    print(report_text)
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {args.output_dir / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
