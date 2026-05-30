import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small ResNet18 hyperparameter sweep.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sweep-dir", type=Path, default=Path("model/sweeps"))
    parser.add_argument("--torch-cache", type=Path, default=Path("model/torch_cache"))
    parser.add_argument("--promote-best", action="store_true")
    return parser.parse_args()


def candidate_runs():
    return [
        {
            "name": "seed42_layer4_base",
            "seed": 42,
            "head_lr": 1e-3,
            "finetune_lr": 1e-5,
            "unfreeze": "layer4",
        },
        {
            "name": "seed7_layer4_base",
            "seed": 7,
            "head_lr": 1e-3,
            "finetune_lr": 1e-5,
            "unfreeze": "layer4",
        },
        {
            "name": "seed123_layer4_base",
            "seed": 123,
            "head_lr": 1e-3,
            "finetune_lr": 1e-5,
            "unfreeze": "layer4",
        },
        {
            "name": "seed42_layer4_lower_head",
            "seed": 42,
            "head_lr": 5e-4,
            "finetune_lr": 1e-5,
            "unfreeze": "layer4",
        },
        {
            "name": "seed42_layer4_higher_ft",
            "seed": 42,
            "head_lr": 1e-3,
            "finetune_lr": 3e-5,
            "unfreeze": "layer4",
        },
        {
            "name": "seed42_layer3_layer4",
            "seed": 42,
            "head_lr": 1e-3,
            "finetune_lr": 1e-5,
            "unfreeze": "layer3_layer4",
        },
    ]


def run_training(args, config):
    output_dir = args.sweep_dir / config["name"]
    command = [
        sys.executable,
        "code/train_resnet18.py",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(output_dir),
        "--torch-cache",
        str(args.torch_cache),
        "--seed",
        str(config["seed"]),
        "--head-lr",
        str(config["head_lr"]),
        "--finetune-lr",
        str(config["finetune_lr"]),
        "--unfreeze",
        config["unfreeze"],
    ]
    if config.get("label_smoothing", 0.0):
        command.extend(["--label-smoothing", str(config["label_smoothing"])])
    print(f"\n=== Running {config['name']} ===", flush=True)
    subprocess.run(command, check=True)

    metrics_path = output_dir / "training_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        **config,
        "output_dir": str(output_dir),
        "best_val_acc": metrics["best_val_acc"],
        "test_accuracy": metrics["test_accuracy"],
        "macro_f1": metrics["classification_report"]["macro avg"]["f1-score"],
        "weighted_f1": metrics["classification_report"]["weighted avg"]["f1-score"],
    }


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "seed",
        "head_lr",
        "finetune_lr",
        "unfreeze",
        "label_smoothing",
        "best_val_acc",
        "test_accuracy",
        "macro_f1",
        "weighted_f1",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for config in candidate_runs():
        rows.append(run_training(args, config))
        write_summary(args.sweep_dir / "summary.csv", rows)

    best = max(rows, key=lambda row: (row["best_val_acc"], row["macro_f1"]))
    with (args.sweep_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"best_by_val": best, "runs": rows}, f, ensure_ascii=False, indent=2)

    if args.promote_best:
        best_dir = Path(best["output_dir"])
        shutil.copy2(best_dir / "best_resnet18_faces.pth", Path("model/best_resnet18_faces.pth"))
        shutil.copy2(best_dir / "training_metrics.json", Path("model/training_metrics.json"))

    print("\nSweep summary:")
    for row in sorted(rows, key=lambda item: item["best_val_acc"], reverse=True):
        print(
            f"{row['name']}: val={row['best_val_acc']:.4f} "
            f"test={row['test_accuracy']:.4f} macro_f1={row['macro_f1']:.4f}"
        )
    print(f"\nBest by validation accuracy: {best['name']}")
    print(f"Wrote: {args.sweep_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
