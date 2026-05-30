import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CLASSES = [
    "Black",
    "East Asian",
    "Indian",
    "Latino_Hispanic",
    "Middle Eastern",
    "Southeast Asian",
    "White",
]

DEFAULT_TRAIN_TARGETS = {
    "Black": 100,
    "East Asian": 100,
    "Indian": 130,
    "Latino_Hispanic": 100,
    "Middle Eastern": 130,
    "Southeast Asian": 100,
    "White": 100,
}

DEFAULT_EVAL_TARGETS = {class_name: 20 for class_name in CLASSES}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import FairFace images into the local train/val/test folders."
    )
    parser.add_argument(
        "--fairface-dir",
        type=Path,
        default=Path("/Users/paojiaoji/Downloads/fairface-img-margin025-trainval"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_existing(data_dir):
    counts = {split: {} for split in ["train", "val", "test"]}
    for split in counts:
        for class_name in CLASSES:
            class_dir = data_dir / split / class_name
            if class_dir.exists():
                counts[split][class_name] = sum(
                    1 for path in class_dir.iterdir() if path.is_file() and not path.name.startswith(".")
                )
            else:
                counts[split][class_name] = 0
    return counts


def target_counts():
    return {
        "train": DEFAULT_TRAIN_TARGETS,
        "val": DEFAULT_EVAL_TARGETS,
        "test": DEFAULT_EVAL_TARGETS,
    }


def needed_counts(existing, targets):
    needed = {split: {} for split in targets}
    for split, split_targets in targets.items():
        for class_name, target in split_targets.items():
            needed[split][class_name] = max(0, target - existing[split][class_name])
    return needed


def read_fairface_rows(fairface_dir):
    csv_path = fairface_dir / "fairface_label_train.csv"
    rows_by_class = defaultdict(list)
    missing = 0
    ignored = Counter()

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            race = row["race"]
            if race not in CLASSES:
                ignored[race] += 1
                continue

            source_rel = Path(row["file"])
            source_path = fairface_dir / source_rel
            if not source_path.exists():
                missing += 1
                continue

            rows_by_class[race].append(
                {
                    "source_rel": source_rel.as_posix(),
                    "source_path": source_path,
                    "race": race,
                }
            )

    return rows_by_class, missing, ignored


def is_valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def destination_name(source_rel):
    source = Path(source_rel)
    return f"fairface_{source.parent.name}_{source.stem}{source.suffix.lower()}"


def import_images(args):
    existing = count_existing(args.data_dir)
    targets = target_counts()
    needed = needed_counts(existing, targets)
    rows_by_class, missing, ignored = read_fairface_rows(args.fairface_dir)

    rng = random.Random(args.seed)
    for rows in rows_by_class.values():
        rng.shuffle(rows)

    imported = []
    skipped_invalid = 0
    skipped_existing = 0

    for class_name in CLASSES:
        cursor = 0
        candidates = rows_by_class[class_name]

        for split in ["train", "val", "test"]:
            amount = needed[split][class_name]
            copied_for_split = 0

            while copied_for_split < amount and cursor < len(candidates):
                row = candidates[cursor]
                cursor += 1

                if not is_valid_image(row["source_path"]):
                    skipped_invalid += 1
                    continue

                dest_dir = args.data_dir / split / class_name
                dest_name = destination_name(row["source_rel"])
                dest_path = dest_dir / dest_name

                if dest_path.exists():
                    skipped_existing += 1
                    copied_for_split += 1
                else:
                    if not args.dry_run:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(row["source_path"], dest_path)
                    copied_for_split += 1

                imported.append(
                    {
                        "split": split,
                        "class": class_name,
                        "source_file": row["source_rel"],
                        "dest_file": dest_path.as_posix(),
                    }
                )

            if copied_for_split < amount:
                raise RuntimeError(
                    f"Not enough valid FairFace images for {split}/{class_name}: "
                    f"needed {amount}, copied {copied_for_split}."
                )

    if not args.dry_run:
        manifest_path = args.data_dir / "fairface_import_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["split", "class", "source_file", "dest_file"]
            )
            writer.writeheader()
            writer.writerows(imported)

    return {
        "existing_before": existing,
        "targets": targets,
        "needed": needed,
        "imported": imported,
        "missing_source_files": missing,
        "ignored_labels": dict(ignored),
        "skipped_invalid": skipped_invalid,
        "skipped_existing": skipped_existing,
    }


def print_summary(summary):
    print("Existing counts before import:")
    for split, split_counts in summary["existing_before"].items():
        print(f"  {split}: {split_counts}")

    print("\nTarget counts:")
    for split, split_targets in summary["targets"].items():
        print(f"  {split}: {split_targets}")

    print("\nNeeded additions:")
    for split, split_needed in summary["needed"].items():
        print(f"  {split}: {split_needed}")

    imported_counts = Counter((row["split"], row["class"]) for row in summary["imported"])
    print("\nImported counts:")
    for split in ["train", "val", "test"]:
        counts = {class_name: imported_counts[(split, class_name)] for class_name in CLASSES}
        print(f"  {split}: {counts}")

    print(f"\nMissing source files skipped: {summary['missing_source_files']}")
    print(f"Invalid images skipped: {summary['skipped_invalid']}")
    print(f"Existing destination files reused: {summary['skipped_existing']}")
    if summary["ignored_labels"]:
        print(f"Ignored labels: {summary['ignored_labels']}")


def main():
    args = parse_args()
    summary = import_images(args)
    print_summary(summary)


if __name__ == "__main__":
    main()
