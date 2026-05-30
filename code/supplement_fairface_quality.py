import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat


CLASSES = [
    "Black",
    "East Asian",
    "Indian",
    "Latino_Hispanic",
    "Middle Eastern",
    "Southeast Asian",
    "White",
]

TARGETS = {
    "train": {
        "Black": 100,
        "East Asian": 100,
        "Indian": 130,
        "Latino_Hispanic": 100,
        "Middle Eastern": 130,
        "Southeast Asian": 100,
        "White": 100,
    },
    "val": {class_name: 20 for class_name in CLASSES},
    "test": {class_name: 20 for class_name in CLASSES},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supplement missing slots with stricter quality-filtered FairFace images."
    )
    parser.add_argument(
        "--fairface-dir",
        type=Path,
        default=Path("/Users/paojiaoji/Downloads/fairface-img-margin025-trainval"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/data_quality"))
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--commit", action="store_true")
    return parser.parse_args()


def count_current(data_dir):
    counts = {}
    for split, split_targets in TARGETS.items():
        counts[split] = {}
        for class_name in split_targets:
            class_dir = data_dir / split / class_name
            counts[split][class_name] = sum(
                1 for path in class_dir.iterdir()
                if path.is_file() and not path.name.startswith(".")
            )
    return counts


def needed_counts(counts):
    needed = {}
    for split, split_targets in TARGETS.items():
        needed[split] = {}
        for class_name, target in split_targets.items():
            needed[split][class_name] = max(0, target - counts[split][class_name])
    return needed


def load_used_sources(data_dir, report_dir):
    used = set()
    manifest = data_dir / "fairface_import_manifest.csv"
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                used.add(row["source_file"])

    review = report_dir / "review_candidates.csv"
    if review.exists():
        with review.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                filename = row.get("file", "")
                if filename.startswith("fairface_train_"):
                    source_id = filename.removeprefix("fairface_train_").removesuffix(".jpg")
                    used.add(f"train/{source_id}.jpg")
    return used


def read_fairface_rows(fairface_dir):
    rows = defaultdict(list)
    with (fairface_dir / "fairface_label_train.csv").open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            race = row["race"]
            if race not in CLASSES:
                continue
            source_rel = row["file"]
            source_path = fairface_dir / source_rel
            if source_path.exists():
                rows[race].append((source_rel, source_path))
    return rows


def edge_variance(image):
    gray = ImageOps.grayscale(image)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return float(np.asarray(edges, dtype=np.float32).var())


def image_metrics(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        stat = ImageStat.Stat(image)
        brightness = sum(stat.mean) / 3.0
        contrast = sum(stat.stddev) / 3.0
        hsv = image.convert("HSV")
        saturation = float(np.asarray(hsv, dtype=np.uint8)[:, :, 1].mean())
        arr = np.asarray(image, dtype=np.float32)
        channel_spread = float(
            np.mean(
                np.abs(arr[:, :, 0] - arr[:, :, 1])
                + np.abs(arr[:, :, 1] - arr[:, :, 2])
                + np.abs(arr[:, :, 0] - arr[:, :, 2])
            )
            / 3.0
        )
        gray_small = np.asarray(
            ImageOps.grayscale(image).resize((64, 64), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        left = gray_small[:, :32]
        right = np.fliplr(gray_small[:, 32:])
        symmetry_error = float(np.mean(np.abs(left - right)))
        return {
            "width": image.width,
            "height": image.height,
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": edge_variance(image),
            "saturation": saturation,
            "channel_spread": channel_spread,
            "symmetry_error": symmetry_error,
        }


def passes_quality(metrics):
    return (
        65.0 <= metrics["brightness"] <= 138.0
        and metrics["contrast"] >= 35.0
        and metrics["sharpness"] >= 240.0
        and metrics["saturation"] >= 32.0
        and metrics["channel_spread"] >= 8.0
        and metrics["symmetry_error"] <= 50.0
    )


def quality_score(metrics):
    brightness_score = max(0.0, 1.0 - abs(metrics["brightness"] - 100.0) / 45.0)
    contrast_score = min(metrics["contrast"] / 65.0, 1.0)
    sharpness_score = min(metrics["sharpness"] / 900.0, 1.0)
    saturation_score = min(metrics["saturation"] / 85.0, 1.0)
    symmetry_score = max(0.0, 1.0 - metrics["symmetry_error"] / 50.0)
    return (
        0.25 * brightness_score
        + 0.20 * contrast_score
        + 0.25 * sharpness_score
        + 0.10 * saturation_score
        + 0.20 * symmetry_score
    )


def destination_name(source_rel):
    source = Path(source_rel)
    return f"fairface_replenish_{source.parent.name}_{source.stem}{source.suffix.lower()}"


def select_candidates(args, needed):
    used = load_used_sources(args.data_dir, args.report_dir)
    rows_by_class = read_fairface_rows(args.fairface_dir)
    rng = random.Random(args.seed)
    for rows in rows_by_class.values():
        rng.shuffle(rows)

    selected = []
    rejected = defaultdict(int)

    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            amount = needed[split].get(class_name, 0)
            if amount <= 0:
                continue

            pool = []
            for source_rel, source_path in rows_by_class[class_name]:
                if len(pool) >= max(80, amount * 20):
                    break
                if source_rel in used:
                    rejected["already_used_or_flagged"] += 1
                    continue
                try:
                    metrics = image_metrics(source_path)
                except Exception:
                    rejected["invalid"] += 1
                    continue
                if not passes_quality(metrics):
                    rejected["quality_filter"] += 1
                    continue

                dest = args.data_dir / split / class_name / destination_name(source_rel)
                pool.append(
                    {
                        "split": split,
                        "class": class_name,
                        "source_file": source_rel,
                        "source_path": source_path.as_posix(),
                        "dest_file": dest.as_posix(),
                        "quality_score": round(quality_score(metrics), 4),
                        **{key: round(value, 3) for key, value in metrics.items()},
                    }
                )

            pool.sort(key=lambda row: row["quality_score"], reverse=True)
            chosen = pool[:amount]
            for row in chosen:
                used.add(row["source_file"])
                selected.append(row)

            if len(chosen) < amount:
                raise RuntimeError(
                    f"Not enough strict candidates for {split}/{class_name}: "
                    f"needed {amount}, selected {len(chosen)}."
                )

            pool_sheet_rows = pool[: min(60, len(pool))]
            if pool_sheet_rows:
                pool_dir = args.report_dir / "replenish" / "candidate_pools"
                write_csv(
                    pool_dir / f"{split}_{class_name.replace(' ', '_')}_pool.csv",
                    pool_sheet_rows,
                )
                draw_contact_sheet(
                    pool_sheet_rows,
                    pool_dir / f"{split}_{class_name.replace(' ', '_')}_pool.jpg",
                    f"{split} / {class_name} / strict candidate pool",
                )

    return selected, rejected


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_contact_sheet(rows, output_path, title, columns=5, thumb_size=170, label_height=42):
    if not rows:
        return
    sheet_rows = (len(rows) + columns - 1) // columns
    title_height = 36
    sheet = Image.new(
        "RGB",
        (columns * thumb_size, title_height + sheet_rows * (thumb_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 10), title, fill=(0, 0, 0), font=font)

    for index, row in enumerate(rows):
        path = Path(row["source_path"])
        y = title_height + (index // columns) * (thumb_size + label_height)
        x = (index % columns) * thumb_size
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (thumb_size, thumb_size), (245, 245, 245))
            canvas.paste(image, ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2))
        sheet.paste(canvas, (x, y))
        label = f"{row['split']} {row['class'][:10]} {Path(row['source_file']).name}"
        draw.text((x + 4, y + thumb_size + 4), label, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def commit_rows(rows, manifest_path):
    for row in rows:
        source = Path(row["source_path"])
        dest = Path(row["dest_file"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

    manifest_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "source_file", "dest_file"])
        if not manifest_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "class": row["class"],
                    "source_file": row["source_file"],
                    "dest_file": row["dest_file"],
                }
            )


def main():
    args = parse_args()
    counts = count_current(args.data_dir)
    needed = needed_counts(counts)
    selected, rejected = select_candidates(args, needed)

    output_dir = args.report_dir / "replenish"
    write_csv(output_dir / "selected_replenish_candidates.csv", selected)
    draw_contact_sheet(
        selected,
        output_dir / "selected_replenish_candidates.jpg",
        "strict FairFace replenish candidates",
    )

    if args.commit:
        commit_rows(selected, args.data_dir / "fairface_import_manifest.csv")

    print("Current counts:", counts)
    print("Needed:", needed)
    print("Selected:", len(selected))
    print("Rejected:", dict(rejected))
    print("Committed:", args.commit)
    print(f"Wrote: {output_dir / 'selected_replenish_candidates.csv'}")
    print(f"Wrote: {output_dir / 'selected_replenish_candidates.jpg'}")


if __name__ == "__main__":
    main()
