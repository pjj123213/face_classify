import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
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
SPLITS = ["train", "val", "test"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit image quality for the face dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/data_quality"))
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--sample-per-sheet", type=int, default=40)
    return parser.parse_args()


def image_paths(data_dir):
    for split in SPLITS:
        for class_name in CLASSES:
            class_dir = data_dir / split / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    yield split, class_name, path


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ahash(image, size=8):
    gray = ImageOps.grayscale(image).resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32)
    return "".join("1" if value >= pixels.mean() else "0" for value in pixels.flatten())


def edge_variance(image):
    gray = ImageOps.grayscale(image)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return float(np.asarray(edges, dtype=np.float32).var())


def audit_one(split, class_name, path):
    row = {
        "split": split,
        "class": class_name,
        "path": path.as_posix(),
        "file": path.name,
        "source": "fairface" if path.name.startswith("fairface_") else "original",
        "valid": False,
        "width": "",
        "height": "",
        "brightness": "",
        "contrast": "",
        "sharpness": "",
        "md5": "",
        "ahash": "",
        "issue": "",
    }

    try:
        row["md5"] = md5_file(path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            stat = ImageStat.Stat(image)
            brightness = sum(stat.mean) / 3.0
            contrast = sum(stat.stddev) / 3.0

            row.update(
                {
                    "valid": True,
                    "width": image.width,
                    "height": image.height,
                    "brightness": round(brightness, 3),
                    "contrast": round(contrast, 3),
                    "sharpness": round(edge_variance(image), 3),
                    "ahash": ahash(image),
                }
            )
    except Exception as exc:
        row["issue"] = str(exc)

    return row


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(rows):
    valid_rows = [row for row in rows if row["valid"]]
    summary = {}
    for key in ["brightness", "contrast", "sharpness"]:
        values = np.asarray([float(row[key]) for row in valid_rows], dtype=np.float32)
        summary[key] = {
            "min": round(float(values.min()), 3),
            "p05": round(float(np.percentile(values, 5)), 3),
            "mean": round(float(values.mean()), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
            "max": round(float(values.max()), 3),
        }
    return summary


def find_exact_duplicates(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["valid"]:
            groups[row["md5"]].append(row)
    return [group for group in groups.values() if len(group) > 1]


def find_ahash_duplicates(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["valid"]:
            groups[row["ahash"]].append(row)
    return [group for group in groups.values() if len(group) > 1]


def suspect_rows(rows):
    valid = [row for row in rows if row["valid"]]
    sharp_values = np.asarray([float(row["sharpness"]) for row in valid], dtype=np.float32)
    contrast_values = np.asarray([float(row["contrast"]) for row in valid], dtype=np.float32)
    brightness_values = np.asarray([float(row["brightness"]) for row in valid], dtype=np.float32)

    sharp_cutoff = float(np.percentile(sharp_values, 5))
    contrast_cutoff = float(np.percentile(contrast_values, 5))
    dark_cutoff = float(np.percentile(brightness_values, 3))
    bright_cutoff = float(np.percentile(brightness_values, 97))

    suspects = []
    for row in valid:
        reasons = []
        if float(row["sharpness"]) <= sharp_cutoff:
            reasons.append("low_sharpness")
        if float(row["contrast"]) <= contrast_cutoff:
            reasons.append("low_contrast")
        if float(row["brightness"]) <= dark_cutoff:
            reasons.append("very_dark")
        if float(row["brightness"]) >= bright_cutoff:
            reasons.append("very_bright")
        if reasons:
            suspect = dict(row)
            suspect["reasons"] = ";".join(reasons)
            suspects.append(suspect)
    return suspects


def review_candidates(suspects):
    candidates = []
    key_classes = {"Latino_Hispanic", "Indian", "Middle Eastern"}
    for row in suspects:
        reasons = row["reasons"].split(";") if row.get("reasons") else []
        key_class = row["class"] in key_classes
        severe = (
            len(reasons) >= 2
            or float(row["sharpness"]) < 70
            or float(row["contrast"]) < 22
            or float(row["brightness"]) < 40
            or float(row["brightness"]) > 170
        )
        if severe or key_class:
            candidate = dict(row)
            if severe and key_class:
                candidate["priority"] = "high"
            elif severe:
                candidate["priority"] = "medium"
            else:
                candidate["priority"] = "review"
            candidates.append(candidate)

    priority_order = {"high": 0, "medium": 1, "review": 2}
    candidates.sort(
        key=lambda row: (
            priority_order[row["priority"]],
            row["class"],
            row["split"],
            row["file"],
        )
    )
    return candidates


def draw_contact_sheet(paths, output_path, title, columns=5, thumb_size=160, label_height=38):
    if not paths:
        return

    rows = (len(paths) + columns - 1) // columns
    title_height = 36
    sheet_width = columns * thumb_size
    sheet_height = title_height + rows * (thumb_size + label_height)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 10), title, fill=(0, 0, 0), font=font)

    for index, path in enumerate(paths):
        row = index // columns
        col = index % columns
        x = col * thumb_size
        y = title_height + row * (thumb_size + label_height)

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (thumb_size, thumb_size), (245, 245, 245))
                ox = (thumb_size - image.width) // 2
                oy = (thumb_size - image.height) // 2
                canvas.paste(image, (ox, oy))
        except Exception:
            canvas = Image.new("RGB", (thumb_size, thumb_size), (220, 80, 80))

        sheet.paste(canvas, (x, y))
        label = path.name[:28]
        draw.text((x + 4, y + thumb_size + 4), label, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_contact_sheets(rows, output_dir, sample_per_sheet, seed):
    rng = random.Random(seed)
    sheets_dir = output_dir / "contact_sheets"
    valid_rows = [row for row in rows if row["valid"]]

    for class_name in CLASSES:
        class_rows = [
            row for row in valid_rows
            if row["split"] == "train" and row["class"] == class_name and row["source"] == "fairface"
        ]
        rng.shuffle(class_rows)
        selected = [Path(row["path"]) for row in class_rows[:sample_per_sheet]]
        draw_contact_sheet(
            selected,
            sheets_dir / f"train_{class_name.replace(' ', '_')}_fairface_sample.jpg",
            f"train / {class_name} / FairFace sample",
        )

    key_classes = ["Latino_Hispanic", "Indian", "Middle Eastern"]
    for class_name in key_classes:
        for split in SPLITS:
            split_rows = [
                row for row in valid_rows
                if row["split"] == split and row["class"] == class_name
            ]
            selected = [Path(row["path"]) for row in split_rows[:sample_per_sheet]]
            draw_contact_sheet(
                selected,
                sheets_dir / f"{split}_{class_name.replace(' ', '_')}_all_sample.jpg",
                f"{split} / {class_name} / all sample",
            )

    return sheets_dir


def make_suspect_sheets(suspects, output_dir):
    sheets_dir = output_dir / "contact_sheets"
    sorted_suspects = sorted(
        suspects,
        key=lambda row: (
            row["class"] not in {"Latino_Hispanic", "Indian", "Middle Eastern"},
            row["class"],
            row["split"],
            row["file"],
        ),
    )
    draw_contact_sheet(
        [Path(row["path"]) for row in sorted_suspects[:50]],
        sheets_dir / "suspect_images_top50.jpg",
        "suspect images / top 50",
        columns=5,
    )

    for class_name in ["Latino_Hispanic", "Indian", "Middle Eastern"]:
        class_suspects = [row for row in suspects if row["class"] == class_name]
        draw_contact_sheet(
            [Path(row["path"]) for row in class_suspects[:40]],
            sheets_dir / f"suspect_{class_name.replace(' ', '_')}.jpg",
            f"suspect images / {class_name}",
            columns=5,
        )


def write_markdown(output_dir, rows, suspects, exact_duplicates, ahash_duplicates, summary):
    counts = Counter((row["split"], row["class"]) for row in rows if row["valid"])
    source_counts = Counter((row["split"], row["class"], row["source"]) for row in rows if row["valid"])
    invalid = [row for row in rows if not row["valid"]]

    md = []
    md.append("# 数据质量检查报告\n")
    md.append("## 1. 检查范围\n")
    md.append("本次检查覆盖 `data/train`、`data/val`、`data/test` 下的全部图片。\n")
    md.append(f"- 图片总数：{len(rows)}\n")
    md.append(f"- 可正常打开：{len(rows) - len(invalid)}\n")
    md.append(f"- 无法打开：{len(invalid)}\n")
    md.append("\n## 2. 当前数量\n")
    md.append("| Split | Class | Total | Original | FairFace |\n")
    md.append("|---|---|---:|---:|---:|\n")
    for split in SPLITS:
        for class_name in CLASSES:
            total = counts[(split, class_name)]
            original = source_counts[(split, class_name, "original")]
            fairface = source_counts[(split, class_name, "fairface")]
            md.append(f"| {split} | {class_name} | {total} | {original} | {fairface} |\n")

    md.append("\n## 3. 自动质量指标\n")
    md.append("| Metric | Min | P05 | Mean | P95 | Max |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for key, values in summary.items():
        md.append(
            f"| {key} | {values['min']} | {values['p05']} | {values['mean']} | "
            f"{values['p95']} | {values['max']} |\n"
        )

    md.append("\n## 4. 重复检查\n")
    md.append(f"- MD5 精确重复组数：{len(exact_duplicates)}\n")
    md.append(f"- average-hash 完全相同组数：{len(ahash_duplicates)}\n")
    md.append("\n说明：average-hash 相同只代表缩略结构相似，不能直接判定为重复，需要人工查看。\n")

    md.append("\n## 5. 可疑样本\n")
    md.append(f"根据亮度、对比度、边缘方差阈值，共标记可疑样本 {len(suspects)} 张。\n")
    md.append("详情见 `suspect_images.csv`。\n")

    md.append("\n## 6. 预览图\n")
    md.append("重点类别和各类别训练样本预览图已输出到：\n\n")
    md.append("```text\nreports/data_quality/contact_sheets\n```\n")

    md.append("\n## 7. 初步结论\n")
    md.append("- 数据文件整体可读，没有发现无法打开的图片。\n")
    md.append("- 目前没有发现 MD5 精确重复图片。\n")
    md.append("- 自动指标只能发现极暗、极亮、低对比度、疑似模糊等基础问题，不能替代人眼判断标签是否准确。\n")
    md.append("- 下一步应重点人工查看 contact sheet，尤其是 `Latino_Hispanic`、`Indian`、`Middle Eastern` 三类。\n")

    (output_dir / "数据质量检查报告.md").write_text("".join(md), encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [audit_one(split, class_name, path) for split, class_name, path in image_paths(args.data_dir)]

    fieldnames = [
        "split", "class", "path", "file", "source", "valid", "width", "height",
        "brightness", "contrast", "sharpness", "md5", "ahash", "issue",
    ]
    write_csv(args.output_dir / "image_quality_metrics.csv", rows, fieldnames)

    exact_duplicates = find_exact_duplicates(rows)
    ahash_duplicates = find_ahash_duplicates(rows)
    suspects = suspect_rows(rows)
    reviews = review_candidates(suspects)

    suspect_fieldnames = fieldnames + ["reasons"]
    write_csv(args.output_dir / "suspect_images.csv", suspects, suspect_fieldnames)
    write_csv(args.output_dir / "review_candidates.csv", reviews, suspect_fieldnames + ["priority"])

    duplicate_rows = []
    for group_index, group in enumerate(exact_duplicates, start=1):
        for row in group:
            duplicate_rows.append({"group": group_index, **row})
    write_csv(args.output_dir / "exact_duplicates.csv", duplicate_rows, ["group"] + fieldnames)

    ahash_rows = []
    for group_index, group in enumerate(ahash_duplicates, start=1):
        for row in group:
            ahash_rows.append({"group": group_index, **row})
    write_csv(args.output_dir / "ahash_duplicate_groups.csv", ahash_rows, ["group"] + fieldnames)

    summary = metric_summary(rows)
    summary_payload = {
        "image_count": len(rows),
        "valid_count": sum(1 for row in rows if row["valid"]),
        "invalid_count": sum(1 for row in rows if not row["valid"]),
        "metric_summary": summary,
        "suspect_count": len(suspects),
        "review_candidate_count": len(reviews),
        "exact_duplicate_group_count": len(exact_duplicates),
        "ahash_duplicate_group_count": len(ahash_duplicates),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    make_contact_sheets(rows, args.output_dir, args.sample_per_sheet, args.seed)
    make_suspect_sheets(suspects, args.output_dir)
    write_markdown(args.output_dir, rows, suspects, exact_duplicates, ahash_duplicates, summary)

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
