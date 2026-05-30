import argparse
from pathlib import Path

from inference import DEFAULT_CHECKPOINT, FaceRaceClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one face image with a trained checkpoint.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()

def main():
    args = parse_args()
    classifier = FaceRaceClassifier(args.checkpoint)
    results = classifier.predict(args.image)
    top_results = results[: args.top_k]

    print(f"Image: {args.image}")
    print(
        f"Prediction: {top_results[0]['class_name']} "
        f"({top_results[0]['probability']:.4f})"
    )
    print("Top probabilities:")
    for row in top_results:
        print(f"  {row['class_name']}: {row['probability']:.4f}")


if __name__ == "__main__":
    main()
