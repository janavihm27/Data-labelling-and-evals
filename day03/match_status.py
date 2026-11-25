
import csv
from iou import iou

CASES = [
    {"id": "a", "existing": (0, 0, 2, 2), "generated": (0, 0, 2, 2), "nobody": False},
    {"id": "b", "existing": (0, 0, 2, 2), "generated": (1, 1, 3, 3), "nobody": False},
    {"id": "c", "existing": (0, 0, 2, 2), "generated": None, "nobody": True},
]

THRESHOLD = 0.5


def assign_status(case):
    if case["nobody"]:
        return None, "no_positive", True

    score = iou(case["existing"], case["generated"])

    if score >= THRESHOLD:
        return score, "matched", True

    return score, "unmatched", False


def main():
    with open("status.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["id", "iou", "status", "trainable"])

        for case in CASES:
            score, status, trainable = assign_status(case)

            writer.writerow([
                case["id"],
                "" if score is None else round(score, 6),
                status,
                trainable
            ])

            print(case["id"], score, status, trainable)


if __name__ == "__main__":
    main()
