import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", type=str, default="data/generated/bigmath_rl_solutions.jsonl"
    )

    return parser.parse_args()


def main(args):

    data = []
    with open(args.file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    accuracy = 0
    for item in data:
        if item["is_correct"]:
            accuracy += 1

    print(f"Accuracy: {accuracy / len(data) * 100}%")


if __name__ == "__main__":

    args = parse_args()
    main(args)
