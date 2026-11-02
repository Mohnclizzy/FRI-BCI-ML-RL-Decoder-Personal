import argparse
from pathlib import Path

from .baseline import load_config, prepare, train


def main():
    parser = argparse.ArgumentParser(description="Rest vs hand-imagery EEG baseline")
    parser.add_argument("command", choices=["prepare", "train", "run"])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command in ("prepare", "run"):
        prepare(config)
    if args.command in ("train", "run"):
        train(config)


if __name__ == "__main__":
    main()
