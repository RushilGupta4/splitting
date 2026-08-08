"""Runner-aware training dispatcher."""

import argparse

from runners.registry import get_runner_class, names


def parse_args():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--runner", choices=names())
    known, _ = bootstrap.parse_known_args()

    parser = argparse.ArgumentParser(description="Train a registered runner")
    parser.add_argument("--runner", required=True, choices=names())
    if known.runner is not None:
        get_runner_class(known.runner).add_train_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    get_runner_class(args.runner).train_from_args(args)


if __name__ == "__main__":
    main()
