"""Runner-aware training dispatcher."""

import argparse

from runners.registry import get_runner_class, names


def parse_args():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--runner", default="ddpm_gmm2d")
    known, _ = bootstrap.parse_known_args()
    runner_cls = get_runner_class(known.runner)

    parser = argparse.ArgumentParser(description="Train a registered runner")
    parser.add_argument("--runner", default="ddpm_gmm2d", choices=names())
    runner_cls.add_train_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    get_runner_class(args.runner).train_from_args(args)


if __name__ == "__main__":
    main()
