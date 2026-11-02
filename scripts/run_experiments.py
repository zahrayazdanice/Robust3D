"""Convenience wrapper: python scripts/run_experiments.py --config configs/quick.yaml"""

from robust3d.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["experiment", *__import__("sys").argv[1:]]))

