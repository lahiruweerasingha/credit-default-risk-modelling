"""Command-line entry point: `python -m creditrisk.cli {features,train,evaluate,all}`."""

from __future__ import annotations

import argparse
import json
import time

from creditrisk.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="creditrisk")
    parser.add_argument("step", choices=["features", "train", "evaluate", "all"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    steps = ["features", "train", "evaluate"] if args.step == "all" else [args.step]
    for step in steps:
        t0 = time.time()
        print(f"=== {step} ===", flush=True)
        if step == "features":
            from creditrisk.features.build import run
        elif step == "train":
            from creditrisk.train import run
        else:
            from creditrisk.evaluate import run
        out = run(cfg)
        print(json.dumps(out, indent=2, default=str)[:4000])
        print(f"=== {step} done in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
