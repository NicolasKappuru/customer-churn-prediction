"""Train TimeGAN on the w2 sequences and save a synthetic tensor.

Usage (from the repo root or anywhere):

    python workshops/w5/train_timegan.py --label 1 --iterations 10000
    python workshops/w5/train_timegan.py --label 0 --max-rows 20000

TimeGAN is unconditional, so train one model per class (``--label 1`` for
churners, ``--label 0`` for stayers) to get labelled synthetic users. Writes to
``artifacts/timegan/label<k>/`` (see ``models/gan_runner.py``), the folder the
w5 notebook reads. The output uses the same encoding and dtype as the w2 export (float16,
``MODEL_CHANNELS`` order), so it can go through the w2 scalers unchanged.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "models"))

from gan_runner import train_or_load  # noqa: E402
from TimeGAN import timegan  # noqa: E402
from w2_sequences import load_w2_sequences  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", default="fit", help="w2 split part to train on (default: fit)")
    parser.add_argument("--label", type=int, choices=(0, 1), default=None,
                        help="train on one class only (1 = churn); default: all rows")
    parser.add_argument("--max-rows", type=int, default=None, help="subsample the training rows")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="synthetic users to generate (default: as many as training rows)")
    parser.add_argument("--hidden-dim", type=int, default=56)   # 4 x 14 channels
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--module", choices=("gru", "lstm"), default="gru")
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--out", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()

    X, y, specs = load_w2_sequences(args.part, args.label, max_rows=args.max_rows, seed=args.seed)
    params = {
        "hidden_dim": args.hidden_dim,
        "num_layer": args.num_layer,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "module": args.module,
        "seed": args.seed,
        "n_samples": args.n_samples or len(X),
    }
    if args.part != "fit":
        params["part"] = args.part
    run = train_or_load("timegan", timegan, X, specs, params, args.out,
                        label=args.label, retrain=True)
    print("saved", run["dir"] / "synthetic.npy", run["synthetic"].shape)


if __name__ == "__main__":
    main()
