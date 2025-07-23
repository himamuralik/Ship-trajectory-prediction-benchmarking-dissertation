# run_tuning.py

import math
import argparse
import json
import numpy as np
import kerastuner as kt

from loading.disk_array import DiskArray
from data_loader import DataLoader
from hypermodel import RNNFusionHyperModel


def compute_sample_size(N, confidence=0.95, margin=0.05, p=0.5):
    """
    Compute required sample size for a finite population of size N.
    confidence: e.g. 0.95
    margin: desired error bound (e.g. 0.05 for ±5%)
    p: estimated proportion (0.5 maximizes variance)
    """
    # Z‑scores for common confidences
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    q = 1 - p
    # infinite‑pop estimate
    n0 = (z**2 * p * q) / (margin**2)
    # finite‑population correction
    n = (N * n0) / (n0 + N - 1)
    return int(math.ceil(n))


def main(args):
    # ─── 1) compute full dataset size ───
    dl = DataLoader(config=args.config, args=args, hard_reload=False)
    X = dl.load_set('train', 'train', 'x', for_analysis=True)
    if isinstance(X, DiskArray):
        N = X.shape[0]
    else:
        N = len(X)
    print(f"Total population (trajectories): {N}")

    # ─── 2) sample size ───
    n = compute_sample_size(N,
                            confidence=args.confidence,
                            margin=args.margin)
    print(f"Using sample size: {n}")

    # ─── 3) draw random sample ───
    idx = np.random.choice(N, size=n, replace=False)
    X_train = X[idx]
    Y_full  = dl.load_set('train', 'train', 'y', for_analysis=True)
    Y_train = Y_full[idx]

    if args.model_type == 'long_term_fusion':
        # static features come back as second element
        _, S_full = dl.load_set('train', 'train', 'x', for_analysis=True)
        S_train   = S_full[idx]
        train_data = ([X_train, S_train], Y_train)
    else:
        train_data = (X_train, Y_train)

    # ─── 4) build HyperModel & tuner ───
    hypermodel = RNNFusionHyperModel(
        input_ts_length    = dl.run_config['input_ts_length'],
        input_num_features = dl.run_config['original_x_shape'][-1],
        static_feature_dim = dl.run_config['input_num_dense_features']
                             if args.model_type=='long_term_fusion' else 0,
        output_dim         = len(dl.run_config['y_idxs']),
        normalization_factors = dl.run_config['normalization_factors']
    )

    tuner = kt.Hyperband(
        hypermodel,
        objective   = 'val_loss',
        max_epochs  = args.max_epochs,
        factor      = 3,
        directory   = args.output_dir,
        project_name= 'rnn_fusion_tuning'
    )

    # ─── 5) run search ───
    tuner.search(
        train_data,
        validation_data = args.val_data,
        epochs          = args.max_epochs,
        batch_size      = args.batch_size,
        callbacks       = args.callbacks
    )

    # ─── 6) save overall best ───
    best = tuner.get_best_hyperparameters(num_trials=1)[0]
    with open(f"{args.output_dir}/best_hparams.json","w") as f:
        f.write(best.to_json())
    print("Best hyperparameters:", best.values)

    # ─── 7) dump per‐model_type trials & best ───
    all_trials = list(tuner.oracle.trials.values())
    by_type = {}
    for trial in all_trials:
        hp = trial.hyperparameters.values.copy()
        mt = hp.pop("model_type")
        history = trial.metrics.get_history("val_loss")
        best_val = min(history["val_loss"]) if history and "val_loss" in history else None
        entry = dict(hp, val_loss=best_val)
        by_type.setdefault(mt, []).append(entry)

    for mt, entries in by_type.items():
        entries.sort(key=lambda e: e["val_loss"] if e["val_loss"] is not None else float("inf"))
        # all trials
        with open(f"{args.output_dir}/all_trials_{mt}.json","w") as f:
            json.dump(entries, f, indent=2)
        # best trial
        with open(f"{args.output_dir}/best_{mt}.json","w") as f:
            json.dump(entries[0], f, indent=2)

    print("Wrote all_trials_*.json and best_*.json to", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True,                help='path to your config module')
    parser.add_argument('--confidence', type=float, default=0.95,     help='Confidence level (e.g. .95)')
    parser.add_argument('--margin',     type=float, default=0.05,     help='Margin of error (e.g. .05)')
    parser.add_argument('--model_type', choices=['bilstm','bilstm_attention','long_term_fusion'],
                                         required=True)
    parser.add_argument('--max_epochs', type=int,   default=30)
    parser.add_argument('--batch_size', type=int,   default=512)
    parser.add_argument('--output_dir', type=str,   default='tuner_output')
    parser.add_argument('--val_data',   required=True,
                        help="Your validation data (tf.data.Dataset or tuple)")
    parser.add_argument('--callbacks',  nargs='*',   default=[],
                        help="Any Keras callbacks to pass to tuner.search")

    args = parser.parse_args()
    main(args)
