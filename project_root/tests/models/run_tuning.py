# run_tuning.py

import math
import argparse
import json                           # ← for dumping JSONs
import numpy as np
import kerastuner as kt
from loading.disk_array import DiskArray
from data_loader import DataLoader
from hypermodel import RNNFusionHyperModel

def compute_sample_size(N, confidence=0.95, margin=0.05, p=0.5):
    """
    Compute required sample size for finite population.

    N          : total population size
    confidence : e.g. 0.95
    margin     : desired margin of error (e.g. 0.05 for ±5%)
    p          : estimated proportion (0.5 gives max variance)
    """
    z_scores = {0.90:1.645, 0.95:1.96, 0.99:2.576}
    z = z_scores.get(confidence, 1.96)
    q = 1 - p
    n0 = (z**2 * p * q) / (margin**2)
    n  = (N * n0) / (n0 + N - 1)
    return int(math.ceil(n))

def main(args):
    # 1) Load dataset size
    dl = DataLoader(config=args.config, args=args, hard_reload=False)
    X = dl.load_set('train', 'train', 'x', for_analysis=True)
    N = X.shape[0] if isinstance(X, DiskArray) else len(X)
    print(f"Total population: {N}")

    # 2) Compute sample size
    sample = compute_sample_size(N, confidence=args.confidence, margin=args.margin)
    print(f"Sampling {sample} trajectories for tuning")

    # 3) Draw a random sample
    idx = np.random.choice(N, size=sample, replace=False)
    Xs = X[idx]
    Ys = dl.load_set('train', 'train', 'y', for_analysis=True)[idx]

    if args.model_type == 'long_term_fusion':
        S = dl.load_set('train','train','x', for_analysis=True)[1]
        Ss = S[idx]
        train_data = ([Xs, Ss], Ys)
    else:
        train_data = (Xs, Ys)

    # 4) Build tuner
    hypermodel = RNNFusionHyperModel(
        input_ts_length=dl.run_config['input_ts_length'],
        input_num_features=dl.run_config['original_x_shape'][-1],
        static_feature_dim=(dl.run_config['input_num_dense_features']
                            if args.model_type=='long_term_fusion' else 0),
        output_dim=len(dl.run_config['y_idxs']),
        normalization_factors=dl.run_config['normalization_factors']
    )
    tuner = kt.Hyperband(
        hypermodel,
        objective='val_loss',
        max_epochs=args.max_epochs,
        factor=3,
        directory=args.output_dir,
        project_name='rnn_fusion_tuning'
    )

    # 5) Search
    tuner.search(
        train_data,
        validation_data=args.val_data,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        callbacks=args.callbacks
    )

    # 6) Save the overall best
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    with open(f"{args.output_dir}/best_hparams.json","w") as f:
        f.write(best_hp.to_json())
    print("Overall best hyperparameters:", best_hp.values)

    # ─── 7) Export every trial and the per‑model‑type bests ───
    all_trials = list(tuner.oracle.trials.values())
    by_type = {}
    for trial in all_trials:
        hp_dict   = trial.hyperparameters.values.copy()
        mtype     = hp_dict.pop("model_type")
        history   = trial.metrics.get_history("val_loss")
        best_val  = min(history["val_loss"]) if history else None
        entry     = {**hp_dict, "val_loss": best_val}
        by_type.setdefault(mtype, []).append(entry)

    for mtype, entries in by_type.items():
        # sort ascending val_loss
        entries.sort(key=lambda e: e["val_loss"] if e["val_loss"] is not None else float("inf"))
        # write all trials for this model_type
        with open(f"{args.output_dir}/all_trials_{mtype}.json", "w") as f:
            json.dump(entries, f, indent=2)
        # write only the single best one
        with open(f"{args.output_dir}/best_{mtype}.json", "w") as f:
            json.dump(entries[0], f, indent=2)

    print(f"Wrote all_trials_*.json and best_*.json into {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      required=True, help='path to your config module')
    parser.add_argument('--confidence',  type=float, default=0.95)
    parser.add_argument('--margin',      type=float, default=0.05)
    parser.add_argument('--model_type',  choices=['bilstm','bilstm_attention','long_term_fusion'], required=True)
    parser.add_argument('--max_epochs',  type=int, default=30)
    parser.add_argument('--batch_size',  type=int, default=512)
    parser.add_argument('--output_dir',  type=str, default='tuner_output')
    parser.add_argument('--val_data',    required=True)  # e.g. tuple or tf.data.Dataset
    parser.add_argument('--callbacks',   nargs='*', default=[])
    args = parser.parse_args()
    main(args)
