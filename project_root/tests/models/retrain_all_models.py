

import json
import numpy as np
from loading.disk_array import DiskArray
from data_loader    import DataLoader
from hypermodel     import RNNFusionHyperModel
import kerastuner    as kt

def load_best_hp(model_type, output_dir):
    with open(f"{output_dir}/best_{model_type}.json") as f:
        vals = json.load(f)
    hp = kt.HyperParameters()
    for k,v in vals.items():
        hp.values[k] = v
    # also force the correct architecture
    hp.values['model_type'] = model_type
    return hp

def main():
    output_dir = "tuner_output"
    # 1) load your full datasets
    dl = DataLoader(config="your_config_module", args=None, hard_reload=False)
    # full train
    X       = dl.load_set('train','train','x', for_analysis=True)
    Y       = dl.load_set('train','train','y', for_analysis=True)
    # full test (for final evaluation)
    X_test  = dl.load_set('test','test','x', for_analysis=True)
    Y_test  = dl.load_set('test','test','y', for_analysis=True)

    # 2) loop over each model variant
    for model_type in ('bilstm','bilstm_attention','long_term_fusion'):
        print(f"\n=== Retraining {model_type} on full data ===")
        hp = load_best_hp(model_type, output_dir)

        # 3) build the model with those hyperparameters
        hyper = RNNFusionHyperModel(
            input_ts_length    = dl.run_config['input_ts_length'],
            input_num_features = dl.run_config['original_x_shape'][-1],
            static_feature_dim = (dl.run_config['input_num_dense_features']
                                  if model_type=='long_term_fusion' else 0),
            output_dim         = len(dl.run_config['y_idxs']),
            normalization_factors = dl.run_config['normalization_factors']
        )
        model = hyper.build(hp)

        # 4) prepare inputs
        if model_type=='long_term_fusion':
            # DataLoader returns tuple (X_seq, X_static) when for_analysis=False
            X_seq, X_static = X
            train_inputs = [X_seq, X_static]
            Xs_test, S_test = X_test
            test_inputs  = [Xs_test, S_test]
        else:
            train_inputs = X
            test_inputs  = X_test

        # 5) train on full dataset
        model.fit(
            train_inputs, Y,
            validation_split=0.1,
            epochs=int(hp.values.get('max_epochs', 50)),  # or a fixed number
            batch_size=hp.values['batch_size'],
            callbacks=[],
            verbose=2
        )

        # 6) save the final model
        model.save(f"final_{model_type}.h5")
        print(f"Saved final_{model_type}.h5")

        # 7) evaluate on your test set
        loss = model.evaluate(test_inputs, Y_test, batch_size=hp.values['batch_size'])
        print(f"{model_type} test loss: {loss}")

if __name__ == "__main__":
    main()
