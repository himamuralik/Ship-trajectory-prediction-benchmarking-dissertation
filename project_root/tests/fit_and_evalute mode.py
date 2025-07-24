#!/usr/bin/env python
import sys
import os
import time
import atexit
import json

# must set PYTHONHASHSEED before TF imports
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow

from mlflow import log_param, log_artifact, log_metric
from tensorflow.keras.callbacks import EarlyStopping

# our utils & models
from utils import ProcessorManager, TestArgParser, set_seed
from loading.data_loader import DataLoader
import loading
import models


def save_predictions_and_errors(preds, errors, args, which):
    """
    Save predictions & errors to CSV artifacts.
    """
    # For bilstm & bilstm_attention we predict 3 hours out (index 0,1,2)
    if args.model_type in ('bilstm','bilstm_attention'):
        for i, (p, e) in enumerate(zip(preds, errors), start=1):
            fn_err = f'{which}_haversine_error_{i}_hour.csv'
            fn_pred = f'{which}_predictions_hour_{i}.csv'
            np.savetxt(fn_err, e, delimiter=',')
            np.savetxt(fn_pred, p, delimiter=',')
            log_artifact(fn_err)
            log_artifact(fn_pred)

    # For long_term_fusion we only have one horizon at args.hours_out
    else:  # long_term_fusion
        fn_err = f'{which}_haversine_error_{args.hours_out}_hour.csv'
        fn_pred = f'{which}_predictions_hour_{args.hours_out}.csv'
        np.savetxt(fn_err, errors[0], delimiter=',')
        np.savetxt(fn_pred, preds[0], delimiter=',')
        log_artifact(fn_err)
        log_artifact(fn_pred)


if __name__ == '__main__':
    # ─── parse & lock a processor ───────────────────
    parser = TestArgParser()
    # ⚠️ Make sure TestArgParser has been updated to:
    #    choices=['bilstm','bilstm_attention','long_term_fusion']
    args = parser.parse()

    manager = ProcessorManager(debug=args.debug)
    manager.open()

    if args.debug:
        mlflow.set_experiment(experiment_name='Ships Debugging')

    # ─── seed & load data ───────────────────────────
    set_seed(args.seed)
    loader = DataLoader(config=__import__('config').config,
                        args=args,
                        conserve_memory=True)

    train_Y = loader.load_set('train','train','y')
    train_X = loader.load_set('train','train','x')
    valid_Y = loader.load_set('valid','train','y')
    valid_X = loader.load_set('valid','train','x')

    # ─── log run parameters ─────────────────────────
    if loader.config.logging:
        loader.run_config.save_to_dir('run_config', register_with_mlflow=True)
        log_param('model_type', args.model_type)
        log_param('layer_type', args.layer_type)
        log_param('direction', args.direction)
        log_param('number_of_rnn_layers', args.number_of_rnn_layers)
        log_param('rnn_layer_size', args.rnn_layer_size)
        log_param('number_of_dense_layers', args.number_of_dense_layers)
        log_param('dense_layer_size', args.dense_layer_size)
        log_param('batch_size', args.batch_size)
        log_param('learning_rate', args.learning_rate)
        if args.model_type == 'long_term_fusion':
            log_param('number_of_fusion_weather_layers', args.number_of_fusion_weather_layers)
            log_param('hours_out', args.hours_out)

        # redirect stdout/stderr into MLflow
        stdout_fp = mlflow.get_artifact_uri().replace('file://','') + '/stdout.txt'
        stderr_fp = mlflow.get_artifact_uri().replace('file://','') + '/stderr.txt'
        sys.stdout = open(stdout_fp, 'w', 1)
        sys.stderr = open(stderr_fp, 'w', 1)
        atexit.register(lambda: sys.stdout.close())
        atexit.register(lambda: sys.stderr.close())

    # ─── pick & build the right runner ──────────────
    with tf.device(manager.device()):
        if args.model_type == 'bilstm':
            runner = models.BiLSTMRunner(
                node_type=args.layer_type,
                number_of_rnn_layers=args.number_of_rnn_layers,
                rnn_layer_size=args.rnn_layer_size,
                direction=args.direction,
                input_ts_length=loader.run_config['input_ts_length'],
                input_num_features=int(train_X.shape[2]),
                output_num_features=len(loader.run_config['y_idxs']),
                normalization_factors=loader.run_config['normalization_factors'],
                y_idxs=loader.run_config['y_idxs'],
                columns=loader.run_config['columns'],
                learning_rate=args.learning_rate,
                loss=args.loss,
            )

        elif args.model_type == 'bilstm_attention':
            runner = models.BiLSTMAttentionRunner(
                node_type=args.layer_type,
                number_of_rnn_layers=args.number_of_rnn_layers,
                rnn_layer_size=args.rnn_layer_size,
                direction=args.direction,
                input_ts_length=loader.run_config['input_ts_length'],
                input_num_features=int(train_X.shape[2]),
                output_num_features=len(loader.run_config['y_idxs']),
                normalization_factors=loader.run_config['normalization_factors'],
                y_idxs=loader.run_config['y_idxs'],
                columns=loader.run_config['columns'],
                learning_rate=args.learning_rate,
                loss=args.loss,
            )

        elif args.model_type == 'long_term_fusion':
            runner = models.FusionModelRunner(
                node_type=args.layer_type,
                number_of_rnn_layers=args.number_of_rnn_layers,
                rnn_layer_size=args.rnn_layer_size,
                number_of_final_dense_layers=args.number_of_dense_layers,
                number_of_fusion_weather_layers=args.number_of_fusion_weather_layers,
                dense_layer_size=args.dense_layer_size,
                direction=args.direction,
                input_ts_length=loader.run_config['input_ts_length'],
                input_num_recurrent_features=int(len(loader.run_config['recurrent_idxs'])),
                weather_shape=train_X[1].shape,
                output_num_features=len(loader.run_config['y_idxs']),
                normalization_factors=loader.run_config['normalization_factors'],
                y_idxs=loader.run_config['y_idxs'],
                columns=loader.run_config['columns'],
                learning_rate=args.learning_rate,
                loss=args.loss,
                recurrent_idxs=loader.run_config['recurrent_idxs'],
                fusion_layer_structure=args.fusion_layer_structure,
            )

        else:
            raise ValueError(f"Unknown model_type {args.model_type}")

        # compile & fit
        runner.compile()
        if loader.config.logging:
            mlflow.tensorflow.autolog(log_models=True)

        es_patience = 3 if args.debug else 30
        es = EarlyStopping(monitor='val_loss',
                           patience=es_patience,
                           restore_best_weights=True)

        # wrap into TF Dataset or Sequence
        train_data = loading.DataGenerator(train_X, train_Y, args.batch_size, shuffle=True)
        valid_data = loading.DataGenerator(valid_X, valid_Y, args.batch_size, shuffle=False)

        history = runner.fit(
            train_data,
            validation_data=valid_data,
            epochs=1000,
            batch_size=args.batch_size,
            callbacks=[es],
            verbose=2
        )

        # ─── final eval & save ───────────────────────
        pred_val, err_val, _ = runner.predict(**{
            'X': loader.load_set('valid','test','x'),
            'Y': loader.load_set('valid','test','y'),
            'args': args
        })
        save_predictions_and_errors(pred_val, err_val, args, 'validation')

        pred_test, err_test, _ = runner.predict(**{
            'X': loader.load_set('test','test','x'),
            'Y': loader.load_set('test','test','y'),
            'args': args
        })
        save_predictions_and_errors(pred_test, err_test, args, 'test')

    manager.close()
