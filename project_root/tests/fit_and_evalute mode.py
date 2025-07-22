import os
import time
import numpy as np
import tensorflow as tf
from mlflow import log_metric, log_param, log_artifact
from tensorflow.keras.callbacks import EarlyStopping

# Updated imports to match the model type classification
from models.implementation.long_term import LongTermTrajectoryPredictor as LongTermRunner
from models.implementation.long_term_attention import LongTermAttentionTrajectoryPredictor as LongTermAttentionRunner
from models.implementation.long_term_fusion import LongTermFusionTrajectoryPredictor as FusionRunner

from loading.data_loader import DataLoader
from utils.processor_manager import ProcessorManager
from utils.test_args_parser import TestArgParser
import utils.utils as utils
from config import config

def save_predictions(predictions, args, prefix):
    """Universal prediction saver matching original format"""
    np.savetxt(f'{prefix}_predictions.csv', predictions, delimiter=',')
    log_artifact(f'{prefix}_predictions.csv')

def main():
    # Initialize with your existing systems
    parser = TestArgParser()
    args = parser.parse()
    manager = ProcessorManager(save_dir=config.data_directory, debug=args.debug)
    manager.open()
    
    utils.set_seed(args.seed)
    loader = DataLoader(config, args, conserve_memory=True)

    # Data loading with shape validation
    if args.model_type == 'long_term_fusion':
        train_X = loader.load_set('train', 'train', 'x')  # (recurrent, static)
        tf.debugging.assert_shapes([
            (train_X[0], (None, config.length_of_history, None)),
            (train_X[1], (None, len(config.static_columns)))
        ])
        train_Y = loader.load_set('train', 'train', 'y')
        train_data = train_X, train_Y
        valid_X = loader.load_set('valid', 'train', 'x')
        valid_Y = loader.load_set('valid', 'train', 'y')
        valid_data = valid_X, valid_Y
        test_X = loader.load_set('test', 'test', 'x')
    else:  # long_term (with or without attention)
        train_X = loader.load_set('train', 'train', 'x')
        tf.debugging.assert_shapes([
            (train_X, (None, config.length_of_history, None))
        ])
        train_Y = loader.load_set('train', 'train', 'y')
        train_data = train_X, train_Y
        valid_X = loader.load_set('valid', 'train', 'x')
        valid_Y = loader.load_set('valid', 'train', 'y')
        valid_data = valid_X, valid_Y
        test_X = loader.load_set('test', 'test', 'x')

    # Model initialization matching search spaces
    model_params = {
        'input_ts_length': config.length_of_history,
        'output_num_features': 2,  # lat/lon
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size
    }
    
    if args.model_type == 'long_term':
        if args.use_attention:
            runner = LongTermAttentionRunner(
                lstm_hidden_dim=args.rnn_layer_size,
                lstm_num_layers=args.number_of_rnn_layers,
                input_num_features=train_X.shape[-1],
                **model_params
            )
        else:
            runner = LongTermRunner(
                lstm_hidden_dim=args.rnn_layer_size,
                lstm_num_layers=args.number_of_rnn_layers,
                input_num_features=train_X.shape[-1],
                **model_params
            )
    elif args.model_type == 'long_term_fusion':
        runner = FusionRunner(
            rnn_layer_size=args.rnn_layer_size,
            number_of_rnn_layers=args.number_of_rnn_layers,
            input_num_recurrent_features=train_X[0].shape[-1],
            **model_params
        )

    # Training with original callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    runner.fit(train_data, validation_data=valid_data, callbacks=[early_stopping])

    # Prediction handling with all original metrics
    test_results = runner.predict(test_X)
    
    if args.model_type == 'long_term' and args.use_attention:
        log_artifact('attention_weights.npy', test_results['attention_weights'])
    
    # Maintain original metric names
    if 'haversine_distances' in test_results:
        for i, dist in enumerate(test_results['haversine_distances']):
            log_metric(f'haversine_test_loss_{i+1}_hr', float(np.mean(dist)))
    elif 'mean_haversine' in test_results:
        log_metric(f'haversine_test_loss_{args.hours_out}_hr', 
                  float(test_results['mean_haversine']))

    save_predictions(test_results['predictions'], args, 'test')
    manager.close()

if __name__ == '__main__':
    main()
