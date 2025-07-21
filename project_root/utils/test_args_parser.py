import argparse
import os

import numpy as np

from config import config
from config.dataset_config import datasets
from utils.arg_validation import NotGiven, Given, Values, ValueRange, Req


class TestArgParser():
    """
    Class for parsing arguments for test script with hyperparameter tuning support
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.transformer_only_args = []  # Empty since we're not using transformer models
        self.long_term_only_args = ['hours_out','regularization',
                                  'regularization_application','regularization_coefficient']
        self.fusion_only_args = ['number_of_fusion_weather_layers']
        self.convolution_only_args = []  # Empty since we're not using convolution layers
        self._add_args()

    def _add_args(self):
        """
        Add relevant arguments including hyperparameter tuning options
        """
        # Main params
        self.parser.add_argument('time', type=int, help='Time gap to consider')
        self.parser.add_argument('dataset_name', type=str, choices=datasets.keys(), help='Data set name to use')
        self.parser.add_argument('--model_type', type=str, default='long_term',
                               choices=['long_term', 'long_term_fusion'],
                               help='Model architecture type')
        
        # Hyperparameter tuning mode
        self.parser.add_argument('--tuning_mode', 
                               type=str, 
                               default='none',
                               choices=['none', 'random', 'bayesian'],
                               help='Hyperparameter tuning strategy')
        
        self.parser.add_argument('-s', '--seed', type=str, default='None',
                               help='Random seed to use when creating validation set')
        self.parser.add_argument('--hours_out', type=int, choices=[1, 2, 3],
                               help='Number of hours to predict into the future')

        # Tools for debugging
        self.parser.add_argument('-nl', '--no_logging', action='store_true',
                               help='If included, will not log run to MLflow')
        self.parser.add_argument('-d', '--debug', action='store_true',
                               help='If true, will only use a sample of data when training')

        # Data preprocessing
        self.parser.add_argument('--time_of_day', type=str, choices=['ignore', 'hour_day'], default='ignore',
                               help='Whether or not to include hour of day/day of week as features.')
        self.parser.add_argument('--sog_cog', type=str, default='raw',
                               choices=['ignore','raw','min_median_max','median'],
                               help='The method for including sog/cog')
        self.parser.add_argument('--length_of_history', type=int, default=3,
                               choices=[1,2,3], help='How many hours of history to use')

        # NN Architecture
        self.parser.add_argument('--layer_type', type=str, default=None,
                               choices=['lstm', 'gru', 'sample', None])
        self.parser.add_argument('--direction', type=str, default='sample',
                               choices=['forward_only', 'bidirectional', 'sample'])
        self.parser.add_argument('--number_of_dense_layers', type=int)
        self.parser.add_argument('--dense_layer_size', type=int)
        self.parser.add_argument('--number_of_rnn_layers', type=int)
        self.parser.add_argument('--rnn_layer_size', type=int)
        self.parser.add_argument('--number_of_fusion_weather_layers', type=int)
        self.parser.add_argument('--rnn_to_dense_connection', type=str, default=None,
                               choices=['all_nodes','final_node',None],
                               help='How to connect RNN layers to dense layers')

        # NN Learning
        self.parser.add_argument('--loss', type=str, default='mse',
                               choices=['mse','haversine'])
        self.parser.add_argument('--batch_size', type=int)
        self.parser.add_argument('--learning_rate', type=float)

        # Deprecated args (kept for backward compatibility)
        self.parser.add_argument('--regularization', type=str, choices=['dropout','l1','l2',None], help='DEPRECATED')
        self.parser.add_argument('--regularization_application', type=str,
                               choices=['recurrent','bias','activity',None], help='DEPRECATED')
        self.parser.add_argument('--regularization_coefficient', type=float, help='DEPRECATED')
        self.parser.add_argument('--median_stopping', type=str, default='do_not_use',
                               choices=['do_not_use'], help='DEPRECATED')

    def parse(self):
        """
        Parse arguments with optional hyperparameter tuning
        """
        self.args = self.parser.parse_args()
        config.dataset_config = datasets[self.args.dataset_name]
        config.box_and_year_dir = os.path.join(
            config.data_directory,
            f'{config.dataset_config.lat_1}_{config.dataset_config.lat_2}_'
            f'{config.dataset_config.lon_1}_{config.dataset_config.lon_2}_'
            f'{config.start_year}_{config.end_year}')

        if self.args.no_logging:
            config.logging = False

        if self.args.tuning_mode != 'none':
            self._run_tuning()  # Overwrite args with tuned values
        else:
            self._sample_args()
        
        self._validate_args()
        return self.args

    def _run_tuning(self):
        """
        Execute hyperparameter tuning and update args with best parameters
        """
        if self.args.tuning_mode == 'none':
            return
            
        try:
            from tuning.bi_lstm_tuner import BiLSTMTuner
            tuner = BiLSTMTuner(self.args.dataset_name)
            best_hps = tuner.tune(loader=self._get_data_loader())
            
            # Update args with best hyperparameters
            for hp_name, hp_value in best_hps.values.items():
                setattr(self.args, hp_name, hp_value)
        except ImportError:
            raise ImportError("Tuning module not found. Ensure tuning/ directory exists with proper tuner implementations.")
        except Exception as e:
            raise RuntimeError(f"Hyperparameter tuning failed: {str(e)}")

    def _get_data_loader(self):
        """
        Implement this method to return your data loader for tuning
        """
        raise NotImplementedError("Data loader implementation required for tuning")

    def _sample_args(self):
        """
        Randomly sample hyperparameters if not specified
        """
        if self.args.tuning_mode != 'none':
            return  # Tuner will handle hyperparameters

        # Sample batch size if one is not specified (must be between 128 and 4096, and a power of 2)
        if self.args.batch_size is None:
            self.args.batch_size = np.random.choice([2 ** x for x in [7, 8, 9, 10, 11, 12]])
        # Sample learning rate if one is not given
        if self.args.learning_rate is None:
            self.args.learning_rate = np.exp(np.random.uniform(1, -14))

        if self.args.model_type in ['long_term', 'long_term_fusion']:
            # Sample layer type if one is not specified
            if self.args.layer_type == 'sample':
                self.args.layer_type = np.random.choice(['gru', 'lstm'])
            # Sample direction if one is not specified
            if self.args.direction == 'sample':
                self.args.direction = np.random.choice(['forward_only', 'bidirectional'])
            # Sample number of rnn layers if not given
            if self.args.number_of_rnn_layers is None:
                self.args.number_of_rnn_layers = np.random.randint(1, 6)
            # Sample cell size if not given
            if self.args.rnn_layer_size is None:
                self.args.rnn_layer_size = np.random.randint(50, 351)
            # Sample number of rnn layers if not given
            if self.args.number_of_dense_layers is None:
                self.args.number_of_dense_layers = np.random.randint(0, 3)
            # Sample cell size if not given
            if self.args.dense_layer_size is None:
                self.args.dense_layer_size = np.random.randint(50, 351)
            if self.args.number_of_fusion_weather_layers is None and self.args.model_type == 'long_term_fusion':
                self.args.number_of_fusion_weather_layers = np.random.randint(0, 5)

        if not self.args.seed or self.args.seed == 'None':
            self.args.seed = np.random.randint(1e8)
        else:
            self.args.seed = int(self.args.seed)

    def _validate_args(self):
        """
        Validate argument combinations
        """
        requirements = [
            Req(Given('dataset_name')),
            Req(Given('model_type')),
            Req(Given('time')),
            Req(Given('loss')),
            Req(Given('hours_out')),

            Req(a=Values('model_type', ['long_term', 'long_term_fusion']),
                b=Given('hours_out')),

            Req(a=Values('model_type', ['long_term', 'long_term_fusion']),
                b=Given('layer_type')),

            Req(a=Values('model_type', ['long_term_fusion']),
                b=Given('number_of_fusion_weather_layers')),
            Req(a=Values('model_type', ['long_term']),
                b=NotGiven('number_of_fusion_weather_layers')),

            Req(a=Values('model_type', ['long_term', 'long_term_fusion']),
                b=Given('rnn_to_dense_connection')),

            Req(a=Given('regularization'), b=Given(['regularization_coefficient', 'regularization_application'])),
            Req(a=NotGiven('regularization'), b=NotGiven(['regularization_coefficient', 'regularization_application'])),

            Req(a=Values('regularization', ['dropout', 'l1', 'l2']),
                b=ValueRange('regularization_coefficient', [0, 1])),
            Req(a=Values('regularization', ['dropout']),
                b=Values('regularization_application', ['recurrent', None])),
            Req(a=Values('regularization', ['l1', 'l2']),
                b=Values('regularization_application', ['recurrent', 'bias', 'activity']))
        ]

        for req in requirements:
            req.validate(self.args)
