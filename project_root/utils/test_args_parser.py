import argparse
import os
import numpy as np

from config import config
from config.dataset_config import datasets
from utils.arg_validation import Given, Values, Req


class TestArgParser():
    """
    Parser for only three RNN variants: bi-lstm, bi-lstm_attention, and long_term_fusion.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_args()

    def _add_args(self):
        # Main parameters
        self.parser.add_argument(
            'time', type=int,
            help='Time gap to consider'
        )
        self.parser.add_argument(
            'dataset_name', type=str,
            choices=datasets.keys(),
            help='Data set name to use'
        )
        self.parser.add_argument(
            '--model_type', type=str, required=True,
            choices=['bilstm', 'bilstm_attention', 'long_term_fusion'],
            help='Which Bi-LSTM variant to train'
        )
        self.parser.add_argument(
            '--seed', type=int, default=None,
            help='Random seed for reproducibility'
        )
        # Long-term fusion only
        self.parser.add_argument(
            '--hours_out', type=int, choices=[1,2,3],
            help='Forecast horizon hours (fusion only)'
        )
        self.parser.add_argument(
            '--number_of_fusion_weather_layers', type=int,
            help='Number of dense layers in weather branch (fusion only)'
        )

        # Common NN hyperparams
        self.parser.add_argument(
            '--direction', type=str, choices=['forward_only','bidirectional'],
            default='bidirectional', help='LSTM direction'
        )
        self.parser.add_argument(
            '--number_of_rnn_layers', type=int,
            help='Number of LSTM layers'
        )
        self.parser.add_argument(
            '--rnn_layer_size', type=int,
            help='Hidden units per LSTM layer'
        )
        self.parser.add_argument(
            '--number_of_dense_layers', type=int,
            help='Number of final dense layers'
        )
        self.parser.add_argument(
            '--dense_layer_size', type=int,
            help='Units per dense layer'
        )
        self.parser.add_argument(
            '--batch_size', type=int,
            help='Training batch size'
        )
        self.parser.add_argument(
            '--learning_rate', type=float,
            help='Learning rate'
        )
        self.parser.add_argument(
            '--loss', type=str, choices=['mse','haversine'], default='mse',
            help='Loss function'
        )

    def parse(self):
        args = self.parser.parse_args()
        # assign dataset config
        config.dataset_config = datasets[args.dataset_name]
        # validate required combos
        self._validate_args(args)
        return args

    def _validate_args(self, args):
        # Always need these
        reqs = [
            Req(Given('dataset_name')),
            Req(Given('model_type')),
            Req(Given('time')),
            Req(Given('loss')),
        ]
        # fusion-specific requirements
        if args.model_type == 'long_term_fusion':
            reqs += [
                Req(Values('model_type', ['long_term_fusion']), Given('hours_out')),
                Req(Values('model_type', ['long_term_fusion']), Given('number_of_fusion_weather_layers')),
            ]
        # validate all
        for r in reqs:
            r.validate(args)
