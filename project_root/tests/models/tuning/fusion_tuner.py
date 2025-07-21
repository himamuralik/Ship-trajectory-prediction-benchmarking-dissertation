import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_tuner import BaseTuner
from models.implementations.bi_lstm_fusion import BiLSTMFusionModelRunner

class FusionTuner(BaseTuner):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name, 'bi_lstm_fusion')
        # Load actual vessel group columns from your data
        self.vessel_cols = [f'vessel_group_{i}' for i in range(5)]  # Replace with actual
        self.mock_columns = pd.DataFrame({'column': self.vessel_cols})
        self.norm_factors = None
        self.recurrent_features = None
        
    def build_model(self, hp: kt.HyperParameters) -> tf.keras.Model:
        params = self.load_search_space()
        
        # Create model runner instance with updated parameters
        model_runner = BiLSTMFusionModelRunner(
            number_of_rnn_layers=hp.Int('lstm_num_layers', 
                                      min_value=params['lstm_num_layers']['min'],
                                      max_value=params['lstm_num_layers']['max']),
            rnn_layer_size=hp.Int('lstm_hidden_dim', 
                                 values=params['lstm_hidden_dim']['values']),
            number_of_final_dense_layers=hp.Int('num_dense_layers', 
                                             min_value=params['num_dense_layers']['min'],
                                             max_value=params['num_dense_layers']['max']),
            dense_layer_size=hp.Int('dense_layer_size', 
                                  values=params['dense_layer_size']['values']),
            input_ts_length=24,  # Match your sequence length
            input_num_recurrent_features=len(self.recurrent_features),  # Actual feature count
            output_num_features=2,  # lat/lon
            normalization_factors=self.norm_factors,  # Loaded from data
            y_idxs=[0, 1],  # Assuming first two are lat/lon
            columns=self.mock_columns,  # Or actual columns
            learning_rate=hp.Float('learning_rate', 
                                 min_value=params['learning_rate']['values'][0],
                                 max_value=params['learning_rate']['values'][-1],
                                 sampling='log'),
            recurrent_idxs=list(range(len(self.recurrent_features))),  # Actual idxs
            dropout=hp.Float('dropout', 
                            min_value=params['dropout']['min'],
                            max_value=params['dropout']['max']),
            regularization=hp.Choice('regularization', 
                                   values=params['regularization']['choices']),
            reg_coefficient=hp.Float('reg_coefficient', 
                                    min_value=params['reg_coefficient']['min'],
                                    max_value=params['reg_coefficient']['max']),
            # Using default values for beta_1, beta_2, epsilon as per implementation
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        return model_runner.model

    def tune(self, loader, **kwargs):
        # Load actual normalization factors and feature indexes
        self.norm_factors = loader.load_normalization_factors()
        self.recurrent_features = loader.get_recurrent_features()  # Should return list of feature names/indices
        
        return super().run_tuning(self.build_model, loader, **kwargs)
