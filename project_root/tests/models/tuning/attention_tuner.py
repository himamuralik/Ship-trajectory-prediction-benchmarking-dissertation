import tensorflow as tf
import keras_tuner as kt
from .base_tuner import BaseTuner
from models.implementations.bi_lstm_attention import BiLSTMAttentionTrajectoryPredictor

class AttentionTuner(BaseTuner):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name, 'bi_lstm_attention')
        
    def build_model(self, hp: kt.HyperParameters) -> tf.keras.Model:
        params = self.load_search_space()
        
        model = BiLSTMAttentionTrajectoryPredictor(
            input_ts_length=hp.Int('input_ts_length', **params['input_ts_length']),
            input_num_features=hp.Int('input_num_features', **params['input_num_features']),
            output_num_features=2,
            lstm_hidden_dim=hp.Int('lstm_hidden_dim', **params['lstm_hidden_dim']),
            lstm_num_layers=hp.Int('lstm_num_layers', **params['lstm_num_layers']),
            dense_layer_size=hp.Int('dense_layer_size', **params['dense_layer_size']),
            num_dense_layers=hp.Int('num_dense_layers', **params['num_dense_layers']),
            dropout=hp.Float('dropout', **params['dropout']),
            regularization=hp.Choice('regularization', params['regularization']['choices']),
            reg_coefficient=hp.Float('reg_coefficient', **params['reg_coefficient']),
            learning_rate=hp.Float('learning_rate', **params['learning_rate']),
            beta_1=hp.Float('beta_1', **params['beta_1']),
            beta_2=hp.Float('beta_2', **params['beta_2']),
            epsilon=hp.Float('epsilon', **params['epsilon'])
        )
        
        return model
