import tensorflow as tf
import keras_tuner as kt
from .base_tuner import BaseTuner
from models.implementations.bi_lstm_attention import BiLSTMAttentionTrajectoryPredictor

class AttentionTuner(BaseTuner):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name, 'bi_lstm_attention')
        
    def build_model(self, hp: kt.HyperParameters) -> tf.keras.Model:
        params = self.load_search_space()
        
        # Attention-specific parameters
        use_attention_loss = hp.Boolean('attention_loss', default=False)
        attention_weight = hp.Float('attention_weight', 
                                  min_value=0.1, 
                                  max_value=0.9,
                                  step=0.1,
                                  default=0.5) if use_attention_loss else 0.5
        
        use_mse_loss = hp.Boolean('mse_loss', default=False)
        mse_weight = hp.Float('mse_weight',
                            min_value=0.1,
                            max_value=0.5,
                            step=0.1,
                            default=0.1) if use_mse_loss else 0.0

        model = BiLSTMAttentionTrajectoryPredictor(
            input_ts_length=self.get_sequence_length(),
            input_num_features=self.get_num_features(),
            output_num_features=2,
            lstm_hidden_dim=hp.Int('lstm_hidden_dim', **params['lstm_hidden_dim']),
            lstm_num_layers=hp.Int('lstm_num_layers', **params['lstm_num_layers']),
            dense_layer_size=hp.Int('dense_layer_size', **params['dense_layer_size']),
            num_dense_layers=hp.Int('num_dense_layers', **params['num_dense_layers']),
            dropout=hp.Float('dropout', **params['dropout']),
            regularization=hp.Choice('regularization', params['regularization']['choices']),
            reg_coefficient=hp.Float('reg_coefficient', **params['reg_coefficient']),
            learning_rate=hp.Choice('learning_rate', params['learning_rate']['values']),
            beta_1=hp.Choice('beta_1', params['beta_1']['values']),
            beta_2=hp.Choice('beta_2', params['beta_2']['values']),
            epsilon=hp.Choice('epsilon', params['epsilon']['values']),
            use_mse_loss=use_mse_loss,
            mse_weight=mse_weight,
            use_attention_loss=use_attention_loss,
            attention_weight=attention_weight,
            norm_factors=self.get_norm_factors()
        )
        
        return model

    def get_sequence_length(self) -> int:
        """Get optimal sequence length from dataset config"""
        return self.dataset_config.sequence_length

    def get_num_features(self) -> int:
        """Get number of input features from dataset"""
        return len(self.dataset_config.feature_columns)

    def get_norm_factors(self) -> Dict:
        """Get normalization factors from dataset"""
        return {
            'lat': {
                'min': self.dataset_config.lat_1,
                'max': self.dataset_config.lat_2
            },
            'lon': {
                'min': self.dataset_config.lon_1,
                'max': self.dataset_config.lon_2
            }
        }
