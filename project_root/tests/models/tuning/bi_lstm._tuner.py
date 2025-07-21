import tensorflow as tf
import keras_tuner as kt
from typing import Dict, Any
from .base_tuner import BaseTuner
from models.implementations.bi_lstm import BiLSTMTrajectoryPredictor

class BiLSTMTuner(BaseTuner):
    def __init__(self, dataset_name: str):
        """
        Hyperparameter tuner for BiLSTM trajectory prediction model
        
        Args:
            dataset_name: Name of dataset to use for tuning
        """
        super().__init__(dataset_name, 'bi_lstm')
        
    def _convert_param(self, 
                      hp: kt.HyperParameters, 
                      param_name: str, 
                      param_config: Dict[str, Any]) -> Any:
        """
        Convert parameter config to appropriate hyperparameter type
        
        Args:
            hp: HyperParameters instance
            param_name: Name of parameter
            param_config: Parameter configuration from YAML
            
        Returns:
            Configured hyperparameter
        """
        if 'values' in param_config:
            return hp.Choice(param_name, param_config['values'])
        elif 'choices' in param_config:
            return hp.Choice(param_name, param_config['choices'])
        elif 'min' in param_config and 'max' in param_config:
            if isinstance(param_config['min'], int):
                return hp.Int(param_name, 
                            min_value=param_config['min'],
                            max_value=param_config['max'])
            return hp.Float(param_name,
                          min_value=param_config['min'],
                          max_value=param_config['max'])
        raise ValueError(f"Invalid parameter config for {param_name}: {param_config}")

    def build_model(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """
        Build model with hyperparameters from search space
        
        Args:
            hp: HyperParameters instance
            
        Returns:
            Configured BiLSTM model
        """
        params = self.load_search_space()
        
        # Get normalization factors from data sampler
        try:
            norm_factors = self.sampler.get_normalization_factors(self.dataset_name)
        except AttributeError:
            norm_factors = None  # Fallback to default in model
        
        model = BiLSTMTrajectoryPredictor(
            input_ts_length=self._convert_param(hp, 'input_ts_length', params['input_ts_length']),
            input_num_features=self._convert_param(hp, 'input_num_features', params['input_num_features']),
            output_num_features=2,  # Fixed for lat/lon
            lstm_hidden_dim=self._convert_param(hp, 'lstm_hidden_dim', params['lstm_hidden_dim']),
            lstm_num_layers=self._convert_param(hp, 'lstm_num_layers', params['lstm_num_layers']),
            dense_layer_size=self._convert_param(hp, 'dense_layer_size', params['dense_layer_size']),
            num_dense_layers=self._convert_param(hp, 'num_dense_layers', params['num_dense_layers']),
            dropout=self._convert_param(hp, 'dropout', params['dropout']),
            regularization=self._convert_param(hp, 'regularization', params['regularization']),
            reg_coefficient=self._convert_param(hp, 'reg_coefficient', params['reg_coefficient']),
            learning_rate=self._convert_param(hp, 'learning_rate', params['learning_rate']),
            beta_1=self._convert_param(hp, 'beta_1', params['beta_1']),
            beta_2=self._convert_param(hp, 'beta_2', params['beta_2']),
            epsilon=self._convert_param(hp, 'epsilon', params['epsilon']),
            norm_factors=norm_factors
        )
        
        return model

    def tune(self, loader, **kwargs) -> kt.HyperParameters:
        """
        Run hyperparameter tuning
        
        Args:
            loader: Data loader instance
            **kwargs: Additional arguments for tuning
            
        Returns:
            Best hyperparameters found
        """
        best_hps = super().run_tuning(self.build_model, loader, **kwargs)
        
        # Validate model can be properly instantiated
        test_model = self.build_model(kt.HyperParameters.from_config(best_hps.get_config()))
        test_input = tf.zeros((1, 
                             best_hps.get('input_ts_length'), 
                             best_hps.get('input_num_features')))
        _ = test_model(test_input)  # Build graph
        
        return best_hps
