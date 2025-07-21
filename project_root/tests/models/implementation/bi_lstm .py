import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from haversine import haversine_vector, Unit
from models.implementation.losses import TrajectoryLoss
from models.implementation.model_runner import TrajectoryModelRunner

class BiLSTMTrajectoryPredictor(TrajectoryModelRunner):
    """
    TensorFlow Bi-LSTM Model for Trajectory Prediction with standardized loss:
    - Uses TrajectoryLoss for consistent loss calculation
    - Implements TrajectoryModelRunner interface
    - Pure sequence-to-value architecture
    """

    def __init__(self,
                 input_ts_length: int,
                 input_num_features: int,
                 output_num_features: int,
                 lstm_hidden_dim: int,
                 lstm_num_layers: int,
                 dense_layer_size: int,
                 num_dense_layers: int,
                 dropout: float = 0.0,
                 regularization: str = None,
                 reg_coefficient: float = 0.0,
                 learning_rate: float = None,
                 beta_1: float = None,
                 beta_2: float = None,
                 epsilon: float = None,
                 norm_factors: Dict = None,
                 use_mse_loss: bool = False,
                 mse_weight: float = 0.1,
                 **kwargs):
        """
        Args:
            norm_factors: Dictionary containing min/max values for denormalization
                         Format: {'lat': {'min': ..., 'max': ...}, 
                                 'lon': {'min': ..., 'max': ...}}
            use_mse_loss: Whether to include MSE as auxiliary loss
            mse_weight: Weight for MSE component in combined loss
        """
        super().__init__()
        self._supports_static_features = False
        
        # Store normalization factors
        self.norm_factors = norm_factors or {
            'lat': {'min': -90, 'max': 90},
            'lon': {'min': -180, 'max': 180}
        }

        # Architecture parameters
        self.input_ts_length = input_ts_length
        self.input_num_features = input_num_features
        self.output_num_features = output_num_features
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.dense_layer_size = dense_layer_size
        self.num_dense_layers = num_dense_layers
        self.dropout_rate = dropout

        # Optimizer parameters
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # Regularization config
        self.regularizer = self._get_regularizer(regularization, reg_coefficient)

        # Loss configuration
        self.loss_fn = TrajectoryLoss(
            normalization_factors=self.norm_factors,
            use_mse=use_mse_loss,
            mse_weight=mse_weight
        )

        # Build model layers
        self._build_layers()
        self._compile_model()

    def _get_regularizer(self, reg_type: str, coefficient: float):
        """Configure regularization"""
        if reg_type == 'l1':
            return tf.keras.regularizers.L1(coefficient)
        elif reg_type == 'l2':
            return tf.keras.regularizers.L2(coefficient)
        return None

    def _compile_model(self):
        """Compile with current parameters"""
        optimizer_config = {
            'learning_rate': self.learning_rate,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon
        }
        optimizer_config = {k: v for k, v in optimizer_config.items() if v is not None}
        
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(**optimizer_config),
            loss=self.loss_fn,
            metrics=[self.loss_fn.haversine_loss, 'mse']
        )

    def _build_layers(self):
        """Build the BiLSTM model architecture"""
        inputs = tf.keras.layers.Input(shape=(self.input_ts_length, self.input_num_features))
        
        # BiLSTM layers
        x = inputs
        for i in range(self.lstm_num_layers):
            return_sequences = (i < self.lstm_num_layers - 1)  # Only return sequences for intermediate layers
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.lstm_hidden_dim,
                    return_sequences=return_sequences,
                    kernel_regularizer=self.regularizer,
                    recurrent_regularizer=self.regularizer
                )
            )(x)
            if self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        
        # Dense layers
        for _ in range(self.num_dense_layers):
            x = tf.keras.layers.Dense(
                self.dense_layer_size,
                activation='relu',
                kernel_regularizer=self.regularizer
            )(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(self.output_num_features)(x)
        
        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def call(self, inputs, training=None):
        """Forward pass"""
        return self._model(inputs, training=training)

    @property
    def model(self):
        """Get the underlying Keras model"""
        return self._model

    def predict(self, 
               x: np.ndarray,
               static_features: Optional[np.ndarray] = None,
               return_metrics: bool = False,
               batch_size: int = 32) -> Dict[str, Union[np.ndarray, float]]:
        """
        Enhanced prediction interface matching base class
        Args:
            x: Input array of shape (n_samples, seq_len, input_dim)
            static_features: Ignored (for interface compatibility)
            return_metrics: Whether to compute evaluation metrics
            batch_size: Batch size for prediction
        Returns:
            Dictionary containing predictions and optional metrics
        """
        dataset = tf.data.Dataset.from_tensor_slices(x)
        dataset = dataset.batch(batch_size)
        
        normalized_preds = []
        for batch in dataset:
            normalized_preds.append(self.model(batch, training=False))
        normalized_preds = tf.concat(normalized_preds, axis=0).numpy()
        
        # Use the loss function's denormalization method
        predictions = self.loss_fn._denormalize(normalized_preds)
        
        results = {'predictions': predictions}
        
        if return_metrics and hasattr(x, 'true_values'):
            true_values = self.loss_fn._denormalize(x.true_values)
            results.update({
                'haversine_distances': self.loss_fn.haversine_loss(
                    x.true_values, normalized_preds, reduce_mean=False
                ).numpy(),
                'mse': tf.keras.losses.mean_squared_error(
                    x.true_values, normalized_preds
                ).numpy().mean()
            })
        
        return results

    def evaluate(self,
                dataset: tf.data.Dataset,
                static_dataset: Optional[tf.data.Dataset] = None) -> Dict[str, float]:
        """
        Evaluation matching base class interface
        Args:
            dataset: Main input dataset
            static_dataset: Ignored (for interface compatibility)
        Returns:
            Dictionary of evaluation metrics
        """
        return self.model.evaluate(dataset, return_dict=True)

    def save(self, 
            filepath: str, 
            include_static_config: bool = True,
            **kwargs):
        """Save implementation matching base class"""
        self.model.save_weights(filepath)
        config = {
            'input_ts_length': self.input_ts_length,
            'input_num_features': self.input_num_features,
            'output_num_features': self.output_num_features,
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_num_layers': self.lstm_num_layers,
            'dense_layer_size': self.dense_layer_size,
            'num_dense_layers': self.num_dense_layers,
            'dropout_rate': self.dropout_rate,
            'norm_factors': self.norm_factors,
            'regularizer_config': {
                'type': self.regularizer.__class__.__name__ if self.regularizer else None,
                'coefficient': self.regularizer.l2 if hasattr(self.regularizer, 'l2') else (
                    self.regularizer.l1 if hasattr(self.regularizer, 'l1') else 0)
            },
            'optimizer_config': {
                'learning_rate': self.learning_rate,
                'beta_1': self.beta_1,
                'beta_2': self.beta_2,
                'epsilon': self.epsilon
            }
        }
        np.save(filepath + '_config.npy', config)

    @classmethod
    def load(cls, 
            filepath: str, 
            static_feature_shape: Optional[Tuple] = None,
            **kwargs):
        """Load implementation matching base class"""
        config = np.load(filepath + '_config.npy', allow_pickle=True).item()
        
        model = cls(
            input_ts_length=config['input_ts_length'],
            input_num_features=config['input_num_features'],
            output_num_features=config['output_num_features'],
            lstm_hidden_dim=config['lstm_hidden_dim'],
            lstm_num_layers=config['lstm_num_layers'],
            dense_layer_size=config['dense_layer_size'],
            num_dense_layers=config['num_dense_layers'],
            dropout=config['dropout_rate'],
            regularization=config['regularizer_config']['type'],
            reg_coefficient=config['regularizer_config']['coefficient'],
            learning_rate=config.get('optimizer_config', {}).get('learning_rate'),
            beta_1=config.get('optimizer_config', {}).get('beta_1'),
            beta_2=config.get('optimizer_config', {}).get('beta_2'),
            epsilon=config.get('optimizer_config', {}).get('epsilon'),
            norm_factors=config['norm_factors']
        )
        
        dummy_input = tf.zeros((1, config['input_ts_length'], config['input_num_features']))
        _ = model.model(dummy_input)
        model.model.load_weights(filepath)
        return model
