import tensorflow as tf
from haversine import haversine_vector, Unit
from typing import Dict, List, Tuple, Any, Union
import numpy as np
import os
from models.implementation.losses import TrajectoryLoss, AttentionAwareLoss

class BiLSTMAttentionTrajectoryPredictor(tf.keras.Model):
    """
    Enhanced Bi-LSTM with Attention Model for Trajectory Prediction:
    - Uses standardized TrajectoryLoss with Haversine distance
    - Supports attention-weighted loss when needed
    - Includes attention mechanism for temporal feature weighting
    - Returns attention weights during prediction
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
                 norm_factors: Dict[str, Dict[str, float]] = None,
                 use_mse_loss: bool = False,
                 mse_weight: float = 0.1,
                 use_attention_loss: bool = False,
                 attention_weight: float = 0.5,
                 **kwargs):
        """
        Args:
            norm_factors: Dictionary containing min/max values for denormalization
                         Format: {'lat': {'min': ..., 'max': ...}, 
                                 'lon': {'min': ..., 'max': ...}}
            use_mse_loss: Whether to include MSE as auxiliary loss
            mse_weight: Weight for MSE component in combined loss
            use_attention_loss: Whether to use attention-aware loss
            attention_weight: Weight for attention component in loss
        """
        super().__init__(**kwargs)
        
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
        if use_attention_loss:
            self.loss_fn = AttentionAwareLoss(
                normalization_factors=self.norm_factors,
                use_mse=use_mse_loss,
                mse_weight=mse_weight,
                attention_weight=attention_weight
            )
        else:
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
        """Configure training with Adam optimizer"""
        optimizer_config = {
            'learning_rate': self.learning_rate,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
            'clipnorm': 1.0  # Gradient clipping
        }
        optimizer_config = {k: v for k, v in optimizer_config.items() if v is not None}
        
        self.compile(
            optimizer=tf.keras.optimizers.Adam(**optimizer_config),
            loss=self.loss_fn,
            metrics=[self.loss_fn.haversine_loss, 'mse']
        )

    def _build_layers(self):
        """Construct the neural network architecture with attention"""
        # Bidirectional LSTM layers (all return sequences for attention)
        self.bilstm_layers = []
        for i in range(self.lstm_num_layers):
            self.bilstm_layers.append(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        units=self.lstm_hidden_dim // 2,
                        return_sequences=True,  # Must be True for attention
                        kernel_regularizer=self.regularizer,
                        recurrent_regularizer=self.regularizer,
                        name=f'lstm_{i}'
                    ),
                    name=f'bilstm_{i}'
                )
            )

        # Attention mechanism
        self.attention = tf.keras.layers.Dense(
            units=1,
            activation='tanh',
            kernel_regularizer=self.regularizer,
            name='attention'
        )
        
        # Layer normalization for attention output
        self.layer_norm = tf.keras.layers.LayerNormalization(name='layer_norm')

        # Dense network
        self.dense_layers = []
        for i in range(self.num_dense_layers):
            self.dense_layers.extend([
                tf.keras.layers.Dense(
                    units=self.dense_layer_size,
                    activation='relu',
                    kernel_regularizer=self.regularizer,
                    name=f'dense_{i}'
                ),
                tf.keras.layers.Dropout(
                    rate=self.dropout_rate,
                    name=f'dropout_{i}'
                )
            ])

        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            units=self.output_num_features,
            activation='linear',
            name='output'
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
      """
    Forward pass returning both predictions and attention weights.
    
    Args:
        inputs: Input tensor of shape [batch, timesteps, features]
        training: Whether in training mode
        
    Returns:
        tuple: (predictions, attention_weights) 
               - predictions: Shape [batch, output_features]
               - attention_weights: Shape [batch, timesteps, 1]
    """
        # Input validation
        if len(inputs.shape) != 3:
            raise ValueError(
                f"Input tensor must be 3D [batch, timesteps, features], got {inputs.shape}"
            )
        
        x = inputs
        
        # Bi-LSTM processing
        for layer in self.bilstm_layers:
            x = layer(x, training=training)
        
        # Attention mechanism
        attention_scores = self.attention(x)  # [batch, seq_len, 1]
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # [batch, seq_len, 1]
        
        # Debug checks (optional)
        tf.debugging.assert_near(
            tf.reduce_sum(attention_weights, axis=1),
            tf.ones_like(attention_weights[:, 0, :]),
            message="Attention weights don't sum to 1"
        )
        
        # Context vector calculation
        context_vector = tf.reduce_sum(attention_weights * x, axis=1)  # [batch, features]
        context_vector = self.layer_norm(context_vector)
        
        # Dense processing
        x = context_vector
        for layer in self.dense_layers:
            x = layer(x, training=training)
        
        return self.output_layer(x), attention_weights

    def predict_only(self, inputs: tf.Tensor) -> tf.Tensor:
        """Convenience method for when only predictions are needed.
        Args:
            inputs: Input tensor of shape [batch, timesteps, features]
        Returns:
            predictions: Shape [batch, output_features] 
        """
        return self(inputs, training=False)[0]  # Returns only predictions

    def predict(self, 
               x: np.ndarray,
               return_metrics: bool = False,
               batch_size: int = 32,
               return_attention: bool = False) -> Dict[str, Any]:
        """Enhanced prediction with attention weights"""
        dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        
        if return_attention:
            preds, attentions = [], []
            for batch in dataset:
                p, a = self(batch, training=False)
                preds.append(p)
                attentions.append(a)
            preds = tf.concat(preds, axis=0)
            attentions = tf.concat(attentions, axis=0)
        else:
            preds = tf.concat([self(batch, training=False)[0] for batch in dataset], axis=0)
        
        results = {
            'predictions': self._denormalize(preds.numpy())
        }
        
        if return_attention:
            results['attention_weights'] = attentions.numpy()
        
        if return_metrics and hasattr(x, 'true_values'):
            true_denorm = self._denormalize(x.true_values)
            pred_denorm = results['predictions']
            results.update({
                'haversine_km': self.loss_fn.haversine_loss(
                    tf.constant(x.true_values),
                    preds,
                    reduce_mean=False
                ).numpy(),
                'mse': tf.keras.losses.mean_squared_error(x.true_values, preds).numpy()
            })
        
        return results

    def _denormalize(self, array: np.ndarray) -> np.ndarray:
        """Convert normalized values back to original coordinates"""
        denorm = array.copy()
        if self.output_num_features >= 1:  # Latitude
            denorm[..., 0] = denorm[..., 0] * (self.norm_factors['lat']['max'] - self.norm_factors['lat']['min']) + self.norm_factors['lat']['min']
        if self.output_num_features >= 2:  # Longitude
            denorm[..., 1] = denorm[..., 1] * (self.norm_factors['lon']['max'] - self.norm_factors['lon']['min']) + self.norm_factors['lon']['min']
        return denorm

    def save_model(self, filepath: str):
        """Save weights and configuration"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save weights
        self.save_weights(filepath)
        
        # Save configuration
        config = {
            'architecture': {
                'input_ts_length': self.input_ts_length,
                'input_num_features': self.input_num_features,
                'output_num_features': self.output_num_features,
                'lstm_hidden_dim': self.lstm_hidden_dim,
                'lstm_num_layers': self.lstm_num_layers,
                'dense_layer_size': self.dense_layer_size,
                'num_dense_layers': self.num_dense_layers,
                'dropout_rate': self.dropout_rate
            },
            'regularization': {
                'type': self.regularizer.__class__.__name__ if self.regularizer else None,
                'coefficient': getattr(self.regularizer, 'l2', 
                                     getattr(self.regularizer, 'l1', 0))
            },
            'optimizer': {
                'learning_rate': self.learning_rate,
                'beta_1': self.beta_1,
                'beta_2': self.beta_2,
                'epsilon': self.epsilon
            },
            'loss_config': {
                'use_mse': self.loss_fn.use_mse,
                'mse_weight': self.loss_fn.mse_weight,
                'use_attention_loss': isinstance(self.loss_fn, AttentionAwareLoss),
                'attention_weight': getattr(self.loss_fn, 'attention_weight', 0.0)
            },
            'norm_factors': self.norm_factors
        }
        np.save(filepath + '_config.npy', config)

    @classmethod
    def load_model(cls, filepath: str) -> 'BiLSTMAttentionTrajectoryPredictor':
        """Load saved model from disk"""
        config = np.load(filepath + '_config.npy', allow_pickle=True).item()
        
        # Recreate model
        model = cls(
            **config['architecture'],
            regularization=config['regularization']['type'],
            reg_coefficient=config['regularization']['coefficient'],
            **config['optimizer'],
            norm_factors=config['norm_factors'],
            use_mse_loss=config['loss_config']['use_mse'],
            mse_weight=config['loss_config']['mse_weight'],
            use_attention_loss=config['loss_config']['use_attention_loss'],
            attention_weight=config['loss_config']['attention_weight']
        )
        
        # Build model graph
        dummy_input = tf.zeros([1, config['architecture']['input_ts_length'], 
                              config['architecture']['input_num_features']])
        _ = model(dummy_input)
        
        # Load weights
        model.load_weights(filepath)
        return model

    def evaluate(self, 
                dataset: tf.data.Dataset,
                return_dict: bool = True) -> Dict[str, float]:
        """Comprehensive evaluation using standardized loss"""
        metrics = {
            'haversine_loss': tf.keras.metrics.Mean(),
            'mse': tf.keras.metrics.Mean()
        }
        
        for x, y_true in dataset:
            y_pred = self(x, training=False)[0]  # Get only predictions
            metrics['haversine_loss'].update_state(
                self.loss_fn.haversine_loss(y_true, y_pred)
            )
            metrics['mse'].update_state(
                tf.keras.losses.mean_squared_error(y_true, y_pred)
            )
        
        return {k: v.result().numpy() for k, v in metrics.items()}

    def plot_attention(self, sample_input: np.ndarray):
        """Visualize attention weights for a sample input"""
        import matplotlib.pyplot as plt
        
        results = self.predict(np.expand_dims(sample_input, 0), return_attention=True)
        weights = results['attention_weights'].squeeze()
        
        plt.figure(figsize=(10, 4))
        plt.plot(weights, 'o-')
        plt.xlabel('Timestep')
        plt.ylabel('Attention Weight')
        plt.title('Temporal Attention Weights')
        plt.show()
