import tensorflow as tf
import numpy as np
from haversine import haversine_vector, Unit
from keras import Input, Model
from keras.layers import LSTM as LSTMKeras, Bidirectional, Dense, Dropout, Concatenate
from keras.optimizer_v2.adam import Adam as AdamKeras
from keras.regularizers import L1, L2
from typing import Dict, Tuple, List, Union
import pandas as pd
import os

from loading import Normalizer
from models.model_runner import ModelRunner


class BiLSTMFusionModelRunner(ModelRunner):
    """
    Enhanced Bi-LSTM Fusion Model with:
    - Bi-directional LSTM sequence processing
    - Vessel group static feature integration
    - Haversine distance as primary loss
    - MSE for comparative analysis
    """
    
    def __init__(self, 
                 number_of_rnn_layers: int,
                 rnn_layer_size: int,
                 number_of_final_dense_layers: int,
                 dense_layer_size: int,
                 input_ts_length: int,
                 input_num_recurrent_features: int,
                 output_num_features: int,
                 normalization_factors: Dict,
                 y_idxs: List[int],
                 columns: pd.DataFrame,
                 learning_rate: float,
                 recurrent_idxs: List[int],
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-7,
                 dropout: float = 0.0,
                 regularization: str = None,
                 reg_coefficient: float = 0.0):
        """
        Initialize the Bi-LSTM fusion model runner with updated loss handling.
        """
        self.rnn_layer = LSTMKeras
        self.direction = 'bidirectional'

        # Architecture parameters
        self.number_of_rnn_layers = number_of_rnn_layers
        self.rnn_layer_size = rnn_layer_size
        self.number_of_final_dense_layers = number_of_final_dense_layers
        self.dense_layer_size = dense_layer_size
        
        # Input/output configuration
        self.ts_length = input_ts_length
        self.input_num_recurrent_features = input_num_recurrent_features
        self.output_size = output_num_features
        
        # Static features handling
        self.num_vessel_groups = self._get_num_vessel_groups(columns)
        
        # Training configuration
        self.normalization_factors = normalization_factors
        self.y_idxs = y_idxs
        self.recurrent_idxs = recurrent_idxs
        self.columns = columns
        self.dropout_rate = dropout
        
        # Optimizer configuration
        self.optimizer = AdamKeras(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon
        )
        
        # Regularization setup
        self.regularizer = self._get_regularizer(regularization, reg_coefficient)
        
        # Build model
        self._init_model()

    def _get_num_vessel_groups(self, columns: pd.DataFrame) -> int:
        """Count vessel group categories from columns DataFrame"""
        vessel_cols = [col for col in columns['column'] 
                      if col.startswith('vessel_group_')]
        return len(vessel_cols)

    def _get_regularizer(self, reg_type: str, coefficient: float):
        """Configure regularization"""
        if reg_type == 'l1':
            return L1(coefficient)
        elif reg_type == 'l2':
            return L2(coefficient)
        return None

    def _haversine_loss(self, y_true, y_pred):
        """Custom loss function calculating Haversine distance"""
        # Denormalize predictions and true values
        lat_true = y_true[..., 0] * (self.normalization_factors['lat']['max'] - self.normalization_factors['lat']['min']) + self.normalization_factors['lat']['min']
        lon_true = y_true[..., 1] * (self.normalization_factors['lon']['max'] - self.normalization_factors['lon']['min']) + self.normalization_factors['lon']['min']
        lat_pred = y_pred[..., 0] * (self.normalization_factors['lat']['max'] - self.normalization_factors['lat']['min']) + self.normalization_factors['lat']['min']
        lon_pred = y_pred[..., 1] * (self.normalization_factors['lon']['max'] - self.normalization_factors['lon']['min']) + self.normalization_factors['lon']['min']
        
        # Convert to radians
        lat_true, lon_true, lat_pred, lon_pred = map(
            tf.deg2rad, [lat_true, lon_true, lat_pred, lon_pred]
        )
        
        # Haversine formula
        dlat = lat_pred - lat_true
        dlon = lon_pred - lon_true
        a = tf.sin(dlat/2)**2 + tf.cos(lat_true) * tf.cos(lat_pred) * tf.sin(dlon/2)**2
        c = 2 * tf.asin(tf.sqrt(a))
        r = 6371  # Earth radius in km
        return r * c

    def _init_model(self):
        """Build the Bi-LSTM fusion model architecture"""
        # 1. Recurrent Pathway (Bi-LSTM)
        recurrent_input = Input(
            shape=(self.ts_length, self.input_num_recurrent_features),
            name='recurrent_input'
        )
        x = self._build_recurrent_pathway(recurrent_input)
        
        # 2. Static Feature Pathway (Vessel Groups)
        static_input = Input(
            shape=(self.num_vessel_groups,),
            name='static_input'
        )
        static_features = self._build_static_pathway(static_input)
        
        # 3. Fusion and Output
        output = self._build_fusion_output(x, static_features)
        
        # 4. Compile Model
        self.model = Model(
            inputs=[recurrent_input, static_input],
            outputs=output,
            name='bi_lstm_fusion_model'
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss=self._haversine_loss,
            metrics=[self._haversine_loss, 'mse']
        )

    def _build_recurrent_pathway(self, input_layer):
        """Construct the Bi-LSTM sequence processing pathway"""
        x = input_layer
        for i in range(self.number_of_rnn_layers):
            return_sequences = i < (self.number_of_rnn_layers - 1)
            x = Bidirectional(
                self.rnn_layer(
                    self.rnn_layer_size,
                    return_sequences=return_sequences,
                    kernel_regularizer=self.regularizer,
                    recurrent_regularizer=self.regularizer,
                    name=f'bi_lstm_{i}'
                )
            )(x)
        return x

    def _build_static_pathway(self, input_layer):
        """Process static vessel group features"""
        x = Dense(
            self.dense_layer_size,
            activation='relu',
            kernel_regularizer=self.regularizer,
            name='static_dense'
        )(input_layer)
        x = Dropout(self.dropout_rate, name='static_dropout')(x)
        return x

    def _build_fusion_output(self, recurrent_features, static_features):
        """Combine pathways and build output layers"""
        if self.number_of_rnn_layers > 0:
            x = Concatenate(name='fusion_concat')([recurrent_features, static_features])
        else:
            x = static_features
        
        for i in range(self.number_of_final_dense_layers):
            x = Dense(
                self.dense_layer_size,
                activation='relu',
                kernel_regularizer=self.regularizer,
                name=f'final_dense_{i}'
            )(x)
            x = Dropout(self.dropout_rate, name=f'final_dropout_{i}')(x)
        
        return Dense(
            self.output_size,
            activation='linear',
            name='output'
        )(x)

   def predict(self, 
           valid_X_long_term: Tuple[np.ndarray, np.ndarray], 
           valid_Y_long_term: np.ndarray = None,
           args: object = None) -> Dict[str, Union[np.ndarray, float]]:
    """
    Make predictions with static feature validation
    
    Args:
        valid_X_long_term: Tuple of (recurrent_features, static_features)
        valid_Y_long_term: Optional ground truth
        args: Additional arguments
        
    Returns:
        Dictionary containing predictions and metrics
    """
    # --- Input Validation ---
    recurrent_X, static_X = valid_X_long_term
    
    # Validate static features
    if static_X.shape[1] != self.num_vessel_groups:
        raise ValueError(
            f"Static feature dimension mismatch. Expected {self.num_vessel_groups} "
            f"vessel groups, got {static_X.shape[1]}. Check DataLoader configuration."
        )
    
    # Validate recurrent features
    if recurrent_X.shape[1] != self.ts_length:
        raise ValueError(
            f"Sequence length mismatch. Expected {self.ts_length} timesteps, "
            f"got {recurrent_X.shape[1]}"
        )
    
    if recurrent_X.shape[2] != self.input_num_recurrent_features:
        raise ValueError(
            f"Feature dimension mismatch. Expected {self.input_num_recurrent_features} "
            f"features, got {recurrent_X.shape[2]}"
        )

    # --- Prediction Logic ---
    normalized_predictions = self.model.predict(
        [recurrent_X, static_X],
        batch_size=args.batch_size if args else 32
    )
    
    # --- Post-processing ---
    predictions = Normalizer().unnormalize(
        normalized_predictions, 
        self.normalization_factors
    )
    
    results = {
        'predictions': predictions,
        'normalized_predictions': normalized_predictions
    }
    
    # --- Metrics Calculation (if ground truth provided) ---
    if valid_Y_long_term is not None:
        true_values = Normalizer().unnormalize(
            valid_Y_long_term,
            self.normalization_factors
        )
        
        results.update({
            'haversine_distances': haversine_vector(true_values, predictions, Unit.KILOMETERS),
            'mean_haversine': float(np.mean(haversine_vector(true_values, predictions, Unit.KILOMETERS))),
            'mse': float(tf.keras.losses.mean_squared_error(valid_Y_long_term, normalized_predictions).numpy().mean())
        })
    
    return results

    def evaluate(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """Comprehensive evaluation with Haversine and MSE metrics"""
        haversine = tf.keras.metrics.Mean()
        mse = tf.keras.metrics.Mean()
        
        for batch_X, batch_Y in dataset:
            y_pred = self.model(batch_X, training=False)
            haversine.update_state(self._haversine_loss(batch_Y, y_pred))
            mse.update_state(tf.keras.losses.mean_squared_error(batch_Y, y_pred))
        
        return {
            'haversine_loss_km': haversine.result().numpy(),
            'mse': mse.result().numpy()
        }

    def save(self, filepath: str, **kwargs):
        """Save model to disk with extended metadata"""
        # Save the core model
        self.model.save(filepath, **kwargs)
        
        # Save additional runner state
        metadata = {
            'normalization_factors': self.normalization_factors,
            'y_idxs': self.y_idxs,
            'recurrent_idxs': self.recurrent_idxs,
            'num_vessel_groups': self.num_vessel_groups,
            'dropout_rate': self.dropout_rate,
            'regularization': self.regularizer.__class__.__name__ if self.regularizer else None,
            'reg_coefficient': self.regularizer.l2 if hasattr(self.regularizer, 'l2') else (
                self.regularizer.l1 if hasattr(self.regularizer, 'l1') else 0),
            'beta_1': self.optimizer.beta_1,
            'beta_2': self.optimizer.beta_2,
            'epsilon': self.optimizer.epsilon
        }
        np.savez(f'{filepath}_runner.npz', **metadata)

    @classmethod
    def load(cls, filepath: str, columns: pd.DataFrame, **kwargs):
        """Load saved model with runner state"""
        # Load core model
        model = tf.keras.models.load_model(filepath, **kwargs)
        
        # Load runner state
        metadata = np.load(f'{filepath}_runner.npz', allow_pickle=True)
        runner = cls(
            number_of_rnn_layers=len([l for l in model.layers if 'bi_lstm' in l.name]),
            rnn_layer_size=model.get_layer('bi_lstm_0').units,
            number_of_final_dense_layers=len([l for l in model.layers if 'final_dense' in l.name]),
            dense_layer_size=model.get_layer('final_dense_0').units,
            input_ts_length=model.input_shape[0][1],
            input_num_recurrent_features=model.input_shape[0][2],
            output_num_features=model.output_shape[1],
            normalization_factors=metadata['normalization_factors'].item(),
            y_idxs=metadata['y_idxs'],
            columns=columns,
            learning_rate=model.optimizer.learning_rate.numpy(),
            recurrent_idxs=metadata['recurrent_idxs'],
            dropout=metadata.get('dropout_rate', 0.0),
            regularization=metadata.get('regularization'),
            reg_coefficient=metadata.get('reg_coefficient', 0.0),
            beta_1=metadata.get('beta_1', 0.9),
            beta_2=metadata.get('beta_2', 0.999),
            epsilon=metadata.get('epsilon', 1e-7)
        )
        runner.model = model
        return runner
