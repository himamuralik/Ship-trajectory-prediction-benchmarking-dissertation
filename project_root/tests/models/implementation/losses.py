import math
import tensorflow as tf
from typing import Dict, Optional, Tuple
from config import config

class TrajectoryLoss:
    """
    Base loss class for trajectory prediction models with:
    - Primary Haversine distance loss
    - Optional MSE auxiliary loss
    - Support for normalized/denormalized coordinates
    """
    
    def __init__(self, 
                normalization_factors: Optional[Dict] = None,
                use_mse: bool = False,
                mse_weight: float = 0.1):
        """
        Args:
            normalization_factors: Dictionary containing min/max values for normalization
                Format: {'lat': {'min': ..., 'max': ...}, 
                        'lon': {'min': ..., 'max': ...}}
            use_mse: Whether to include MSE as auxiliary loss
            mse_weight: Weight for MSE component in combined loss
        """
        self.norm_factors = normalization_factors or self._get_default_norm_factors()
        self.use_mse = use_mse
        self.mse_weight = mse_weight
        self._requires_static_features = False
        
        # Constants
        self.EARTH_RADIUS = 6371.0  # km
        self.pi = tf.constant(math.pi, dtype=tf.float32)
        
    def _get_default_norm_factors(self) -> Dict:
        """Fallback normalization factors if none provided"""
        return {
            'lat': {'min': config.dataset_config.lat_1,
                    'max': config.dataset_config.lat_2},
            'lon': {'min': config.dataset_config.lon_1,
                    'max': config.dataset_config.lon_2}
        }

    @property
    def requires_static_features(self) -> bool:
        """Whether this loss variant requires static features"""
        return self._requires_static_features

    @tf.function
    def __call__(self, 
                y_true: tf.Tensor, 
                y_pred: tf.Tensor,
                **kwargs) -> tf.Tensor:
        """
        Calculate combined loss with optional MSE component.
        
        Args:
            y_true: Ground truth coordinates (normalized)
            y_pred: Predicted coordinates (normalized)
            kwargs: Additional arguments for specialized losses
            
        Returns:
            Combined loss value
        """
        haversine_loss = self.haversine_loss(y_true, y_pred)
        
        if self.use_mse:
            mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
            return haversine_loss + self.mse_weight * mse_loss
        return haversine_loss

    @tf.function
    def haversine_loss(self, 
                      y_true: tf.Tensor, 
                      y_pred: tf.Tensor,
                      reduce_mean: bool = True) -> tf.Tensor:
        """
        Calculate Haversine distance between predicted and true coordinates.
        
        Args:
            y_true: Ground truth coordinates [..., {lat, lon}]
            y_pred: Predicted coordinates [..., {lat, lon}]
            reduce_mean: Whether to reduce to mean loss or return per-sample
            
        Returns:
            Haversine distance in kilometers
        """
        # Denormalize coordinates if they're normalized
        if tf.reduce_max(y_true) <= 1.0 and tf.reduce_min(y_true) >= 0.0:
            y_true = self._denormalize(y_true)
            y_pred = self._denormalize(y_pred)
        
        # Convert to radians
        y_true_rad = y_true * (self.pi / 180.0)
        y_pred_rad = y_pred * (self.pi / 180.0)
        
        # Extract lat/lon components
        lat_true = y_true_rad[..., 0]
        lon_true = y_true_rad[..., 1]
        lat_pred = y_pred_rad[..., 0]
        lon_pred = y_pred_rad[..., 1]
        
        # Haversine formula
        dlat = lat_pred - lat_true
        dlon = lon_pred - lon_true
        
        a = (tf.math.sin(dlat/2)**2 + 
             tf.math.cos(lat_true) * tf.math.cos(lat_pred) * 
             tf.math.sin(dlon/2)**2
        
        # Clip to avoid numerical instability
        a = tf.clip_by_value(a, 0.0, 1.0)
        
        distance = 2 * self.EARTH_RADIUS * tf.math.asin(tf.math.sqrt(a))
        
        if reduce_mean:
            return tf.reduce_mean(distance)
        return distance
def _denormalize(self, coords: tf.Tensor) -> tf.Tensor:
    """Convert normalized coordinates back to original scale.
    
    Args:
        coords: Normalized coordinates tensor of shape [..., 2] where:
            coords[..., 0] = normalized latitude
            coords[..., 1] = normalized longitude
            
    Returns:
        Denormalized coordinates tensor in original scale [..., 2]
    """

        lat = coords[..., 0] * (self.norm_factors['lat']['max'] - 
                               self.norm_factors['lat']['min']) + \
              self.norm_factors['lat']['min']
        
        lon = coords[..., 1] * (self.norm_factors['lon']['max'] - 
                               self.norm_factors['lon']['min']) + \
              self.norm_factors['lon']['min']
        
        return tf.stack([lat, lon], axis=-1)

    def get_config(self) -> Dict:
        """Get configuration for saving/loading"""
        return {
            'normalization_factors': self.norm_factors,
            'use_mse': self.use_mse,
            'mse_weight': self.mse_weight,
            'requires_static_features': self._requires_static_features
        }

    @classmethod
    def from_config(cls, config: Dict) -> 'TrajectoryLoss':
        """Create loss from saved configuration"""
        return cls(**config)


class BiLSTMLoss(TrajectoryLoss):
    """Alias for backward compatibility"""
    pass


class AttentionAwareLoss(TrajectoryLoss):
    """
    Enhanced loss for attention-based models with:
    - Temporal weighting capability
    - Focus on critical trajectory segments
    """
    
    def __init__(self, 
                normalization_factors: Optional[Dict] = None,
                use_mse: bool = False,
                mse_weight: float = 0.1,
                attention_weight: float = 0.5):
        """
        Args:
            attention_weight: Weight for attention-aware loss component
        """
        super().__init__(normalization_factors, use_mse, mse_weight)
        self.attention_weight = attention_weight

    @tf.function
    def __call__(self, 
                y_true: tf.Tensor, 
                y_pred: tf.Tensor,
                attention_weights: Optional[tf.Tensor] = None,
                **kwargs) -> tf.Tensor:
        """
        Calculate loss with optional attention weighting.
        """
        base_loss = super().__call__(y_true, y_pred)
        
        if attention_weights is not None:
            # Apply temporal attention to the loss
            temporal_loss = self._apply_attention(y_true, y_pred, attention_weights)
            return (1 - self.attention_weight) * base_loss + \
                   self.attention_weight * temporal_loss
        return base_loss

    def _apply_attention(self, 
                        y_true: tf.Tensor,
                        y_pred: tf.Tensor,
                        attention_weights: tf.Tensor) -> tf.Tensor:
        """Calculate attention-weighted loss"""
        # Calculate per-timestep haversine distance
        per_step_loss = self.haversine_loss(y_true, y_pred, reduce_mean=False)
        
        # Apply attention weights
        return tf.reduce_sum(per_step_loss * attention_weights) / \
               tf.reduce_sum(attention_weights)

    def get_config(self) -> Dict:
        config = super().get_config()
        config['attention_weight'] = self.attention_weight
        return config


class FusionModelLoss(TrajectoryLoss):
    """
    Specialized loss for fusion models with:
    - Static feature integration
    - Vessel-type aware regularization
    """
    
    def __init__(self, 
                normalization_factors: Optional[Dict] = None,
                use_mse: bool = False,
                mse_weight: float = 0.1,
                static_weight: float = 0.3):
        """
        Args:
            static_weight: Influence of static features on loss (0-1)
        """
        super().__init__(normalization_factors, use_mse, mse_weight)
        self.static_weight = static_weight
        self._requires_static_features = True

    @tf.function
    def __call__(self, 
                y_true: tf.Tensor, 
                y_pred: tf.Tensor,
                static_features: Optional[tf.Tensor] = None,
                **kwargs) -> tf.Tensor:
        """
        Calculate loss with static feature consideration.
        
        Args:
            static_features: Tensor of shape [batch_size, static_feature_dim]
        """
        if static_features is None and self.requires_static_features:
            raise ValueError("This loss requires static features")
        
        base_loss = super().__call__(y_true, y_pred)
        
        if static_features is not None:
            static_reg = self._static_feature_regularization(y_pred, static_features)
            return (1 - self.static_weight) * base_loss + \
                   self.static_weight * static_reg
        return base_loss

    def _static_feature_regularization(self,
                                     y_pred: tf.Tensor,
                                     static_features: tf.Tensor) -> tf.Tensor:
        """
        Vessel-type aware regularization:
        - Groups predictions by vessel type
        - Penalizes high variance within groups
        """
        # Assuming first static feature is vessel type
        vessel_types = static_features[:, 0]
        unique_types = tf.unique(vessel_types)[0]
        
        variances = []
        for t in unique_types:
            mask = tf.equal(vessel_types, t)
            group_preds = tf.boolean_mask(y_pred, mask)
            variances.append(tf.math.reduce_variance(group_preds, axis=0))
            
        return tf.reduce_mean(variances)

    def get_config(self) -> Dict:
        config = super().get_config()
        config['static_weight'] = self.static_weight
        return config
