from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Dict, Union, Tuple, Optional
import numpy as np

class TrajectoryModelRunner(ABC):
    """Enhanced base class supporting both regular and fusion models"""
    
    @abstractmethod
    def __init__(self):
        self._model = None
        self._supports_static_features = False  # To be set by subclasses
    
    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @property
    def supports_static_features(self) -> bool:
        """Whether this model variant uses static features"""
        return self._supports_static_features

    def fit(self, 
            dataset: tf.data.Dataset,
            epochs: int,
            validation_data: Optional[tf.data.Dataset] = None,
            callbacks: list = None,
            **kwargs) -> tf.keras.callbacks.History:
        """Handles both regular and static-feature datasets"""
        return self.model.fit(
            dataset,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks or [],
            **kwargs
        )

    @abstractmethod
    def predict(self, 
               x: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
               static_features: Optional[np.ndarray] = None,
               return_metrics: bool = False,
               batch_size: int = 32) -> Dict[str, Union[np.ndarray, float]]:
        """
        Enhanced prediction interface supporting static features
        Args:
            x: Main input features (array or tuple for fusion models)
            static_features: Optional static features array
            return_metrics: Whether to compute evaluation metrics
            batch_size: Prediction batch size
        """
        pass

    @abstractmethod
    def evaluate(self,
                dataset: tf.data.Dataset,
                static_dataset: Optional[tf.data.Dataset] = None) -> Dict[str, float]:
        """Evaluation supporting optional static features"""
        pass

    @abstractmethod
    def save(self, filepath: str, include_static_config: bool = True, **kwargs):
        """Save with explicit static feature handling"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, 
            filepath: str, 
            static_feature_shape: Optional[Tuple] = None,
            **kwargs):
        """Load with static feature configuration"""
        pass

    def compile(self, optimizer=None, loss=None, metrics=None):
        """Optional compilation with static feature awareness"""
        self.model.compile(
            optimizer=optimizer or self.model.optimizer,
            loss=loss or self.model.loss,
            metrics=metrics or self.model.metrics
        )
