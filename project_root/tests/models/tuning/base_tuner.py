# models/tuning/base_tuner.py
import os
import keras_tuner as kt
import yaml
from typing import Callable, Dict, Any
import numpy as np
from .sampling import AisSampler
from .. import config

class BaseTuner:
    def __init__(self, dataset_name: str, model_name: str):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.sampler = AisSampler()
        self.norm_factors = None

    def load_search_space(self) -> Dict[str, Any]:
        """Load parameter ranges from YAML file in search_space/"""
        search_space_path = os.path.join(
            'models', 
            'model_config', 
            'search_space', 
            f'{self.model_name}_params.yaml'  # Matches your filenames
        )
        # Verify file exists
        if not os.path.exists(search_space_path):
            raise FileNotFoundError(
                f"Search space config not found at: {search_space_path}\n"
                f"Expected files: bilstm_params.yaml, attention_params.yaml, fusion_params.yaml"
            )    
        with open(search_space_path) as f:
            return yaml.safe_load(f)
        
    def _prepare_data(self, loader):
        data = self.sampler.get_tuning_data(self.dataset_name, loader)
        self.norm_factors = self.sampler.get_normalization_factors()
        split_idx = int(len(data[0]) * (1 - config.val_ratio))
        return (data[0][:split_idx], data[1][:split_idx]), (data[0][split_idx:], data[1][split_idx:])

    def run_tuning(self, build_model_fn: Callable, loader, **tuner_kwargs):
        train_data, val_data = self._prepare_data(loader)
        
        tuner = kt.Hyperband(
            build_model_fn,
            objective='val_loss',
            max_epochs=50,
            directory=os.path.join(config.model_dir, 'tuning_results'),
            project_name=f"{self.model_name}_{self.dataset_name}",
            **tuner_kwargs
        )
        
        tuner.search(
            *train_data,
            validation_data=val_data,
            callbacks=[kt.keras.callbacks.TensorBoard(config.tuning_log_dir)],
            **config.tuning_params
        )
        
        return tuner.get_best_hyperparameters()[0].values
