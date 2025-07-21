import math
import numpy as np
import pandas as pd
from config import config
from typing import Tuple

class AisSampler:
    @staticmethod
    def calculate_sample_size(dataset_name: str) -> int:
        """Enhanced sampling with temporal considerations"""
        pop_size = config.dataset_stats[dataset_name]['population_size']
        time_dim = config.dataset_stats[dataset_name]['time_dimension']
        
        min_sample = math.ceil((1.96**2 * 0.5 * 0.5) / (0.05**2))
        
        if time_dim == 'yearly':
            time_factor = 5
        else:
            time_factor = 1
            
        return min(
            max(min_sample * time_factor, 10_000),
            int(pop_size * 0.2),
            100_000
        )

    @staticmethod
    def get_tuning_data(dataset_name: str, loader) -> Tuple[np.ndarray, np.ndarray]:
        """Get temporally-stratified sample"""
        sample_size = AisSampler.calculate_sample_size(dataset_name)
        full_data = loader.load(dataset_name)
        
        if hasattr(full_data, 'timestamps'):
            time_bins = pd.cut(full_data.timestamps, bins=12)
            sampled = full_data.groupby(time_bins).apply(
                lambda x: x.sample(n=int(sample_size/12), replace=False)
        else:
            sampled = full_data.sample(n=sample_size)
            
        return sampled.values
