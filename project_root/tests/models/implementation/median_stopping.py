import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import numpy as np
import os
from typing import Optional, Literal
import tensorflow as tf

class EnhancedMedianStopper(tf.keras.callbacks.Callback):
    """
    Advanced Median Stopper with fusion model support featuring:
    - Model-type aware stopping thresholds
    - Dual convergence monitoring (main loss + static feature impact)
    - Dynamic patience adjustment
    - Enhanced MLflow tracking
    
    Args:
        experiment_id: MLflow experiment ID
        model_type: Either 'standard' or 'fusion'
        iterations: Check stopping every N epochs
        cutoff_file: Alternate path for cutoff storage
        fusion_config: Optional dictionary with fusion-specific params:
            - static_feature_weight: Importance of static features (0-1)
            - min_epochs: Minimum training epochs before checking
            - patience_factor: Multiplier for standard patience
    """

    def __init__(self, 
                 experiment_id: str,
                 model_type: Literal['standard', 'fusion'] = 'standard',
                 iterations: int = 30,
                 cutoff_file: Optional[str] = None,
                 fusion_config: Optional[dict] = None):
        
        super().__init__()
        self.cutoffs_path = cutoff_file or os.path.join(
            config.box_and_year_dir, 
            '.median_losses.csv'
        )
        self.run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        self.experiment_id = experiment_id
        self.model_type = model_type
        self.early_stopping_occurred = False
        self.stopped_epoch = -1
        self.best_loss = np.Inf
        self.iterations = iterations
        
        # Fusion model specific configuration
        self.fusion_config = fusion_config or {}
        self._init_model_type_settings()
        
        # Tracking buffers
        self.static_feature_impacts = []
        self.convergence_history = []

    def _init_model_type_settings(self):
        """Initialize model-specific parameters"""
        if self.model_type == 'fusion':
            self.min_epochs = self.fusion_config.get('min_epochs', 20)
            self.patience_factor = self.fusion_config.get('patience_factor', 1.5)
            self.static_feature_weight = self.fusion_config.get('static_feature_weight', 0.3)
        else:
            self.min_epochs = 10
            self.patience_factor = 1.0
            self.static_feature_weight = 0.0

    def on_train_begin(self, logs=None):
        """Initialize tracking and load historical cutoffs"""
        try:
            self._load_cutoffs()
            if self.model_type == 'fusion':
                self._load_fusion_baselines()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.cutoffs = pd.DataFrame(columns=['steps', 'average_loss', 'experiment_id', 'model_type'])
            if not os.path.exists(os.path.dirname(self.cutoffs_path)):
                os.makedirs(os.path.dirname(self.cutoffs_path))

    def on_epoch_end(self, epoch, logs=None):
        """Enhanced epoch monitoring with fusion support"""
        current_loss = logs.get('val_loss')
        
        # Track static feature impact if available
        if self.model_type == 'fusion' and 'static_feature_loss' in logs:
            sf_impact = logs['static_feature_loss'] * self.static_feature_weight
            self.static_feature_impacts.append(sf_impact)
            mlflow.log_metric('static_feature_impact', sf_impact, step=epoch)
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.convergence_history.append(('improvement', epoch))
        else:
            self.convergence_history.append(('plateau', epoch))
        
        # Skip early checks for model stability
        if epoch < self.min_epochs:
            return
            
        if ((epoch + 1) % self.iterations) == 0:
            self._update_cutoff(epoch + 1)
            if self._should_stop(epoch + 1):
                self._handle_stopping(epoch)

    def _handle_stopping(self, epoch):
        """Execute stopping procedure with enhanced tracking"""
        self.stopped_epoch = epoch
        self.model.stop_training = True
        self.early_stopping_occurred = True
        
        mlflow.log_metrics({
            'early_stopped': 1,
            'final_epoch': epoch + 1,
            'best_loss': self.best_loss
        })
        
        if self.model_type == 'fusion':
            avg_sf_impact = np.mean(self.static_feature_impacts) if self.static_feature_impacts else 0
            mlflow.log_metric('avg_static_feature_impact', avg_sf_impact)

    def _should_stop(self, current_step: int) -> bool:
        """Enhanced stopping logic with model-type awareness"""
        cutoff = self.cutoffs[
            (self.cutoffs['steps'] == current_step) & 
            (self.cutoffs['model_type'] == self.model_type)
        ]['average_loss'].iloc[0]
        
        # Apply model-specific thresholds
        if self.model_type == 'fusion':
            return self.best_loss > (cutoff * (1 + self.static_feature_weight))
        return self.best_loss > cutoff

    def _recalculate_cutoffs(self):
        """Enhanced cutoff calculation with model-type separation"""
        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            filter_string="attributes.status IN ('FINISHED', 'RUNNING')"
        )

        run_info = []
        for run in runs:
            try:
                metrics = client.get_metric_history(run.info.run_id, 'val_loss')
                if metrics:
                    model_type = run.data.tags.get('model_type', 'standard')
                    df = pd.DataFrame({
                        'step': [m.step for m in metrics],
                        'loss': [m.value for m in metrics],
                        'model_type': model_type
                    })
                    df['steps'] = df['step'] + 1
                    df['average_loss'] = df['loss'].cumsum() / df['steps']
                    df['experiment_id'] = run.info.experiment_id
                    run_info.append(df)
            except Exception as e:
                print(f"Skipping run {run.info.run_id}: {str(e)}")

        if run_info:
            all_runs = pd.concat(run_info)
            self.cutoffs = all_runs.groupby(
                ['experiment_id', 'steps', 'model_type']
            ).median().reset_index()

    def _load_fusion_baselines(self):
        """Load fusion-specific baseline metrics if available"""
        fusion_cutoffs = self.cutoffs[
            self.cutoffs['model_type'] == 'fusion'
        ]
        if not fusion_cutoffs.empty:
            self.fusion_baselines = fusion_cutoffs.set_index('steps')['average_loss'].to_dict()

    def on_train_end(self, logs=None):
        """Finalize training and persist updated cutoffs"""
        if self.early_stopping_occurred:
            print(f'\nEarly stopping at epoch {self.stopped_epoch + 1} '
                  f'(best val_loss: {self.best_loss:.4f})')
            
        self._update_cutoff_values()
        self._log_convergence_pattern()

    def _log_convergence_pattern(self):
        """Analyze and log convergence characteristics"""
        improvements = sum(1 for t, _ in self.convergence_history if t == 'improvement')
        plateaus = len(self.convergence_history) - improvements
        
        mlflow.log_metrics({
            'convergence_improvements': improvements,
            'convergence_plateaus': plateaus,
            'improvement_ratio': improvements / max(1, len(self.convergence_history))
        })
