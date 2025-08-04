"""Temporal Fusion Transformer model using PyTorch Forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


class TFTDemandModel:
    """Wrapper for Temporal Fusion Transformer from PyTorch Forecasting."""
    
    def __init__(self,
                 max_prediction_length: int = 24,
                 max_encoder_length: int = 168,
                 quantiles: List[float] = [0.1, 0.5, 0.9],
                 hidden_size: int = 128,
                 attention_head_size: int = 4,
                 dropout: float = 0.1,
                 hidden_continuous_size: int = 64,
                 learning_rate: float = 0.001):
        """
        Initialize TFT model configuration.
        
        Args:
            max_prediction_length: Forecast horizon (24 hours)
            max_encoder_length: Historical context (168 hours = 7 days)
            quantiles: Quantiles for prediction intervals
            hidden_size: Size of hidden layers
            attention_head_size: Number of attention heads
            dropout: Dropout rate
            hidden_continuous_size: Size of hidden layers for continuous variables
            learning_rate: Learning rate for optimization
        """
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.quantiles = quantiles
        
        # Model configuration
        self.model_config = {
            "hidden_size": hidden_size,
            "attention_head_size": attention_head_size,
            "dropout": dropout,
            "hidden_continuous_size": hidden_continuous_size,
            "output_size": len(quantiles),
            "loss": QuantileLoss(quantiles),
            "log_interval": 10,
            "reduce_on_plateau_patience": 4,
            "learning_rate": learning_rate
        }
        
        self.model = None
        self.training_dataset = None
        self.validation_dataset = None
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    time_idx_col: str = "time_idx",
                    target_col: str = "demande_chaleur",
                    group_ids: Optional[List[str]] = None,
                    time_varying_known_reals: Optional[List[str]] = None,
                    time_varying_unknown_reals: Optional[List[str]] = None,
                    static_categoricals: Optional[List[str]] = None,
                    static_reals: Optional[List[str]] = None,
                    val_split_date: Optional[pd.Timestamp] = None) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Prepare data for TFT training using PyTorch Forecasting format.
        
        Args:
            df: DataFrame with features and target
            time_idx_col: Column name for time index
            target_col: Target column name
            group_ids: List of group identifier columns
            time_varying_known_reals: Known future real variables
            time_varying_unknown_reals: Unknown future real variables
            static_categoricals: Static categorical variables
            static_reals: Static real variables
            val_split_date: Date to split train/validation
        """
        # Add time index if not present
        if time_idx_col not in df.columns:
            df[time_idx_col] = (df.index - df.index[0]).total_seconds() / 3600
            df[time_idx_col] = df[time_idx_col].astype(int)
        
        # Add group column if not specified
        if group_ids is None:
            df["group"] = "total"
            group_ids = ["group"]
        
        # Default time-varying known reals (we know future weather forecast)
        if time_varying_known_reals is None:
            time_varying_known_reals = [
                "hour", "day_of_week", "day_of_month", "month",
                "temperature", "lumiere", "pluviometrie",
                "hour_sin", "hour_cos", "day_sin", "day_cos"
            ]
            # Filter to existing columns
            time_varying_known_reals = [col for col in time_varying_known_reals if col in df.columns]
        
        # Default time-varying unknown reals (derived from target)
        if time_varying_unknown_reals is None:
            time_varying_unknown_reals = [
                col for col in df.columns 
                if col.startswith(f"{target_col}_lag") or col.startswith(f"{target_col}_roll")
            ]
        
        # Train/validation split
        if val_split_date is None:
            # Use last 20% of data for validation
            split_idx = int(len(df) * 0.8)
            val_split_date = df.index[split_idx]
        
        # Create training dataset
        self.training_dataset = TimeSeriesDataSet(
            df[df.index < val_split_date],
            time_idx=time_idx_col,
            target=target_col,
            group_ids=group_ids,
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=static_categoricals or [],
            static_reals=static_reals or [],
            time_varying_known_categoricals=[],
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=time_varying_unknown_reals + [target_col],
            target_normalizer=GroupNormalizer(
                groups=group_ids,
                transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        # Create validation dataset
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, 
            df, 
            predict=True, 
            stop_randomization=True
        )
        
        return self.training_dataset, self.validation_dataset
    
    def create_model(self) -> TemporalFusionTransformer:
        """Create TFT model instance."""
        if self.training_dataset is None:
            raise ValueError("Must call prepare_data() before creating model")
        
        # Create model with configuration
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            **self.model_config
        )
        
        return self.model
    
    def optimize_hyperparameters(self, 
                               train_dataloader: torch.utils.data.DataLoader,
                               val_dataloader: torch.utils.data.DataLoader,
                               n_trials: int = 20,
                               timeout: int = 3600) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds
            
        Returns:
            Best hyperparameters found
        """
        # Use built-in hyperparameter optimization
        study = optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            model_path="optuna_trial",
            n_trials=n_trials,
            max_epochs=50,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(64, 256),
            hidden_continuous_size_range=(32, 128),
            attention_head_size_range=(1, 8),
            learning_rate_range=(0.0001, 0.01),
            dropout_range=(0.05, 0.3),
            trainer_kwargs={
                "limit_train_batches": 50,
                "accelerator": "auto",
            },
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
            timeout=timeout
        )
        
        return study.best_params
    
    def get_prediction_with_intervals(self, 
                                    predictions: torch.Tensor,
                                    prediction_outputs: Dict) -> pd.DataFrame:
        """
        Extract predictions with confidence intervals.
        
        Args:
            predictions: Raw model predictions
            prediction_outputs: Additional outputs from model
            
        Returns:
            DataFrame with predictions and intervals
        """
        # Extract quantile predictions
        quantile_preds = predictions.cpu().numpy()
        
        # Get time index from outputs
        decoder_lengths = prediction_outputs["decoder_lengths"].cpu().numpy()
        
        results = []
        for batch_idx in range(len(quantile_preds)):
            length = decoder_lengths[batch_idx]
            
            for t in range(length):
                result = {
                    "prediction_step": t + 1,
                    "demand_lower": quantile_preds[batch_idx, t, 0],  # 10th percentile
                    "demand_median": quantile_preds[batch_idx, t, 1],  # 50th percentile  
                    "demand_upper": quantile_preds[batch_idx, t, 2],  # 90th percentile
                    "demand_mean": quantile_preds[batch_idx, t, :].mean()
                }
                results.append(result)
        
        return pd.DataFrame(results)