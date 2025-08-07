"""N-BEATS model using PyTorch Forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, QuantileLoss
import optuna
from optuna.integration import PyTorchLightningPruningCallback


class NBEATSDemandModel:
    """Wrapper for N-BEATS from PyTorch Forecasting."""
    
    def __init__(self,
                 max_prediction_length: int = 24,
                 max_encoder_length: int = 168,
                 widths: List[int] = [512, 512, 512, 512],
                 num_blocks: List[int] = [1, 1, 1, 1],
                 num_block_layers: List[int] = [4, 4, 4, 4],
                 expansion_coefficient_lengths: List[int] = [32, 32, 32, 32],
                 share_weights_in_stack: bool = False,
                 learning_rate: float = 0.001,
                 dropout: float = 0.1,
                 backcast_loss_ratio: float = 0.0):
        """
        Initialize N-BEATS model configuration.
        
        Args:
            max_prediction_length: Forecast horizon (24 hours)
            max_encoder_length: Historical context (168 hours = 7 days)
            widths: Widths of the fully connected layers for each stack
            num_blocks: Number of blocks in each stack
            num_block_layers: Number of fully connected layers in each block
            expansion_coefficient_lengths: Size of expansion coefficients for each stack
            share_weights_in_stack: Whether to share weights in stack
            learning_rate: Learning rate for optimization
            dropout: Dropout rate
            backcast_loss_ratio: Weight of backcast in loss (0 = only forecast)
        """
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        
        # Model configuration
        self.model_config = {
            "widths": widths,
            "num_blocks": num_blocks,
            "num_block_layers": num_block_layers,
            "expansion_coefficient_lengths": expansion_coefficient_lengths,
            "share_weights_in_stack": share_weights_in_stack,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "backcast_loss_ratio": backcast_loss_ratio,
            "loss": MAE(),
            "log_interval": 10,
            "reduce_on_plateau_patience": 4,
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
                    val_split_date: Optional[pd.Timestamp] = None) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Prepare data for N-BEATS training using PyTorch Forecasting format.
        
        Args:
            df: DataFrame with features and target
            time_idx_col: Column name for time index
            target_col: Target column name
            group_ids: List of group identifier columns
            time_varying_known_reals: Known future real variables (weather forecast)
            time_varying_unknown_reals: Unknown future real variables (lagged features)
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
        
        # N-BEATS typically works best with just the target, but we can include covariates
        # Default time-varying known reals (future weather)
        if time_varying_known_reals is None:
            time_varying_known_reals = []
            # N-BEATS can work with or without external regressors
            # Uncomment to use external features:
            # time_varying_known_reals = [
            #     "temperature", "lumiere", "pluviometrie"
            # ]
            # time_varying_known_reals = [col for col in time_varying_known_reals if col in df.columns]
        
        # N-BEATS doesn't typically use unknown reals since it's autoregressive
        if time_varying_unknown_reals is None:
            time_varying_unknown_reals = []
        
        # Train/validation split
        if val_split_date is None:
            split_idx = int(len(df) * 0.8)
            val_split_date = df.index[split_idx]
        
        # Create training dataset
        self.training_dataset = TimeSeriesDataSet(
            df[df.index < val_split_date],
            time_idx=time_idx_col,
            target=target_col,
            group_ids=group_ids,
            min_encoder_length=self.max_encoder_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=self.max_prediction_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=time_varying_unknown_reals,
            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=True,
            categorical_encoders={"group": NaNLabelEncoder().fit(df["group"])}
        )
        
        # Create validation dataset
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, 
            df, 
            predict=True, 
            stop_randomization=True
        )
        
        return self.training_dataset, self.validation_dataset
    
    def create_model(self) -> NBeats:
        """Create N-BEATS model instance."""
        if self.training_dataset is None:
            raise ValueError("Must call prepare_data() before creating model")
        
        # Create model with configuration
        self.model = NBeats.from_dataset(
            self.training_dataset,
            **self.model_config
        )
        
        return self.model
    
    def optimize_hyperparameters(self, 
                               train_dataloader: torch.utils.data.DataLoader,
                               val_dataloader: torch.utils.data.DataLoader,
                               n_trials: int = 20,
                               timeout: int = 3600,
                               use_interpretable: bool = False) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds
            use_interpretable: Whether to use interpretable N-BEATS architecture
            
        Returns:
            Best hyperparameters found
        """
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3),
                "backcast_loss_ratio": trial.suggest_float("backcast_loss_ratio", 0.0, 1.0),
            }
            
            if use_interpretable:
                # Interpretable N-BEATS (trend + seasonality stacks)
                params["stack_types"] = ["trend", "seasonality"]
                params["num_blocks"] = [
                    trial.suggest_int("num_blocks_trend", 1, 3),
                    trial.suggest_int("num_blocks_seasonality", 1, 3)
                ]
                params["num_block_layers"] = [
                    trial.suggest_int("num_layers_trend", 2, 4),
                    trial.suggest_int("num_layers_seasonality", 2, 4)
                ]
                params["widths"] = [
                    trial.suggest_int("width_trend", 256, 512, step=128),
                    trial.suggest_int("width_seasonality", 256, 512, step=128)
                ]
            else:
                # Generic N-BEATS
                n_stacks = trial.suggest_int("n_stacks", 2, 6)
                params["widths"] = [trial.suggest_int(f"width_{i}", 256, 512, step=128) 
                                   for i in range(n_stacks)]
                params["num_blocks"] = [trial.suggest_int(f"blocks_{i}", 1, 3) 
                                       for i in range(n_stacks)]
                params["num_block_layers"] = [trial.suggest_int(f"layers_{i}", 2, 4) 
                                             for i in range(n_stacks)]
                params["expansion_coefficient_lengths"] = [trial.suggest_int(f"expansion_{i}", 16, 64, step=16) 
                                                          for i in range(n_stacks)]
            
            params["share_weights_in_stack"] = trial.suggest_categorical("share_weights", [True, False])
            
            # Create model
            model = NBeats.from_dataset(
                self.training_dataset,
                loss=MAE(),
                log_interval=10,
                reduce_on_plateau_patience=4,
                **params
            )
            
            # Train
            trainer = pl.Trainer(
                max_epochs=50,
                accelerator="auto",
                enable_model_summary=False,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=5, min_delta=0.001),
                    PyTorchLightningPruningCallback(trial, monitor="val_loss")
                ],
                logger=False,
                enable_checkpointing=False,
            )
            
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
            
            return trainer.callback_metrics["val_loss"].item()
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        return study.best_params
    
    def get_prediction(self, 
                      predictions: torch.Tensor) -> pd.DataFrame:
        """
        Extract predictions from model output.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            DataFrame with predictions
        """
        # N-BEATS returns point forecasts
        preds = predictions.cpu().numpy()
        
        results = []
        for batch_idx in range(len(preds)):
            for t in range(self.max_prediction_length):
                result = {
                    "prediction_step": t + 1,
                    "demand_forecast": preds[batch_idx, t]
                }
                results.append(result)
        
        return pd.DataFrame(results)
    
    def ensemble_predictions(self,
                           models: List[NBeats],
                           dataloader: torch.utils.data.DataLoader,
                           weights: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Create ensemble predictions from multiple N-BEATS models.
        
        Args:
            models: List of trained N-BEATS models
            dataloader: Data loader for predictions
            weights: Optional weights for ensemble (equal weights if None)
            
        Returns:
            DataFrame with ensemble predictions and uncertainty estimates
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        all_predictions = []
        
        for model in models:
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch in dataloader:
                    x, _ = batch
                    pred = model(x)
                    predictions.append(pred.cpu().numpy())
            
            all_predictions.append(np.concatenate(predictions, axis=0))
        
        # Stack predictions
        stacked = np.stack(all_predictions, axis=0)  # (n_models, n_samples, horizon)
        
        # Weighted average
        ensemble_mean = np.average(stacked, axis=0, weights=weights)
        
        # Estimate uncertainty from ensemble
        ensemble_std = np.std(stacked, axis=0)
        ensemble_lower = ensemble_mean - 1.96 * ensemble_std  # ~95% CI
        ensemble_upper = ensemble_mean + 1.96 * ensemble_std
        
        results = []
        for batch_idx in range(ensemble_mean.shape[0]):
            for t in range(self.max_prediction_length):
                result = {
                    "prediction_step": t + 1,
                    "demand_mean": ensemble_mean[batch_idx, t],
                    "demand_std": ensemble_std[batch_idx, t],
                    "demand_lower": ensemble_lower[batch_idx, t],
                    "demand_upper": ensemble_upper[batch_idx, t]
                }
                results.append(result)
        
        return pd.DataFrame(results)