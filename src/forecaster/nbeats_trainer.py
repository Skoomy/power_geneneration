"""Training and inference pipeline for N-BEATS demand forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import NBeats
import optuna

from .nbeats_model import NBEATSDemandModel


class NBEATSForecaster:
    """Main class for training and using the N-BEATS demand forecasting model."""
    
    def __init__(self,
                 max_prediction_length: int = 24,
                 max_encoder_length: int = 168,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 learning_rate: float = 0.001,
                 use_interpretable: bool = False,
                 ensemble_size: int = 1):
        """
        Initialize N-BEATS forecaster.
        
        Args:
            max_prediction_length: Forecast horizon (default 24 hours)
            max_encoder_length: Historical context (default 168 hours = 7 days)
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            learning_rate: Initial learning rate
            use_interpretable: Use interpretable N-BEATS (trend + seasonality)
            ensemble_size: Number of models to train for ensemble (1 = no ensemble)
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_interpretable = use_interpretable
        self.ensemble_size = ensemble_size
        
        # Initialize N-BEATS model wrapper
        if use_interpretable:
            # Interpretable configuration with trend and seasonality stacks
            self.nbeats_model = NBEATSDemandModel(
                max_prediction_length=max_prediction_length,
                max_encoder_length=max_encoder_length,
                widths=[512, 512],  # Trend and seasonality stacks
                num_blocks=[3, 3],
                num_block_layers=[4, 4],
                expansion_coefficient_lengths=[3, 7],  # Trend uses 3, seasonality uses 7
                learning_rate=learning_rate
            )
        else:
            # Generic N-BEATS configuration
            self.nbeats_model = NBEATSDemandModel(
                max_prediction_length=max_prediction_length,
                max_encoder_length=max_encoder_length,
                learning_rate=learning_rate
            )
        
        self.trainers = []
        self.best_model_paths = []
        self.ensemble_models = []
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    target_col: str = "demande_chaleur",
                    time_varying_known_reals: Optional[List[str]] = None,
                    val_split_date: Optional[pd.Timestamp] = None,
                    use_external_features: bool = False) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            time_varying_known_reals: Known future features (e.g., weather)
            val_split_date: Date to split train/validation
            use_external_features: Whether to include external features
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        # Configure external features if requested
        if use_external_features and time_varying_known_reals is None:
            # Use weather features if available
            possible_features = ["temperature", "lumiere", "pluviometrie"]
            time_varying_known_reals = [col for col in possible_features if col in df.columns]
        elif not use_external_features:
            time_varying_known_reals = []
        
        # Create datasets
        train_dataset, val_dataset = self.nbeats_model.prepare_data(
            df=df,
            target_col=target_col,
            time_varying_known_reals=time_varying_known_reals,
            val_split_date=val_split_date
        )
        
        # Create data loaders
        train_dataloader = train_dataset.to_dataloader(
            train=True, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
        
        val_dataloader = val_dataset.to_dataloader(
            train=False, 
            batch_size=self.batch_size * 2, 
            num_workers=self.num_workers
        )
        
        return train_dataloader, val_dataloader
    
    def train(self,
             train_dataloader: DataLoader,
             val_dataloader: DataLoader,
             max_epochs: int = 100,
             patience: int = 10,
             min_delta: float = 0.001,
             gpus: Union[int, List[int]] = 0,
             log_dir: str = "lightning_logs",
             model_name: str = "nbeats_demand_model",
             optimize_hyperparameters: bool = False,
             optuna_trials: int = 20) -> Union[pl.Trainer, List[pl.Trainer]]:
        """
        Train the N-BEATS model(s).
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            gpus: GPU configuration
            log_dir: Directory for logs
            model_name: Name for saved model
            optimize_hyperparameters: Whether to optimize hyperparameters
            optuna_trials: Number of Optuna trials if optimizing
            
        Returns:
            Trained PyTorch Lightning trainer(s)
        """
        # Hyperparameter optimization if requested
        if optimize_hyperparameters:
            print("Optimizing hyperparameters with Optuna...")
            best_params = self.nbeats_model.optimize_hyperparameters(
                train_dataloader,
                val_dataloader,
                n_trials=optuna_trials,
                use_interpretable=self.use_interpretable
            )
            print(f"Best hyperparameters: {best_params}")
            
            # Update model with best parameters
            for key, value in best_params.items():
                if hasattr(self.nbeats_model.model_config, key):
                    self.nbeats_model.model_config[key] = value
        
        # Train ensemble or single model
        for ensemble_idx in range(self.ensemble_size):
            # Create model
            model = self.nbeats_model.create_model()
            
            # Setup callbacks
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=min_delta,
                patience=patience,
                verbose=True,
                mode="min"
            )
            
            lr_monitor = LearningRateMonitor(logging_interval="epoch")
            
            suffix = f"_ensemble_{ensemble_idx}" if self.ensemble_size > 1 else ""
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"checkpoints/{model_name}{suffix}",
                filename="{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                mode="min"
            )
            
            # Setup logger
            logger = TensorBoardLogger(log_dir, name=f"{model_name}{suffix}")
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices=gpus if isinstance(gpus, list) else [gpus] if gpus > 0 else "auto",
                enable_model_summary=True,
                gradient_clip_val=0.1,
                callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
                logger=logger,
            )
            
            # Train model
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            
            # Store trainer and best model path
            self.trainers.append(trainer)
            self.best_model_paths.append(checkpoint_callback.best_model_path)
            
            print(f"Model {ensemble_idx + 1}/{self.ensemble_size} trained. Best checkpoint: {checkpoint_callback.best_model_path}")
        
        return self.trainers if self.ensemble_size > 1 else self.trainers[0]
    
    def predict_next_day(self, 
                        current_data: pd.DataFrame,
                        model_paths: Optional[List[str]] = None,
                        return_ensemble: bool = True) -> pd.DataFrame:
        """
        Predict demand for the next 24 hours.
        
        Args:
            current_data: Recent data including features
            model_paths: Paths to saved models (uses best from training if None)
            return_ensemble: If True and ensemble trained, return ensemble predictions
            
        Returns:
            DataFrame with hourly predictions
        """
        if model_paths is None:
            model_paths = self.best_model_paths
            
        if not model_paths:
            raise ValueError("No model paths provided and no models trained")
        
        # Create prediction dataloader
        prediction_data = self.nbeats_model.validation_dataset.filter(
            lambda x: x.time_idx >= current_data["time_idx"].max() - self.nbeats_model.max_encoder_length
        )
        
        dataloader = prediction_data.to_dataloader(
            train=False, 
            batch_size=1, 
            num_workers=0
        )
        
        if self.ensemble_size > 1 and return_ensemble:
            # Load all ensemble models
            models = []
            for path in model_paths:
                model = NBeats.load_from_checkpoint(path)
                models.append(model)
            
            # Get ensemble predictions
            predictions_df = self.nbeats_model.ensemble_predictions(
                models, dataloader
            )
        else:
            # Single model prediction
            model = NBeats.load_from_checkpoint(model_paths[0])
            model.eval()
            
            predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    x, _ = batch
                    pred = model(x)
                    pred_df = self.nbeats_model.get_prediction(pred)
                    predictions.append(pred_df)
            
            predictions_df = pd.concat(predictions, ignore_index=True)
        
        # Add timestamps
        last_time = current_data.index[-1] if isinstance(current_data.index, pd.DatetimeIndex) else pd.to_datetime(current_data["datetime"].iloc[-1])
        predictions_df["datetime"] = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=len(predictions_df),
            freq="H"
        )
        
        # Select next 24 hours
        next_day_predictions = predictions_df.head(24).copy()
        
        # Rename columns for consistency
        if "demand_mean" in next_day_predictions.columns:
            return next_day_predictions[["datetime", "demand_lower", "demand_mean", "demand_upper", "demand_std"]]
        else:
            return next_day_predictions[["datetime", "demand_forecast"]]
    
    def evaluate_model(self, 
                      test_dataloader: DataLoader,
                      model_paths: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_dataloader: Test data loader
            model_paths: Paths to saved models
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_paths is None:
            model_paths = self.best_model_paths
        
        all_metrics = []
        
        for path in model_paths:
            # Load model
            model = NBeats.load_from_checkpoint(path)
            model.eval()
            
            # Get predictions and targets
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch in test_dataloader:
                    x, y = batch
                    pred = model(x)
                    predictions.append(pred.cpu())
                    targets.append(y[0].cpu())  # y is (target, weight) tuple
            
            y_pred = torch.cat(predictions)
            y_true = torch.cat(targets)
            
            # Calculate metrics
            mae = torch.abs(y_true - y_pred).mean().item()
            rmse = torch.sqrt(((y_true - y_pred) ** 2).mean()).item()
            mape = (torch.abs((y_true - y_pred) / (y_true + 1e-8)) * 100).mean().item()
            
            metrics = {
                "mae": mae,
                "rmse": rmse,
                "mape": mape
            }
            
            all_metrics.append(metrics)
        
        # Average metrics if ensemble
        if len(all_metrics) > 1:
            avg_metrics = {
                "mae": np.mean([m["mae"] for m in all_metrics]),
                "rmse": np.mean([m["rmse"] for m in all_metrics]),
                "mape": np.mean([m["mape"] for m in all_metrics]),
                "mae_std": np.std([m["mae"] for m in all_metrics]),
                "rmse_std": np.std([m["rmse"] for m in all_metrics]),
                "mape_std": np.std([m["mape"] for m in all_metrics]),
            }
            return avg_metrics
        else:
            return all_metrics[0]
    
    def plot_predictions(self,
                        actual_data: pd.DataFrame,
                        predictions: pd.DataFrame,
                        title: str = "N-BEATS Demand Forecast") -> None:
        """
        Plot predictions with confidence intervals if available.
        
        Args:
            actual_data: Historical demand data
            predictions: Predictions (with or without intervals)
            title: Plot title
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        if len(actual_data) > 168:  # Show last week of history
            actual_data = actual_data.iloc[-168:]
            
        ax.plot(actual_data.index, actual_data["demande_chaleur"], 
                label="Historical Demand", color="black", alpha=0.7)
        
        # Plot predictions
        pred_dates = pd.to_datetime(predictions["datetime"])
        
        if "demand_mean" in predictions.columns:
            # Ensemble predictions with uncertainty
            ax.plot(pred_dates, predictions["demand_mean"], 
                    label="Forecast (Mean)", color="blue", linewidth=2)
            
            if "demand_lower" in predictions.columns:
                ax.fill_between(pred_dates, 
                               predictions["demand_lower"],
                               predictions["demand_upper"],
                               alpha=0.3, color="blue", 
                               label="95% Confidence Interval")
        else:
            # Single model predictions
            ax.plot(pred_dates, predictions["demand_forecast"], 
                    label="Forecast", color="blue", linewidth=2)
        
        # Formatting
        ax.set_xlabel("Date")
        ax.set_ylabel("Demand (MW)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_tft(self,
                        test_data: pd.DataFrame,
                        tft_predictions: pd.DataFrame,
                        nbeats_predictions: pd.DataFrame) -> None:
        """
        Compare N-BEATS predictions with TFT predictions.
        
        Args:
            test_data: Actual test data
            tft_predictions: TFT model predictions
            nbeats_predictions: N-BEATS model predictions
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot predictions
        ax1.plot(test_data.index[:24], test_data["demande_chaleur"][:24], 
                label="Actual", color="black", linewidth=2)
        ax1.plot(test_data.index[:24], tft_predictions["demand_median"][:24], 
                label="TFT", color="red", alpha=0.7)
        
        if "demand_mean" in nbeats_predictions.columns:
            ax1.plot(test_data.index[:24], nbeats_predictions["demand_mean"][:24], 
                    label="N-BEATS", color="blue", alpha=0.7)
        else:
            ax1.plot(test_data.index[:24], nbeats_predictions["demand_forecast"][:24], 
                    label="N-BEATS", color="blue", alpha=0.7)
        
        ax1.set_ylabel("Demand (MW)")
        ax1.set_title("Model Comparison: 24-hour Forecast")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot errors
        tft_error = test_data["demande_chaleur"][:24] - tft_predictions["demand_median"][:24]
        
        if "demand_mean" in nbeats_predictions.columns:
            nbeats_error = test_data["demande_chaleur"][:24] - nbeats_predictions["demand_mean"][:24]
        else:
            nbeats_error = test_data["demande_chaleur"][:24] - nbeats_predictions["demand_forecast"][:24]
        
        ax2.plot(test_data.index[:24], tft_error, label="TFT Error", color="red", alpha=0.7)
        ax2.plot(test_data.index[:24], nbeats_error, label="N-BEATS Error", color="blue", alpha=0.7)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Prediction Error (MW)")
        ax2.set_title("Prediction Errors")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison metrics
        tft_mae = np.abs(tft_error).mean()
        nbeats_mae = np.abs(nbeats_error).mean()
        tft_rmse = np.sqrt((tft_error ** 2).mean())
        nbeats_rmse = np.sqrt((nbeats_error ** 2).mean())
        
        print("\nModel Comparison Metrics (24-hour forecast):")
        print(f"TFT    - MAE: {tft_mae:.2f} MW, RMSE: {tft_rmse:.2f} MW")
        print(f"N-BEATS - MAE: {nbeats_mae:.2f} MW, RMSE: {nbeats_rmse:.2f} MW")
        print(f"Improvement: MAE {((tft_mae - nbeats_mae) / tft_mae * 100):.1f}%, "
              f"RMSE {((tft_rmse - nbeats_rmse) / tft_rmse * 100):.1f}%")