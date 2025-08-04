"""Training and inference pipeline for demand forecasting using PyTorch Forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer

from .model import TFTDemandModel


class DemandForecaster:
    """Main class for training and using the demand forecasting model."""
    
    def __init__(self,
                 max_prediction_length: int = 24,
                 max_encoder_length: int = 168,
                 quantiles: List[float] = [0.1, 0.5, 0.9],
                 batch_size: int = 64,
                 num_workers: int = 0,
                 learning_rate: float = 0.001,
                 hidden_size: int = 128,
                 attention_head_size: int = 4,
                 dropout: float = 0.1,
                 hidden_continuous_size: int = 64):
        """
        Initialize demand forecaster.
        
        Args:
            max_prediction_length: Forecast horizon (default 24 hours)
            max_encoder_length: Historical context (default 168 hours = 7 days)
            quantiles: Quantiles for prediction intervals
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            learning_rate: Initial learning rate
            hidden_size: Hidden layer size
            attention_head_size: Number of attention heads
            dropout: Dropout rate
            hidden_continuous_size: Hidden size for continuous variables
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize TFT model wrapper
        self.tft_model = TFTDemandModel(
            max_prediction_length=max_prediction_length,
            max_encoder_length=max_encoder_length,
            quantiles=quantiles,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            learning_rate=learning_rate
        )
        
        self.trainer = None
        self.best_model_path = None
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    target_col: str = "demande_chaleur",
                    time_varying_known_reals: Optional[List[str]] = None,
                    time_varying_unknown_reals: Optional[List[str]] = None,
                    val_split_date: Optional[pd.Timestamp] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            time_varying_known_reals: Known future features
            time_varying_unknown_reals: Unknown future features
            val_split_date: Date to split train/validation
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        # Create datasets
        train_dataset, val_dataset = self.tft_model.prepare_data(
            df=df,
            target_col=target_col,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
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
             model_name: str = "tft_demand_model") -> pl.Trainer:
        """
        Train the TFT model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            gpus: GPU configuration
            log_dir: Directory for logs
            model_name: Name for saved model
            
        Returns:
            Trained PyTorch Lightning trainer
        """
        # Create model
        model = self.tft_model.create_model()
        
        # Setup callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=min_delta,
            patience=patience,
            verbose=True,
            mode="min"
        )
        
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"checkpoints/{model_name}",
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            mode="min"
        )
        
        # Setup logger
        logger = TensorBoardLogger(log_dir, name=model_name)
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=gpus if isinstance(gpus, list) else [gpus] if gpus > 0 else "auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
            logger=logger,
        )
        
        # Train model
        self.trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # Store best model path
        self.best_model_path = checkpoint_callback.best_model_path
        
        return self.trainer
    
    def predict_next_day(self, 
                        current_data: pd.DataFrame,
                        model_path: Optional[str] = None) -> pd.DataFrame:
        """
        Predict demand for the next 24 hours with confidence intervals.
        
        Args:
            current_data: Recent data including features
            model_path: Path to saved model (uses best from training if None)
            
        Returns:
            DataFrame with hourly predictions and confidence intervals
        """
        if model_path is None:
            model_path = self.best_model_path
            
        if model_path is None:
            raise ValueError("No model path provided and no model trained")
        
        # Load best model
        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        
        # Create prediction dataloader
        prediction_data = self.tft_model.validation_dataset.filter(
            lambda x: x.time_idx >= current_data["time_idx"].max() - self.tft_model.max_encoder_length
        )
        
        dataloader = prediction_data.to_dataloader(
            train=False, 
            batch_size=1, 
            num_workers=0
        )
        
        # Make predictions
        predictions = []
        model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # Get predictions
                prediction_outputs = model.predict(
                    batch, 
                    mode="quantiles",
                    return_x=True
                )
                
                # Extract predictions with intervals
                pred_df = self.tft_model.get_prediction_with_intervals(
                    prediction_outputs["prediction"],
                    prediction_outputs
                )
                
                predictions.append(pred_df)
        
        # Combine predictions
        all_predictions = pd.concat(predictions, ignore_index=True)
        
        # Add timestamps
        last_time = current_data.index[-1] if isinstance(current_data.index, pd.DatetimeIndex) else pd.to_datetime(current_data["datetime"].iloc[-1])
        all_predictions["datetime"] = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=len(all_predictions),
            freq="H"
        )
        
        # Select next 24 hours
        next_day_predictions = all_predictions.head(24).copy()
        
        return next_day_predictions[["datetime", "demand_lower", "demand_median", "demand_upper", "demand_mean"]]
    
    def evaluate_model(self, 
                      test_dataloader: DataLoader,
                      model_path: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_dataloader: Test data loader
            model_path: Path to saved model
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_path is None:
            model_path = self.best_model_path
            
        # Load model
        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        
        # Get predictions
        predictions = model.predict(test_dataloader, mode="quantiles", return_x=True)
        
        # Calculate metrics
        y_true = torch.cat([batch[1][0] for batch in test_dataloader])
        y_pred = predictions["prediction"]
        
        # Quantile coverage
        lower_coverage = (y_true >= y_pred[..., 0]).float().mean().item()
        upper_coverage = (y_true <= y_pred[..., 2]).float().mean().item()
        interval_coverage = ((y_true >= y_pred[..., 0]) & (y_true <= y_pred[..., 2])).float().mean().item()
        
        # Mean absolute error on median prediction
        mae = torch.abs(y_true - y_pred[..., 1]).mean().item()
        
        # Root mean squared error
        rmse = torch.sqrt(((y_true - y_pred[..., 1]) ** 2).mean()).item()
        
        # Interval width
        interval_width = (y_pred[..., 2] - y_pred[..., 0]).mean().item()
        
        return {
            "mae": mae,
            "rmse": rmse,
            "interval_coverage": interval_coverage,
            "lower_coverage": lower_coverage,
            "upper_coverage": upper_coverage,
            "mean_interval_width": interval_width
        }
    
    def plot_predictions(self,
                        actual_data: pd.DataFrame,
                        predictions: pd.DataFrame,
                        title: str = "Demand Forecast with Confidence Intervals") -> None:
        """
        Plot predictions with confidence intervals.
        
        Args:
            actual_data: Historical demand data
            predictions: Predictions with intervals
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
        ax.plot(pred_dates, predictions["demand_median"], 
                label="Forecast (Median)", color="blue", linewidth=2)
        
        # Confidence intervals
        ax.fill_between(pred_dates, 
                       predictions["demand_lower"],
                       predictions["demand_upper"],
                       alpha=0.3, color="blue", 
                       label="80% Confidence Interval")
        
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