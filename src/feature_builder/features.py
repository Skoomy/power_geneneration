"""Feature engineering for power demand forecasting."""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime


class FeatureBuilder:
    """Creates features for demand forecasting from weather and temporal data."""
    
    def __init__(self, 
                 lag_periods: List[int] = [1, 2, 3, 24, 48, 168],
                 rolling_windows: List[int] = [6, 12, 24, 48]):
        """
        Initialize feature builder.
        
        Args:
            lag_periods: List of lag periods for demand features (in hours)
            rolling_windows: List of window sizes for rolling statistics
        """
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from datetime index."""
        df = df.copy()
        
        # Basic temporal features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['week_of_year'] = df.index.isocalendar().week
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-based features."""
        df = df.copy()
        
        # Temperature features
        if 'temperature' in df.columns:
            df['temp_squared'] = df['temperature'] ** 2
            df['temp_cubed'] = df['temperature'] ** 3
            df['is_cold'] = (df['temperature'] < 0).astype(int)
            df['is_hot'] = (df['temperature'] > 25).astype(int)
            
            # Heating/cooling degree days approximation
            df['heating_degrees'] = np.maximum(18 - df['temperature'], 0)
            df['cooling_degrees'] = np.maximum(df['temperature'] - 22, 0)
        
        # Light intensity features
        if 'lumiere' in df.columns:
            df['is_dark'] = (df['lumiere'] == 0).astype(int)
            df['log_lumiere'] = np.log1p(df['lumiere'])
        
        # Rain features
        if 'pluviometrie' in df.columns:
            df['is_raining'] = (df['pluviometrie'] > 0).astype(int)
            df['rain_intensity'] = pd.cut(df['pluviometrie'], 
                                         bins=[0, 0.1, 1, 5, np.inf], 
                                         labels=[0, 1, 2, 3]).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'demande_chaleur') -> pd.DataFrame:
        """Create lag features for the target variable."""
        df = df.copy()
        
        for lag in self.lag_periods:
            df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
            
            # Lag differences
            if lag > 1:
                df[f'{target_col}_diff_{lag}h'] = df[target_col] - df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'demande_chaleur') -> pd.DataFrame:
        """Create rolling statistics features."""
        df = df.copy()
        
        for window in self.rolling_windows:
            # Rolling statistics
            df[f'{target_col}_roll_mean_{window}h'] = df[target_col].rolling(window, min_periods=1).mean()
            df[f'{target_col}_roll_std_{window}h'] = df[target_col].rolling(window, min_periods=1).std()
            df[f'{target_col}_roll_min_{window}h'] = df[target_col].rolling(window, min_periods=1).min()
            df[f'{target_col}_roll_max_{window}h'] = df[target_col].rolling(window, min_periods=1).max()
            
            # Relative position within rolling window
            df[f'{target_col}_roll_relative_{window}h'] = (
                (df[target_col] - df[f'{target_col}_roll_min_{window}h']) / 
                (df[f'{target_col}_roll_max_{window}h'] - df[f'{target_col}_roll_min_{window}h'] + 1e-8)
            )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different variables."""
        df = df.copy()
        
        # Temperature and time interactions
        if 'temperature' in df.columns:
            df['temp_hour_interaction'] = df['temperature'] * df['hour']
            df['temp_weekend_interaction'] = df['temperature'] * df['is_weekend']
            df['heating_degrees_night'] = df['heating_degrees'] * df['is_night']
        
        # Light and time interactions
        if 'lumiere' in df.columns:
            df['lumiere_hour_interaction'] = df['lumiere'] * df['hour']
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'demande_chaleur') -> pd.DataFrame:
        """Apply all feature engineering steps."""
        # Ensure datetime index
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        df.index = pd.to_datetime(df.index)
        
        # Apply feature engineering steps
        df = self.create_temporal_features(df)
        df = self.create_weather_features(df)
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df = self.create_interaction_features(df)
        
        # Handle missing values from lag/rolling features
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of engineered feature names."""
        original_cols = ['temperature', 'lumiere', 'pluviometrie', 'demande_chaleur']
        return [col for col in df.columns if col not in original_cols]