"""Feature Engineering Module for Demand Forecasting

This module handles:
- Time-based features (hour, day of week, month, seasonality)
- Weather feature transformations
- Lag features for demand
- Rolling statistics
- Holiday and special event indicators
"""

from .features import FeatureBuilder

__all__ = ['FeatureBuilder']