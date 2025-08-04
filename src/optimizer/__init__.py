"""Power Generation Optimization Module

This module implements:
- Mixed Integer Linear Programming (MILP) optimization
- Stochastic optimization considering demand uncertainty
- Robust optimization with confidence intervals
- Integration with demand forecasts
"""

from .power_optimizer import PowerOptimizer
from .models import OptimizationResult, PlantSchedule

__all__ = ['PowerOptimizer', 'OptimizationResult', 'PlantSchedule']