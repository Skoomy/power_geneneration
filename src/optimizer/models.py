"""Data models for optimization results."""

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class PlantSchedule:
    """Schedule for a single power plant."""
    plant_id: int
    plant_name: str
    hours: List[int]
    on_off_status: List[int]  # Binary: 1 if on, 0 if off
    power_output: List[float]  # MW generated each hour
    operating_cost: List[float]  # Cost per hour
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert schedule to DataFrame."""
        return pd.DataFrame({
            'hour': self.hours,
            'on_off': self.on_off_status,
            'power_mw': self.power_output,
            'cost_eur': self.operating_cost
        })
    
    @property
    def total_cost(self) -> float:
        """Total operating cost."""
        return sum(self.operating_cost)
    
    @property
    def total_energy(self) -> float:
        """Total energy generated (MWh)."""
        return sum(self.power_output)
    
    @property
    def capacity_factor(self) -> float:
        """Capacity utilization factor."""
        if not self.power_output:
            return 0.0
        max_capacity = max(self.power_output) if max(self.power_output) > 0 else 1.0
        return self.total_energy / (max_capacity * len(self.hours))


@dataclass
class OptimizationResult:
    """Results from power generation optimization."""
    
    # Demand information
    demand_forecast: pd.DataFrame  # Contains demand predictions with intervals
    
    # Plant schedules
    plant_schedules: Dict[int, PlantSchedule]
    
    # Overall metrics
    total_cost: float
    total_generation: float
    optimization_status: str  # 'optimal', 'feasible', 'infeasible'
    solve_time: float  # seconds
    
    # Stochastic optimization results (if applicable)
    expected_cost: Optional[float] = None
    worst_case_cost: Optional[float] = None
    confidence_level: Optional[float] = None
    
    # Scenario results for stochastic optimization
    scenario_results: Optional[List[Dict]] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to comprehensive DataFrame."""
        # Combine all plant schedules
        all_schedules = []
        
        for plant_id, schedule in self.plant_schedules.items():
            df = schedule.to_dataframe()
            df['plant_id'] = plant_id
            df['plant_name'] = schedule.plant_name
            all_schedules.append(df)
        
        if all_schedules:
            combined_df = pd.concat(all_schedules, ignore_index=True)
            
            # Add demand information
            demand_by_hour = self.demand_forecast.groupby('hour')['demand_median'].first()
            combined_df = combined_df.merge(
                demand_by_hour.to_frame('demand_mw'), 
                left_on='hour', 
                right_index=True,
                how='left'
            )
            
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_summary(self) -> Dict:
        """Get optimization summary."""
        summary = {
            'status': self.optimization_status,
            'total_cost_eur': self.total_cost,
            'total_generation_mwh': self.total_generation,
            'solve_time_seconds': self.solve_time,
            'num_plants_used': sum(1 for p in self.plant_schedules.values() 
                                 if any(p.on_off_status)),
            'average_cost_per_mwh': self.total_cost / self.total_generation if self.total_generation > 0 else 0
        }
        
        if self.expected_cost is not None:
            summary.update({
                'expected_cost_eur': self.expected_cost,
                'worst_case_cost_eur': self.worst_case_cost,
                'confidence_level': self.confidence_level
            })
        
        # Plant utilization
        for plant_id, schedule in self.plant_schedules.items():
            summary[f'plant_{plant_id}_hours_on'] = sum(schedule.on_off_status)
            summary[f'plant_{plant_id}_total_mwh'] = schedule.total_energy
            summary[f'plant_{plant_id}_capacity_factor'] = schedule.capacity_factor
        
        return summary
    
    def plot_schedule(self, figsize: tuple = (12, 8)) -> None:
        """Plot the generation schedule."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Get schedule data
        df = self.to_dataframe()
        
        # Plot 1: Stacked generation by plant
        pivot_power = df.pivot_table(
            values='power_mw', 
            index='hour', 
            columns='plant_name',
            fill_value=0
        )
        
        pivot_power.plot(kind='area', stacked=True, ax=ax1, alpha=0.7)
        
        # Add demand line
        demand_data = self.demand_forecast.set_index('hour')
        ax1.plot(demand_data.index, demand_data['demand_median'], 
                'k--', linewidth=2, label='Demand (Median)')
        
        if 'demand_upper' in demand_data.columns:
            ax1.fill_between(demand_data.index,
                           demand_data['demand_lower'],
                           demand_data['demand_upper'],
                           alpha=0.2, color='gray',
                           label='Demand Uncertainty')
        
        ax1.set_ylabel('Power (MW)')
        ax1.set_title('Power Generation Schedule')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hourly costs
        pivot_cost = df.pivot_table(
            values='cost_eur', 
            index='hour', 
            columns='plant_name',
            fill_value=0
        )
        
        pivot_cost.plot(kind='bar', stacked=True, ax=ax2, alpha=0.7)
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Cost (€)')
        ax2.set_title('Hourly Generation Costs')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_to_excel(self, filename: str) -> None:
        """Export results to Excel file."""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([self.get_summary()]).T
            summary_df.columns = ['Value']
            summary_df.to_excel(writer, sheet_name='Summary')
            
            # Full schedule
            self.to_dataframe().to_excel(writer, sheet_name='Full_Schedule', index=False)
            
            # Individual plant schedules
            for plant_id, schedule in self.plant_schedules.items():
                schedule.to_dataframe().to_excel(
                    writer, 
                    sheet_name=f'Plant_{plant_id}',
                    index=False
                )
            
            # Demand forecast
            self.demand_forecast.to_excel(writer, sheet_name='Demand_Forecast', index=False)
            
            # Scenario results if available
            if self.scenario_results:
                pd.DataFrame(self.scenario_results).to_excel(
                    writer, 
                    sheet_name='Scenarios',
                    index=False
                )