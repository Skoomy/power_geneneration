"""Power generation optimization using Pyomo with demand uncertainty."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import time

from .models import OptimizationResult, PlantSchedule


class PowerOptimizer:
    """Optimizer for power generation scheduling with demand uncertainty."""
    
    def __init__(self,
                 plants: Dict[int, Dict],
                 solver_name: str = 'glpk',
                 solver_options: Optional[Dict] = None):
        """
        Initialize power optimizer.
        
        Args:
            plants: Dictionary of plant configurations
                {plant_id: {'name': str, 'cost': float, 'min_capacity': float, 
                           'max_capacity': float}}
            solver_name: Solver to use ('glpk', 'gurobi', 'cplex', 'cbc')
            solver_options: Options to pass to solver
        """
        self.plants = plants
        self.num_plants = len(plants)
        self.plant_ids = list(plants.keys())
        
        # Initialize solver
        self.solver = SolverFactory(solver_name)
        if solver_options:
            for key, value in solver_options.items():
                self.solver.options[key] = value
    
    def _create_base_model(self, time_periods: List[int]) -> pyo.ConcreteModel:
        """Create base Pyomo model structure."""
        model = pyo.ConcreteModel()
        
        # Sets
        model.PLANTS = pyo.Set(initialize=self.plant_ids)
        model.TIME = pyo.Set(initialize=time_periods)
        
        # Parameters
        model.cost = pyo.Param(model.PLANTS, initialize={
            i: self.plants[i]['cost'] for i in self.plant_ids
        })
        
        model.min_capacity = pyo.Param(model.PLANTS, initialize={
            i: self.plants[i]['min_capacity'] for i in self.plant_ids
        })
        
        model.max_capacity = pyo.Param(model.PLANTS, initialize={
            i: self.plants[i]['max_capacity'] for i in self.plant_ids
        })
        
        # Variables
        model.x = pyo.Var(model.PLANTS, model.TIME, 
                         domain=pyo.NonNegativeReals,
                         bounds=lambda m, i, t: (0, m.max_capacity[i]))
        
        model.y = pyo.Var(model.PLANTS, model.TIME, 
                         domain=pyo.Binary)
        
        return model
    
    def optimize_deterministic(self, 
                             demand: Union[List[float], np.ndarray],
                             time_periods: Optional[List[int]] = None) -> OptimizationResult:
        """
        Solve deterministic optimization problem.
        
        Args:
            demand: Demand for each time period (MW)
            time_periods: Time period indices (default: range(len(demand)))
            
        Returns:
            OptimizationResult with optimal schedule
        """
        if time_periods is None:
            time_periods = list(range(len(demand)))
        
        start_time = time.time()
        
        # Create model
        model = self._create_base_model(time_periods)
        
        # Demand parameter
        demand_dict = {t: demand[t] for t in time_periods}
        model.demand = pyo.Param(model.TIME, initialize=demand_dict)
        
        # Objective: Minimize total cost
        def obj_rule(m):
            return sum(m.cost[i] * m.x[i, t] 
                      for i in m.PLANTS for t in m.TIME)
        
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        # Constraints
        # Demand satisfaction
        def demand_rule(m, t):
            return sum(m.x[i, t] for i in m.PLANTS) >= m.demand[t]
        
        model.demand_constraint = pyo.Constraint(model.TIME, rule=demand_rule)
        
        # Minimum production when on
        def min_production_rule(m, i, t):
            return m.x[i, t] >= m.min_capacity[i] * m.y[i, t]
        
        model.min_production = pyo.Constraint(
            model.PLANTS, model.TIME, rule=min_production_rule
        )
        
        # Maximum capacity when on
        def max_production_rule(m, i, t):
            return m.x[i, t] <= m.max_capacity[i] * m.y[i, t]
        
        model.max_production = pyo.Constraint(
            model.PLANTS, model.TIME, rule=max_production_rule
        )
        
        # Solve
        results = self.solver.solve(model, tee=False)
        solve_time = time.time() - start_time
        
        # Extract results
        return self._extract_results(model, results, time_periods, demand, solve_time)
    
    def optimize_robust(self,
                       demand_forecast: pd.DataFrame,
                       robustness_parameter: float = 0.2,
                       time_periods: Optional[List[int]] = None) -> OptimizationResult:
        """
        Solve robust optimization considering demand uncertainty.
        
        Uses the Bertsimas-Sim robust optimization approach.
        
        Args:
            demand_forecast: DataFrame with columns:
                - demand_median: Point forecast
                - demand_lower: Lower bound
                - demand_upper: Upper bound
            robustness_parameter: Gamma parameter (0 to 1) controlling conservatism
            time_periods: Time period indices
            
        Returns:
            OptimizationResult with robust schedule
        """
        if time_periods is None:
            time_periods = list(range(len(demand_forecast)))
        
        start_time = time.time()
        
        # Extract demand data
        demand_nominal = demand_forecast['demand_median'].values
        demand_deviation = (demand_forecast['demand_upper'].values - 
                          demand_forecast['demand_median'].values)
        
        # Create model
        model = self._create_base_model(time_periods)
        
        # Demand parameters
        model.demand_nominal = pyo.Param(model.TIME, initialize={
            t: demand_nominal[t] for t in time_periods
        })
        
        model.demand_deviation = pyo.Param(model.TIME, initialize={
            t: demand_deviation[t] for t in time_periods
        })
        
        # Robustness budget (number of periods that can deviate)
        model.gamma = pyo.Param(initialize=robustness_parameter * len(time_periods))
        
        # Dual variables for robust counterpart
        model.p = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.q = pyo.Var(domain=pyo.NonNegativeReals)
        
        # Objective: Minimize cost
        def obj_rule(m):
            return sum(m.cost[i] * m.x[i, t] 
                      for i in m.PLANTS for t in m.TIME)
        
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        # Robust demand constraint
        def robust_demand_rule(m, t):
            return (sum(m.x[i, t] for i in m.PLANTS) >= 
                   m.demand_nominal[t] + 
                   m.demand_deviation[t] * m.p[t])
        
        model.robust_demand = pyo.Constraint(model.TIME, rule=robust_demand_rule)
        
        # Robustness budget constraint
        def budget_rule(m):
            return sum(m.p[t] for t in m.TIME) <= m.gamma
        
        model.budget = pyo.Constraint(rule=budget_rule)
        
        # Dual constraint
        def dual_rule(m, t):
            return m.p[t] <= 1
        
        model.dual_constraint = pyo.Constraint(model.TIME, rule=dual_rule)
        
        # Plant operation constraints
        def min_production_rule(m, i, t):
            return m.x[i, t] >= m.min_capacity[i] * m.y[i, t]
        
        model.min_production = pyo.Constraint(
            model.PLANTS, model.TIME, rule=min_production_rule
        )
        
        def max_production_rule(m, i, t):
            return m.x[i, t] <= m.max_capacity[i] * m.y[i, t]
        
        model.max_production = pyo.Constraint(
            model.PLANTS, model.TIME, rule=max_production_rule
        )
        
        # Solve
        results = self.solver.solve(model, tee=False)
        solve_time = time.time() - start_time
        
        # Extract results
        result = self._extract_results(
            model, results, time_periods, demand_nominal, solve_time
        )
        
        # Add robust optimization info
        result.confidence_level = 1 - robustness_parameter
        
        # Calculate worst-case cost
        worst_case_demand = demand_nominal + demand_deviation * np.array([
            pyo.value(model.p[t]) for t in time_periods
        ])
        
        result.worst_case_cost = sum(
            self.plants[i]['cost'] * pyo.value(model.x[i, t])
            for i in self.plant_ids for t in time_periods
        )
        
        return result
    
    def optimize_stochastic(self,
                          demand_forecast: pd.DataFrame,
                          num_scenarios: int = 100,
                          time_periods: Optional[List[int]] = None,
                          use_progressive_hedging: bool = False) -> OptimizationResult:
        """
        Solve stochastic optimization with multiple demand scenarios.
        
        Args:
            demand_forecast: DataFrame with demand distribution parameters
            num_scenarios: Number of scenarios to generate
            time_periods: Time period indices
            use_progressive_hedging: Use decomposition for large problems
            
        Returns:
            OptimizationResult with expected value solution
        """
        if time_periods is None:
            time_periods = list(range(len(demand_forecast)))
        
        start_time = time.time()
        
        # Generate scenarios
        scenarios = self._generate_demand_scenarios(demand_forecast, num_scenarios)
        scenario_probs = [1.0 / num_scenarios] * num_scenarios
        
        # Create extensive form model
        model = pyo.ConcreteModel()
        
        # Sets
        model.PLANTS = pyo.Set(initialize=self.plant_ids)
        model.TIME = pyo.Set(initialize=time_periods)
        model.SCENARIOS = pyo.Set(initialize=range(num_scenarios))
        
        # Parameters
        model.cost = pyo.Param(model.PLANTS, initialize={
            i: self.plants[i]['cost'] for i in self.plant_ids
        })
        
        model.min_capacity = pyo.Param(model.PLANTS, initialize={
            i: self.plants[i]['min_capacity'] for i in self.plant_ids
        })
        
        model.max_capacity = pyo.Param(model.PLANTS, initialize={
            i: self.plants[i]['max_capacity'] for i in self.plant_ids
        })
        
        model.prob = pyo.Param(model.SCENARIOS, initialize={
            s: scenario_probs[s] for s in range(num_scenarios)
        })
        
        # Scenario demands
        demand_init = {}
        for s in range(num_scenarios):
            for t in time_periods:
                demand_init[s, t] = scenarios[s][t]
        
        model.demand = pyo.Param(model.SCENARIOS, model.TIME, initialize=demand_init)
        
        # First-stage variables (plant commitment)
        model.y = pyo.Var(model.PLANTS, model.TIME, domain=pyo.Binary)
        
        # Second-stage variables (power output per scenario)
        model.x = pyo.Var(model.PLANTS, model.TIME, model.SCENARIOS,
                         domain=pyo.NonNegativeReals)
        
        # Objective: Minimize expected cost
        def obj_rule(m):
            return sum(m.prob[s] * m.cost[i] * m.x[i, t, s]
                      for i in m.PLANTS 
                      for t in m.TIME 
                      for s in m.SCENARIOS)
        
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        # Constraints
        # Demand satisfaction per scenario
        def demand_rule(m, t, s):
            return sum(m.x[i, t, s] for i in m.PLANTS) >= m.demand[s, t]
        
        model.demand_constraint = pyo.Constraint(
            model.TIME, model.SCENARIOS, rule=demand_rule
        )
        
        # Non-anticipativity: commitment decisions same across scenarios
        def min_production_rule(m, i, t, s):
            return m.x[i, t, s] >= m.min_capacity[i] * m.y[i, t]
        
        model.min_production = pyo.Constraint(
            model.PLANTS, model.TIME, model.SCENARIOS, rule=min_production_rule
        )
        
        def max_production_rule(m, i, t, s):
            return m.x[i, t, s] <= m.max_capacity[i] * m.y[i, t]
        
        model.max_production = pyo.Constraint(
            model.PLANTS, model.TIME, model.SCENARIOS, rule=max_production_rule
        )
        
        # Solve
        if use_progressive_hedging and num_scenarios > 50:
            # For large problems, could implement progressive hedging
            # For now, solve extensive form
            pass
        
        results = self.solver.solve(model, tee=False)
        solve_time = time.time() - start_time
        
        # Extract expected value solution
        result = self._extract_stochastic_results(
            model, results, time_periods, scenarios, solve_time
        )
        
        return result
    
    def _generate_demand_scenarios(self, 
                                 demand_forecast: pd.DataFrame,
                                 num_scenarios: int) -> np.ndarray:
        """Generate demand scenarios from forecast distribution."""
        periods = len(demand_forecast)
        scenarios = np.zeros((num_scenarios, periods))
        
        for t in range(periods):
            lower = demand_forecast.iloc[t]['demand_lower']
            upper = demand_forecast.iloc[t]['demand_upper']
            median = demand_forecast.iloc[t]['demand_median']
            
            # Use beta distribution to respect bounds
            # Convert to alpha, beta parameters
            mean = median
            std = (upper - lower) / 4  # Rough approximation
            
            # Sample from truncated normal
            samples = np.random.normal(mean, std, num_scenarios)
            scenarios[:, t] = np.clip(samples, lower, upper)
        
        return scenarios
    
    def _extract_results(self, model: pyo.ConcreteModel, 
                        solver_results,
                        time_periods: List[int], 
                        demand: np.ndarray,
                        solve_time: float) -> OptimizationResult:
        """Extract results from solved model."""
        # Check solver status
        if (solver_results.solver.status != SolverStatus.ok or
            solver_results.solver.termination_condition != TerminationCondition.optimal):
            status = "infeasible"
        else:
            status = "optimal"
        
        # Extract solution
        plant_schedules = {}
        total_cost = 0
        total_generation = 0
        
        for plant_id in self.plant_ids:
            hours = []
            on_off_status = []
            power_output = []
            operating_cost = []
            
            for t in time_periods:
                hours.append(t)
                
                y_val = pyo.value(model.y[plant_id, t])
                x_val = pyo.value(model.x[plant_id, t])
                
                on_off = int(round(y_val)) if y_val is not None else 0
                power = x_val if x_val is not None else 0.0
                cost = power * self.plants[plant_id]['cost']
                
                on_off_status.append(on_off)
                power_output.append(power)
                operating_cost.append(cost)
                
                total_cost += cost
                total_generation += power
            
            plant_schedules[plant_id] = PlantSchedule(
                plant_id=plant_id,
                plant_name=self.plants[plant_id]['name'],
                hours=hours,
                on_off_status=on_off_status,
                power_output=power_output,
                operating_cost=operating_cost
            )
        
        # Create demand forecast DataFrame
        demand_df = pd.DataFrame({
            'hour': time_periods,
            'demand_median': demand
        })
        
        return OptimizationResult(
            demand_forecast=demand_df,
            plant_schedules=plant_schedules,
            total_cost=total_cost,
            total_generation=total_generation,
            optimization_status=status,
            solve_time=solve_time
        )
    
    def _extract_stochastic_results(self, model: pyo.ConcreteModel,
                                   solver_results,
                                   time_periods: List[int],
                                   scenarios: np.ndarray,
                                   solve_time: float) -> OptimizationResult:
        """Extract results from stochastic optimization."""
        # Check solver status
        if (solver_results.solver.status != SolverStatus.ok or
            solver_results.solver.termination_condition != TerminationCondition.optimal):
            status = "infeasible"
        else:
            status = "optimal"
        
        # Calculate expected values
        plant_schedules = {}
        total_expected_cost = 0
        total_expected_generation = 0
        
        num_scenarios = len(scenarios)
        
        for plant_id in self.plant_ids:
            hours = []
            on_off_status = []
            power_output = []
            operating_cost = []
            
            for t in time_periods:
                hours.append(t)
                
                # Commitment decision (same across scenarios)
                y_val = pyo.value(model.y[plant_id, t])
                on_off = int(round(y_val)) if y_val is not None else 0
                
                # Expected power output
                expected_power = 0
                expected_cost = 0
                
                for s in range(num_scenarios):
                    x_val = pyo.value(model.x[plant_id, t, s])
                    power = x_val if x_val is not None else 0.0
                    expected_power += power / num_scenarios
                    expected_cost += power * self.plants[plant_id]['cost'] / num_scenarios
                
                on_off_status.append(on_off)
                power_output.append(expected_power)
                operating_cost.append(expected_cost)
                
                total_expected_cost += expected_cost
                total_expected_generation += expected_power
            
            plant_schedules[plant_id] = PlantSchedule(
                plant_id=plant_id,
                plant_name=self.plants[plant_id]['name'],
                hours=hours,
                on_off_status=on_off_status,
                power_output=power_output,
                operating_cost=operating_cost
            )
        
        # Calculate scenario costs
        scenario_costs = []
        for s in range(num_scenarios):
            s_cost = sum(
                pyo.value(model.x[i, t, s]) * self.plants[i]['cost']
                for i in self.plant_ids for t in time_periods
                if pyo.value(model.x[i, t, s]) is not None
            )
            scenario_costs.append(s_cost)
        
        # Create demand forecast DataFrame (using mean scenario demand)
        demand_df = pd.DataFrame({
            'hour': time_periods,
            'demand_median': np.mean(scenarios, axis=0),
            'demand_lower': np.percentile(scenarios, 10, axis=0),
            'demand_upper': np.percentile(scenarios, 90, axis=0)
        })
        
        result = OptimizationResult(
            demand_forecast=demand_df,
            plant_schedules=plant_schedules,
            total_cost=total_expected_cost,
            total_generation=total_expected_generation,
            optimization_status=status,
            solve_time=solve_time,
            expected_cost=total_expected_cost,
            worst_case_cost=max(scenario_costs) if scenario_costs else 0,
            scenario_results=[
                {'scenario': s, 'cost': c, 'demand_total': sum(scenarios[s])}
                for s, c in enumerate(scenario_costs)
            ]
        )
        
        return result