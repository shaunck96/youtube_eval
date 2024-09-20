# Electricity Procurement Strategy Optimization Process: An In-Depth Report

## 1. Introduction

This report details the optimization process for electricity procurement strategies using a genetic algorithm approach. The goal is to find optimal Long-term Fixed-Price Contract (LFFC) strategies that minimize both the average price and the Conditional Value at Risk (CVaR) of electricity procurement.

## 2. Problem Formulation

The optimization problem can be formulated as follows:

Minimize:
1. Average Electricity Price
2. Conditional Value at Risk (CVaR)

Subject to constraints on:
- Number of strips (4 ≤ strips ≤ 6)
- Strip duration (3 ≤ duration ≤ 24 months)
- Lead time (0 ≤ lead time ≤ 18 months)
- Number of blocks per strip (2 ≤ blocks ≤ 5)
- Block size (5% ≤ size ≤ 40%)
- Start month (1 ≤ month ≤ 12)

## 3. Genetic Algorithm Components

### 3.1 Individual Representation

Each individual in the genetic algorithm represents a complete LFFC strategy. The individual is encoded as a list of floating-point numbers:

```
[num_strips, strip1_duration, strip1_lead_time, strip1_num_blocks, strip1_block_size, strip1_start_month, 
 strip2_duration, strip2_lead_time, strip2_num_blocks, strip2_block_size, strip2_start_month, ...]
```

### 3.2 Fitness Function

The fitness function evaluates each individual based on two objectives:

1. Average Electricity Price
2. Conditional Value at Risk (CVaR)

These objectives are calculated using the `ElectricityProcurementSimulator` class.

### 3.3 Genetic Operators

#### 3.3.1 Selection

The NSGA-II selection method is used, which is designed for multi-objective optimization problems.

#### 3.3.2 Crossover

Two-point crossover is employed to create offspring from parent individuals.

#### 3.3.3 Mutation

A custom mutation operator is implemented that can:
- Change the number of strips
- Modify individual strip parameters

## 4. Optimization Process

### 4.1 Initialization

1. Create an initial population of individuals, each representing a random LFFC strategy.
2. Evaluate the fitness of each individual using the `evaluate` function.

### 4.2 Evolution

For a specified number of generations:

1. Select parents using NSGA-II selection.
2. Create offspring through crossover and mutation.
3. Evaluate the fitness of offspring.
4. Combine parents and offspring.
5. Select the next generation using NSGA-II selection.

### 4.3 Multi-Run Optimization

To improve robustness, the optimization process is repeated multiple times:

1. Run the genetic algorithm optimization `num_runs` times (default: 5).
2. Collect all non-dominated solutions from each run.

### 4.4 Risk Score Calculation

For each strategy, calculate a risk score:

```
risk_score = 0.6 * normalized_avg_price + 0.4 * normalized_cvar
```

Where:
- `normalized_avg_price = (avg_price - min_price) / (max_price - min_price)`
- `normalized_cvar = (cvar - min_cvar) / (max_cvar - min_cvar)`

## 5. Key Functions

### 5.1 Create Individual

```python
def create_individual(toolbox: base.Toolbox, max_strips: int) -> List[float]:
    # Create a random individual with 4 to max_strips strips
    # Each strip has 5 parameters: duration, lead time, num_blocks, block_size, start_month
    # Normalize block sizes to sum to 100%
```

### 5.2 Evaluate

```python
def evaluate(individual: List[float], config: Dict[str, Any]) -> tuple:
    # Convert individual to LFFC strategy
    # Run simulation using ElectricityProcurementSimulator
    # Return (avg_price, cvar)
```

### 5.3 Custom Mutate

```python
def custom_mutate(individual: List[float], indpb: float, toolbox: base.Toolbox, max_strips: int) -> tuple:
    # Potentially change number of strips
    # Mutate individual parameters with probability indpb
    # Repair individual to ensure constraints are met
```

### 5.4 Calculate CVaR

```python
def calculate_cvar(prices: List[float], alpha: float = 0.05) -> float:
    # Sort prices
    # Calculate Value at Risk (VaR) at alpha percentile
    # Calculate mean of prices above VaR
```

## 6. Visualization

Two main visualizations are generated:

1. LFFC Strategies Comparison
   - Scatter plot showing start month, strategy number, block size, and duration for each strip

2. Risk vs. Average Price
   - Scatter plot of average price vs. CVaR for each strategy
   - Color-coded by risk score

## 7. Output

The optimization process produces:

1. A JSON file containing the top 10 strategies, including:
   - Strategy details (strips, block sizes, durations, etc.)
   - Average price
   - CVaR
   - Risk score

2. Visualization images saved in the specified output directory

## 8. Conclusion

This optimization process uses a genetic algorithm approach to find Pareto-optimal LFFC strategies that balance the trade-off between average electricity price and risk (CVaR). The multi-run approach and risk score calculation help to identify robust strategies that perform well across multiple objectives.
