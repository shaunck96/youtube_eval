# Comprehensive Electricity Procurement Strategy Optimization Report

## 1. Problem Formulation

### 1.1 Objective Functions

Minimize:

1. Average Electricity Price: $f_1(\mathbf{x})$
2. Conditional Value at Risk (CVaR): $f_2(\mathbf{x})$

Where $\mathbf{x}$ represents the decision variables of the LFFC strategy.

### 1.2 Decision Variables

Let $\mathbf{x} = [x_1, x_2, \ldots, x_{5n+1}]$, where:

- $x_1$: Number of strips $(n)$
- For each strip $i$ (where $i = 1, 2, \ldots, n$):
  - $x_{5i-3}$: Strip duration (months)
  - $x_{5i-2}$: Lead time (months)
  - $x_{5i-1}$: Number of blocks
  - $x_{5i}$: Block size (%)
  - $x_{5i+1}$: Start month

### 1.3 Constraints

1. Number of strips:
   $4 \leq x_1 \leq 6$

2. For each strip $i$ (where $i = 1, 2, \ldots, n$):
   
   a. Strip duration:
      $3 \leq x_{5i-3} \leq 24$
   
   b. Lead time:
      $0 \leq x_{5i-2} \leq 18$
   
   c. Number of blocks:
      $2 \leq x_{5i-1} \leq 5$
   
   d. Block size:
      $5 \leq x_{5i} \leq 40$
   
   e. Start month:
      $1 \leq x_{5i+1} \leq 12$

3. Total block size:
   $\sum_{i=1}^n x_{5i} = 100$

### 1.4 Evaluation Functions

1. Average Electricity Price:
   $f_1(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^T P_t(\mathbf{x})$

   Where $T$ is the total number of time periods, and $P_t(\mathbf{x})$ is the electricity price at time $t$ given strategy $\mathbf{x}$.

2. Conditional Value at Risk (CVaR):
  $f_2(\mathbf{x}) = \text{CVaR}\alpha(P(\mathbf{x})) = \mathbb{E}[P(\mathbf{x}) \mid P(\mathbf{x}) \geq \text{VaR}\alpha(P(\mathbf{x}))]$
  Where $\alpha$ is the confidence level (typically 0.95 or 0.99), $P(\mathbf{x})$ is the distribution of electricity prices given strategy $\mathbf{x}$, and $\text{VaR}_\alpha$ is the Value at Risk at confidence level $\alpha$.

### 1.5 Risk Score

The risk score for a strategy $\mathbf{x}$ is calculated as:

$R(\mathbf{x}) = 0.6 \cdot \frac{f_1(\mathbf{x}) - \min_{\mathbf{x}} f_1(\mathbf{x})}{\max_{\mathbf{x}} f_1(\mathbf{x}) - \min_{\mathbf{x}} f_1(\mathbf{x})} + 0.4 \cdot \frac{f_2(\mathbf{x}) - \min_{\mathbf{x}} f_2(\mathbf{x})}{\max_{\mathbf{x}} f_2(\mathbf{x}) - \min_{\mathbf{x}} f_2(\mathbf{x})}$

Where $\min_{\mathbf{x}}$ and $\max_{\mathbf{x}}$ are taken over all strategies in the population.

### 1.6 Optimization Problem

The multi-objective optimization problem can be formally stated as:

$\min_{\mathbf{x}} \{f_1(\mathbf{x}), f_2(\mathbf{x})\}$

subject to the constraints listed above.

## 2. Implementation Details

### 2.1 Multiple Optimization Runs

To improve robustness, the optimization process is repeated multiple times:

$\text{Best Strategies} = \bigcup_{i=1}^{N} \text{ParetoFront}(\text{GeneticAlgorithm}_i(\mathbf{x}))$

Where $N$ is the number of optimization runs (default: 5), and $\text{ParetoFront}$ selects the non-dominated solutions from each run.

### 2.2 Genetic Algorithm Components

#### 2.2.1 Individual Representation

Each individual in the genetic algorithm represents a complete LFFC strategy, encoded as described in the Decision Variables section.

#### 2.2.2 Custom Mutation Operator

The custom mutation operator can:
1. Change the number of strips:
   $x_1' = x_1 + \Delta n$, where $\Delta n \in \{-1, 0, 1\}$
2. Modify individual strip parameters:
   $x_i' = x_i + \mathcal{N}(0, \sigma_i)$, where $\sigma_i$ is specific to each parameter type

#### 2.2.3 Repair Function

After mutation or crossover, a repair function is applied to ensure constraint satisfaction:

$\mathbf{x}_\text{repaired} = \text{Repair}(\mathbf{x})$

This function adjusts parameters to meet constraints, particularly normalizing block sizes to sum to 100%.

#### 2.2.4 Selection

The NSGA-II selection method is used for multi-objective optimization.

### 2.3 Simulation-Based Evaluation

The `ElectricityProcurementSimulator` class is used to evaluate strategies:

$[f_1(\mathbf{x}), f_2(\mathbf{x})] = \text{ElectricityProcurementSimulator}(\mathbf{x})$

### 2.4 Pareto Front Selection

After optimization, the Pareto front of non-dominated solutions is selected:

$\text{ParetoFront} = \{\mathbf{x} \mid \nexists \mathbf{y} : f_1(\mathbf{y}) \leq f_1(\mathbf{x}) \land f_2(\mathbf{y}) \leq f_2(\mathbf{x}) \land (f_1(\mathbf{y}) < f_1(\mathbf{x}) \lor f_2(\mathbf{y}) < f_2(\mathbf{x}))\}$

## 3. Post-Optimization Analysis

### 3.1 Risk Score Calculation

Risk scores are calculated for each strategy in the Pareto front using the formula in section 1.5.

### 3.2 Visualization

Two main visualizations are generated:

1. LFFC Strategies Comparison: A scatter plot showing the characteristics of each strip in the top strategies.
2. Risk vs. Average Price: A scatter plot of average price vs. CVaR for each strategy, color-coded by risk score.

## 4. Conclusion

This optimization approach combines a formal multi-objective optimization problem with practical implementation considerations. The genetic algorithm, enhanced with custom operators and multiple runs, aims to find robust Pareto-optimal solutions. The simulation-based evaluation and post-optimization analysis provide insights beyond what's captured in the mathematical formulation alone.
