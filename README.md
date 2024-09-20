# Mathematical Formulation of Electricity Procurement Strategy Optimization

## Objective Functions

Minimize:

1. Average Electricity Price: $f_1(\mathbf{x})$
2. Conditional Value at Risk (CVaR): $f_2(\mathbf{x})$

Where $\mathbf{x}$ represents the decision variables of the LFFC strategy.

## Decision Variables

Let $\mathbf{x} = [x_1, x_2, \ldots, x_{5n+1}]$, where:

- $x_1$: Number of strips $(n)$
- For each strip $i$ (where $i = 1, 2, \ldots, n$):
  - $x_{5i-3}$: Strip duration (months)
  - $x_{5i-2}$: Lead time (months)
  - $x_{5i-1}$: Number of blocks
  - $x_{5i}$: Block size (%)
  - $x_{5i+1}$: Start month

## Constraints

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

## Evaluation Functions

1. Average Electricity Price:
   $f_1(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^T P_t(\mathbf{x})$

   Where $T$ is the total number of time periods, and $P_t(\mathbf{x})$ is the electricity price at time $t$ given strategy $\mathbf{x}$.

2. Conditional Value at Risk (CVaR):
  $f_2(\mathbf{x}) = \text{CVaR}\alpha(P(\mathbf{x})) = \mathbb{E}[P(\mathbf{x}) \mid P(\mathbf{x}) \geq \text{VaR}\alpha(P(\mathbf{x}))]$
  Where $\alpha$ is the confidence level (typically 0.95 or 0.99), $P(\mathbf{x})$ is the distribution of electricity prices given strategy $\mathbf{x}$, and $\text{VaR}_\alpha$ is the Value at Risk at confidence level $\alpha$.

   Where $\alpha$ is the confidence level (typically 0.95 or 0.99), $P(\mathbf{x})$ is the distribution of electricity prices given strategy $\mathbf{x}$, and $\text{VaR}_\alpha$ is the Value at Risk at confidence level $\alpha$.

## Risk Score

The risk score for a strategy $\mathbf{x}$ is calculated as:

$R(\mathbf{x}) = 0.6 \cdot \frac{f_1(\mathbf{x}) - \min_{\mathbf{x}} f_1(\mathbf{x})}{\max_{\mathbf{x}} f_1(\mathbf{x}) - \min_{\mathbf{x}} f_1(\mathbf{x})} + 0.4 \cdot \frac{f_2(\mathbf{x}) - \min_{\mathbf{x}} f_2(\mathbf{x})}{\max_{\mathbf{x}} f_2(\mathbf{x}) - \min_{\mathbf{x}} f_2(\mathbf{x})}$

Where $\min_{\mathbf{x}}$ and $\max_{\mathbf{x}}$ are taken over all strategies in the population.

## Optimization Problem

The multi-objective optimization problem can be formally stated as:

$\min_{\mathbf{x}} \{f_1(\mathbf{x}), f_2(\mathbf{x})\}$

subject to the constraints listed above.

This problem is solved using a genetic algorithm approach, specifically the NSGA-II algorithm, which aims to find a set of Pareto-optimal solutions that represent the best trade-offs between the two objective functions.
