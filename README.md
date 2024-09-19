# LFFC Strategy Optimization Report with Corrected Mathematical Formulation

## 9. Mathematical Formulation

### 9.1 Problem Formulation

The LFFC strategy optimization can be formulated as a multi-objective optimization problem:

$$
\begin{align*}
\text{minimize} \quad & F(\mathbf{x}) = (f_1(\mathbf{x}), f_2(\mathbf{x})) \\
\text{subject to} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \\
& \mathbf{x} \in \mathbb{R}^n
\end{align*}
$$

Where:
- $\mathbf{x}$ is the decision vector representing an LFFC strategy
- $f_1(\mathbf{x})$ is the average electricity price
- $f_2(\mathbf{x})$ is the Conditional Value at Risk (CVaR)
- $g_i(\mathbf{x})$ are constraint functions

### 9.2 Decision Vector

The decision vector $\mathbf{x}$ is structured as follows:

$$
\mathbf{x} = (N_s, d_1, l_1, b_1, s_1, m_1, \ldots, d_{N_s}, l_{N_s}, b_{N_s}, s_{N_s}, m_{N_s})
$$

Where:
- $N_s$ is the number of strips
- For each strip $i$:
  - $d_i$ is the duration (months)
  - $l_i$ is the lead time (months)
  - $b_i$ is the number of blocks
  - $s_i$ is the block size (percentage)
  - $m_i$ is the start month

### 9.3 Objective Functions

1. Average Price:

   $$f_1(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^T P_t(\mathbf{x})$$

   where $T$ is the total number of months in the simulation, and $P_t(\mathbf{x})$ is the price in month $t$ given strategy $\mathbf{x}$.

2. Conditional Value at Risk (CVaR):
$$f_2(\mathbf{x}) = \text{CVaR}\alpha(P(\mathbf{x})) = \mathbb{E}[P(\mathbf{x}) | P(\mathbf{x}) \geq \text{VaR}\alpha(P(\mathbf{x}))]$$
where $\alpha$ is the confidence level (e.g., 95%), and $\text{VaR}_\alpha$ is the Value at Risk.

### 9.4 Constraints

1. Total block size:

   $$g_1(\mathbf{x}) = |\sum_{i=1}^{N_s} s_i b_i - 100| - 5 \leq 0$$

2. Minimum block size:

   $$g_2(\mathbf{x}) = 5 - \min_{i} s_i \leq 0$$

3. Maximum strip duration:

   $$g_3(\mathbf{x}) = \max_{i} d_i - 24 \leq 0$$

4. Maximum lead time:

   $$g_4(\mathbf{x}) = \max_{i} l_i - 18 \leq 0$$

### 9.5 Genetic Algorithm Operators

Crossover (Two-point crossover):
For parents $\mathbf{p}_1$ and $\mathbf{p}_2$, and randomly chosen crossover points $c_1 < c_2$:
$$\mathbf{o}_1 = (\mathbf{p}_1[1:c_1], \mathbf{p}_2[c_1:c_2], \mathbf{p}_1[c_2:])$$
$$\mathbf{o}_2 = (\mathbf{p}_2[1:c_1], \mathbf{p}_1[c_1:c_2], \mathbf{p}_2[c_2:])$$
Mutation:
For each gene $x_i$ in individual $\mathbf{x}$:
$$
x_i' =
\begin{cases}
x_i + \mathcal{N}(0, \sigma_i), & \text{with probability } p_m \
x_i, & \text{otherwise}
\end{cases}
$$
where $\sigma_i$ is the step size for the $i$-th parameter, and $p_m$ is the mutation probability.

### 9.6 NSGA-II Selection

The NSGA-II algorithm uses non-dominated sorting and crowding distance to select individuals:

1. Non-dominated sorting: Partition the population into fronts $F_1, F_2, \ldots$ where individuals in $F_i$ are not dominated by any individual in $F_j$ for $j > i$.

2. Crowding distance: For individuals $i$ in the same front, calculate:

   $$d_i = \sum_{k=1}^2 \frac{f_k(i+1) - f_k(i-1)}{f_k^{max} - f_k^{min}}$$

   where $f_k(i)$ is the $k$-th objective value of the $i$-th individual.

3. Selection: Choose individuals based on front ranking and crowding distance.

### 9.7 Risk Score Calculation

For a given strategy $\mathbf{x}$, the risk score is calculated as:

$$
\text{RiskScore}(\mathbf{x}) = w_1 \cdot \frac{f_1(\mathbf{x}) - f_1^{min}}{f_1^{max} - f_1^{min}} + w_2 \cdot \frac{f_2(\mathbf{x}) - f_2^{min}}{f_2^{max} - f_2^{min}}
$$

where $w_1$ and $w_2$ are weights (e.g., $w_1 = 0.6$, $w_2 = 0.4$), and $f_k^{min}$ and $f_k^{max}$ are the minimum and maximum values of objective $k$ across all strategies.

## 10. Optimization Algorithm

The optimization process can be summarized in the following steps:

1. Initialize population $P_0$ of size $N$
2. For generation $t = 1$ to $T$:
   a. Create offspring population $Q_t$ through crossover and mutation
   b. Combine parent and offspring: $R_t = P_t \cup Q_t$
   c. Perform non-dominated sorting on $R_t$ to get fronts $F_1, F_2, \ldots$
   d. Select new population $P_{t+1}$:
      - Add fronts $F_1, F_2, \ldots$ until $|P_{t+1}| + |F_i| > N$
      - Fill remaining slots using crowding distance selection from $F_i$
3. Return non-dominated solutions from final population $P_T$

This mathematical formulation provides a rigorous basis for understanding the LFFC strategy optimization process. It defines the problem structure, objective functions, constraints, and key algorithms used in the genetic algorithm approach.
