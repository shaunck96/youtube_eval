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

#### Crossover (Two-point crossover)
For parents \(\mathbf{p}_1\) and \(\mathbf{p}_2\), and randomly chosen crossover points \(c_1 < c_2\):
$$
\begin{align*}
\mathbf{o}_1 &= (\mathbf{p}_1[1:c_1], \mathbf{p}_2[c_1:c_2], \mathbf{p}_1[c_2:N]) \\
\mathbf{o}_2 &= (\mathbf{p}_2[1:c_1], \mathbf{p}_1[c_1:c_2], \mathbf{p}_2[c_2:N])
\end{align*}
$$

Where \(\mathbf{o}_1\) and \(\mathbf{o}_2\) are the offspring generated from parents \(\mathbf{p}_1\) and \(\mathbf{p}_2\).

#### Mutation
To introduce variability, a mutation operator can be applied to each offspring. For an offspring \(\mathbf{o}\):
- Select a random index \(j\) within \(\mathbf{o}\).
- Modify the value at index \(j\) based on predefined mutation rules (e.g., adjusting the duration, lead time, or block size by a small percentage).

### 9.6 Selection Process
A selection mechanism, such as tournament selection, can be employed to choose parents for crossover:
- Randomly select \(k\) individuals from the population.
- Choose the individual with the best fitness (i.e., lowest \(F(\mathbf{x})\)) among these.

### 9.7 Termination Criteria
The algorithm can be terminated based on one of the following criteria:
- A maximum number of generations is reached.
- A satisfactory fitness level is achieved.
- The population has converged (i.e., little to no improvement in fitness over several generations).

### 9.8 Performance Metrics
To evaluate the performance of the LFFC strategy optimization, the following metrics can be monitored:
1. Convergence rate of the objective functions \(f_1(\mathbf{x})\) and \(f_2(\mathbf{x})\).
2. Diversity of the population (to ensure exploration of the solution space).
3. The best found solution and its robustness against various scenarios.

### 9.9 Implementation
The optimization can be implemented using a programming language such as Python, leveraging libraries like DEAP (Distributed Evolutionary Algorithms in Python) or PyGMO (Python Global Multiobjective Optimization). 

### Conclusion
The outlined mathematical formulation and genetic algorithm framework provide a structured approach to optimizing the LFFC strategy, balancing cost efficiency and risk management. Future work could involve testing the framework under real-world conditions and refining the model based on empirical results.
