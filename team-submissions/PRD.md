# Product Requirements Document

**Project Name:** Ryzzz Gate  
**Team Name:** Planck Scale  
**GitHub Repository:** [https://github.com/Meeeee6623/2026-NVIDIA](https://github.com/Meeeee6623/2026-NVIDIA)

---

## 1. Team Roles & Responsibilities

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Benjamin Chauhan | [@Meeeee6623](https://github.com/Meeeee6623) | @vivacious_cheetah_00727 |
| **GPU Acceleration PIC** (Builder) | Trent Seaman | [@Tridentinus](https://github.com/Tridentinus) | @_chevaliermalfet |
| **Quality Assurance PIC** (Verifier) | Martin Castellanos-Cubides | [@martcc](https://github.com/martcc) | @bonniegamerfreddyplaysforever |
| **Quality Assurance PIC** (Verifier) | Sanjeev Chauhan | [@sanjeev-one](https://github.com/sanjeev-one) | @exquisite_dove_79161 |
| **Technical Marketing PIC** (Storyteller) | Joseph Telaak | [@The1TrueJoe](https://github.com/The1TrueJoe) | @jtela |

---

## 2. The Architecture

### Choice of Quantum Algorithm
* **Algorithm:** Digital-Counterdiabatic Quantum Optimization (DCQO) with hybrid impulse-to-adiabatic transition and variable λ schedule
    * We implement a modified DCQO approach that uses an impulse region during the initial optimization phase, then transitions back to adiabatic evolution near the end of the protocol.
    * The counterdiabatic strength parameter λ is varied throughout the optimization, providing dynamic control over the balance between adiabatic and diabatic evolution.

* **Motivation:** This hybrid approach aims to combine the benefits of both strategies:
    * *Impulse region:* Provides rapid exploration of the solution space early in the optimization, potentially escaping local minima more effectively than pure adiabatic methods.
    * *Adiabatic transition:* Returns to slower, more controlled evolution at the end to ensure convergence to high-quality solutions and maintain quantum coherence during the critical final stages.
    * The strategy is particularly suited for LABS optimization where initial exploration is crucial, but final convergence requires careful tuning.

### Literature Review

* **Reference 1: GPU-Accelerated Memetic Tabu Search (MTS)** * **Citation:** Zhang, Z., et al. (2025). "New Improvements in Solving Large LABS Instances Using Massively Parallelizable Memetic Tabu Search." [https://arxiv.org/html/2504.00987v2](https://arxiv.org/html/2504.00987v2)  
  * **Relevance:** Establishes the classical benchmark for our hybrid approach. It demonstrates state-of-the-art 26x GPU speedups and provides the parallelization strategies (block-level and thread-level) that inform our classical CUDA implementation.

* **Reference 2: Quantum Coherence & The Impulse Regime** * **Citation:** Huerta-Ruiz, R., et al. (2025). "Quantum coherence and counterdiabatic quantum computing." [https://arxiv.org/html/2504.17642v1](https://arxiv.org/html/2504.17642v1)  
  * **Relevance:** Provides the theoretical foundation for our "Impulse-to-Adiabatic" transition. It defines the condition where counterdiabatic (CD) terms dominate, allowing for faster evolution and high coherence, which justifies our variable λ schedule.

* **Reference 3: Digitized Counterdiabatic Quantum Optimization (DCQO)** * **Citation:** Hegade, N. N., et al. (2022). "Digitized counterdiabatic quantum optimization." [https://arxiv.org/html/2210.15962v2](https://arxiv.org/html/2210.15962v2) and [https://arxiv.org/html/2308.02342v2](https://arxiv.org/html/2308.02342v2).  
  * **Relevance:** Establishes the framework for implementing CD driving on gate-based quantum computers using nested commutator expansions for the adiabatic gauge potential (AGP).

* **Reference 4: Multi-GPU Scaling for LABS** * **Citation:** "Massively parallel solvers for the LABS problem." Journal of Parallel and Distributed Computing (2024). [https://www.sciencedirect.com/science/article/abs/pii/S074373152400176X](https://www.sciencedirect.com/science/article/abs/pii/S074373152400176X)  
  * **Relevance:** Provides the architectural roadmap for scaling our MTS implementation across 8x A100 GPUs, specifically addressing the $O(N^2)$ energy evaluation time complexity.

* **Reference 5: The Bernasconi Model & Ground Truth** * **Citation:** Packebusch, T., & Mertens, S. (2016). "Low Autocorrelation Binary Sequences." [https://arxiv.org/pdf/1512.02475](https://arxiv.org/pdf/1512.02475)  
  * **Relevance:** Our primary verification source. It provides the optimal energy tables for $N \le 66$ and establishes the ground truth used to validate our GPU kernels and quantum-seeded results.

* **Reference 6: Adaptive Quantum Simulated Annealing** * **Citation:** Harrow, A. W., & Wei, A. Y. (2020). "Adaptive Quantum Simulated Annealing." [https://arxiv.org/html/1907.09965v2](https://arxiv.org/html/1907.09965v2)  
  * **Relevance:** Informs our seeding strategy. We use adaptive schedules derived from this work to ensure "qsamples" effectively represent the lowest-energy states of the LABS landscape.

---

## 3. The Acceleration Strategy

### Quantum Acceleration (CUDA-Q)
* **Strategy:** We will initially test our DCQO circuits on a single NVIDIA A100 (80GB) to establish a performance baseline. To push the limits of the problem size $N$, we will scale up to **8 A100s** using the `nvidia-mgpu` backend, enabling shared memory across GPUs to accommodate the exponential growth of the state vector and determine the maximum attainable $N$.
* **Memory Optimization (cuTensorNet):** If state-vector simulations encounter memory limits even with multi-GPU distribution, we will pivot to the **CUDA-Q TensorNet backend (`nvidia-cu-tensornet`)**. This approach uses tensor network contraction (via NVIDIA's cuTensorNet library) to simulate larger circuits by avoiding the explicit $2^N$ storage of the state vector, leveraging the massive parallel throughput of multiple GPUs for efficient contraction paths.
* **Batch Sampling:** We will use `cudaq.sample` with a high shot count to generate a statistically significant "Quantum Seed" population, which will be transferred directly to GPU global memory for the MTS phase to minimize PCIe overhead.

### Classical Acceleration (MTS)
* **Strategy:** Our strategy shifts from sequential neighbor evaluation to a massively parallel population-based approach using native CUDA C++.
* **Parallel Energy Evaluation:** Instead of evaluating one neighbor flip at a time, we use a custom CUDA kernel to evaluate the entire population of 8,192 sequences simultaneously.
* **Warp-Level Optimization:** We implement warp-level primitives (`__shfl_down_sync`) to perform rapid reductions for finding the minimum energy move within each Tabu search thread block.
* **Shared Memory Utilization:** To maximize throughput on A100 hardware, we load the sequence and correlation vectors into Shared Memory (L1 cache), reducing global memory latency during the frequent energy-update cycles of the Tabu search.
* **Asynchronous Execution:** We will use CUDA streams to overlap the "Quantum Seeding" process with the initialization of the classical population, ensuring the GPU is never idle.

### Hardware Targets
* **Dev Environment:** **qBraid (CPU)** for initial logic validation of the $G_2$ and $G_4$ interaction sets; **Brev L4** for debugging CUDA kernels and verifying energy symmetry.
* **Production Environment:** **Brev A100-80GB** for final benchmarks, utilizing the high memory bandwidth to scale the MTS population and handle the large state vectors required for $N=40+$ quantum simulations.
---

### 4. The Verification Plan

### Unit Testing Strategy
* **Framework:** We utilize the standard Python `unittest` framework to implement a **Unified Test Suite** (`TestLABS`). This suite bridges high-level sequence transformations with our core energy logic.
* **AI Hallucination Guardrails:** To ensure the integrity of our optimized search logic, we maintain a "Golden Reference" `calculate_energy` function. Our Tabu search's incremental energy updates (`get_delta_energy`) are strictly validated against this reference to ensure numerical consistency.

### Core Correctness Checks
* **Check 1 (Symmetry & Invariance):** The problem holds that $E(S) = E(-S)$ and $E(S) = E(S_{reversed})$. Our `test_invariance_properties` suite asserts these physical identities using random sequences to ensure no directional bias exists in our energy calculations.
* **Check 2 (Ground Truth Validation):** We perform specific sequence checks for known energy outputs based on the visualization provided for $N=7$ (e.g., sequence "0100111" must yield $E=3$). These hard-coded tests prevent regression in our bit-to-spin transformations.
* **Check 3 (Exhaustive Integration):** Our `test_known_minima_small_n` suite runs the full Memetic Tabu Search (MTS) for $N=3$ through $N=20$. It asserts that the solver reaches the global minima defined in the Bernasconi model within the iteration limit.

---

## 5. Execution Strategy & Success Metrics

### Agentic Workflow
* **Plan:** We use VS Code with GitHub Copilot as our primary development environment. Development follows a test-driven workflow: the QA leads validate implementations with unit tests, and any failures are fed back to the agent for refactoring. The team coordinates through GitHub for version control and Discord for real-time communication.

<!-- ### Success Metrics
* **Metric 1 (Approximation):** [e.g., Target Ratio > 0.9 for N=30]
* **Metric 2 (Speedup):** [e.g., 10x speedup over the CPU-only Tutorial baseline]
* **Metric 3 (Scale):** [e.g., Successfully run a simulation for N=40]

### Visualization Plan
* **Plot 1:** [e.g., "Time-to-Solution vs. Problem Size (N)" comparing CPU vs. GPU]
* **Plot 2:** [e.g., "Convergence Rate" (Energy vs. Iteration count) for the Quantum Seed vs. Random Seed] -->

---

## 6. Resource Management Plan

### The $20 Budget Strategy

**Optimal GPU Selection:**

| Phase | GPU | Hourly Cost | Purpose | Hours | Budget |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dev & Debug | L4 | $0.60-0.87 | Test quantum circuits (N≤30), verify MTS logic, debug CUDA kernels | 6 | $4.50 |
| Production Testing | A100 80GB | $3.00-4.00 | Scale quantum algorithm N=35-40, parallelize MTS, benchmark N>50 | 3 | $10.50 |
| Final Benchmark | A100 80GB | $3.00-4.00 | Production runs for presentation plots | 1 | $3.50 |
| Buffer | - | - | Handle failures, re-runs | - | $1.50 |
| **Total** | | | | **10** | **$20.00** |

**Anti-Zombie Protocol (Resource Protection Plan):**
1. **Timer Alarms:** Set 30-minute timer alarms during active development sessions
2. **Auto-Shutdown:** Use Brev auto-shutdown feature (max 2-hour session without activity)
3. **Test-Then-Migrate:** Test all code on L4 before migrating to A100 instances
4. **Session Planning:** Pre-commit to A100 session plan with written checklist of experiments before spinning up
5. **Active Monitoring:** GPU Acceleration PIC monitors Brev dashboard every hour during active sessions
6. **Team Coordination:** All team members must announce in Discord before spinning up any GPU instance and when shutting it down
7. **Meal Break Protocol:** Mandatory instance shutdown during team meal breaks or extended discussions
