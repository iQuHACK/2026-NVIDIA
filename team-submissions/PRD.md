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
|  | Sanjeev Chauhan | [@sanjeev-one](https://github.com/sanjeev-one) | @exquisite_dove_79161 |
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
* **Reference 1:** "New Improvements in Solving Large LABS Instances Using Massively Parallelizable Memetic Tabu Search" by Zhiwei Zhang, Jiayu Shen, Niraj Kumar, and Marco Pistoia (2025) - [https://arxiv.org/html/2504.00987v2](https://arxiv.org/html/2504.00987v2)
* **Relevance:** This paper provides the classical benchmark for our quantum-classical hybrid approach. It demonstrates:
    * State-of-the-art classical MTS implementation achieving up to 26× speedup on GPU compared to CPU implementations
    * Best-known LABS solutions for N=92-120, with new merit factor records for 16 problem sizes
    * Efficient GPU parallelization strategies (block-level and thread-level) that inform our classical acceleration design
    * Evidence that general-purpose solvers outperform skew-symmetry-constrained methods, validating our unrestricted quantum search approach
    * Serves as the performance baseline we aim to match or exceed with our DCQO + GPU-accelerated MTS hybrid approach

* **Reference 2:** "Quantum coherence and counterdiabatic quantum computing" by Raziel Huerta-Ruiz, Maximiliano Araya-Gaete, Diego Tancara, Enrique Solano, Nancy Barraza, and Francisco Albarrán-Arriagada (2025) - [https://arxiv.org/html/2504.17642v1](https://arxiv.org/html/2504.17642v1)
* **Relevance:** This paper provides the theoretical foundation for our impulse-to-adiabatic transition strategy:
    * Defines the impulse regime condition: max[λ̇(t)/Δ] ≫ 1, where Δ is the energy gap
    * Shows that in the impulse regime, counterdiabatic (CD) terms dominate and produce high quantum coherence, leading to better performance
    * Demonstrates that higher-order CD approximations generate more coherence and energy fluctuations in the impulse regime, enabling faster evolution
    * Explains why pure adiabatic evolution becomes necessary at the end: as λ̇(t) naturally decreases (by the fundamental theorem of calculus) and the energy gap Δ shrinks near the target state, CD dominance drops
    * Our strategy: Start in the impulse regime with CD terms for rapid exploration, then transition to pure adiabatic evolution by reintroducing the adiabatic Hamiltonian in the final stretch to maintain coherent evolution while preserving the hardware/time efficiency DCQO provides

* **Reference 3:** "Digitized counterdiabatic quantum optimization" by Narendra N. Hegade, Xi Chen, and Enrique Solano (2022) - Phys. Rev. Research 4, L042030
* **Relevance:** Establishes the DCQO framework that we build upon:
    * Introduces digitized implementation of counterdiabatic driving for gate-based quantum computers
    * Provides the nested commutator expansion method for approximating the adiabatic gauge potential (AGP)
    * Demonstrates DCQO's applicability to combinatorial optimization problems
    * Forms the basis for our variable λ schedule and hybrid impulse-adiabatic protocol

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

## 4. The Verification Plan

<!-- ### Unit Testing Strategy
* **Framework:** [e.g., `pytest`, `unittest`]
* **AI Hallucination Guardrails:** [How do you know the AI code is right?]
    * *Example:* "We will require AI-generated kernels to pass a 'property test' (Hypothesis library) ensuring outputs are always within theoretical energy bounds before they are integrated."

### Core Correctness Checks
* **Check 1 (Symmetry):** [Describe a specific physics check]
    * *Example:* "LABS sequence $S$ and its negation $-S$ must have identical energies. We will assert `energy(S) == energy(-S)`."
* **Check 2 (Ground Truth):**
    * *Example:* "For $N=3$, the known optimal energy is 1.0. Our test suite will assert that our GPU kernel returns exactly 1.0 for the sequence `[1, 1, -1]`." -->

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
