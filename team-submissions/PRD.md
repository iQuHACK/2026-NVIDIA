# Product Requirements Document (PRD)

**Project Name:** LABS Solver

**Team Name:** Squaxions

**GitHub Repository:** https://github.com/chtang-hmc/2026-NVIDIA

---

## 1. Team Roles & Responsibilities

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Brayden Mendoza | brayjmendoza | BrayJ |
| **GPU Acceleration PIC** (Builder) | Chengyi Tang | chtang-hmc | chengyitang |
| **Quality Assurance PIC** (Verifier) | Sofiia Zaozerska, Jiani Fu | szaozerska, jiafu1234 | sofigoldfox, fjn004 |
| **Technical Marketing PIC** (Storyteller) | Zaara Bhatia | Zaara230761 | zaarabhatia_31348 |

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm
* **Algorithm:** [Identify the specific algorithm or ansatz]
    * *Example:* "Quantum Approximate Optimization Algorithm (QAOA) with a hardware-efficient ansatz."
    * *Example:* "Variational Quantum Eigensolver (VQE) using a custom warm-start initialization."

* **Motivation:** [Why this algorithm? Connect it to the problem structure or learning goals.]
    * *Example (Metric-driven):* "We chose QAOA because we believe the layer depth corresponds well to the correlation length of the LABS sequences."
    *  Example (Skills-driven):* "We selected VQE to maximize skill transfer. Our senior members want to test a novel 'warm-start' adaptation, while the standard implementation provides an accessible ramp-up for our members new to quantum variational methods."
   

### Literature Review
#### Classical MTS
* **Reference:** ["New Improvements in Solving Large LABS Instances Using Massively Parallelizable Memetic Tabu Search", Zhang et al., https://arxiv.org/pdf/2504.00987]
* **Relevance:**
    * This paper demonstrates how classical MTS could be massively parallelized with modern GPU architecture. We will use this state-of-the-art classical algorithm for our final benchmark and comparisons.

* **Reference:** ["A GitHub Archive for Solvers and Solutions of the labs problem", Bošković, https://github.com/borkob/git_labs]
* **Relevance:**
    * Provides known correct implementation of LABS solver that we can check our answers to and benchmark with.

#### Quantum Algorithms
* **Reference:** [Title, Author, Link]
* **Relevance:** [How does this paper support your plan?]
    * *Example:* "Reference: 'QAOA for MaxCut.' Relevance: Although LABS is different from MaxCut, this paper demonstrates how parameter concentration can speed up optimization, which we hope to replicate."

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Strategy:** [How will you use the GPU for the quantum part?]
    * *Example:* "After testing with a single L4, we will target the `nvidia-mgpu` backend to distribute the circuit simulation across multiple L4s for large $N$."

 

### Classical Acceleration (MTS)
* **Strategy:** We will follow the GPU acceleration strategy proposed by Zhang et al. in "New Improvements in Solving Large LABS Instances Using Massively Parallelizable Memetic Tabu Search". In their paper, they described a GPU architecture that achieves an up to 26x speedup over 16-core CPU implementations. Their strategy has the following advantages:

1. **All-in GPU Architecture:** Instead of using the GPU merely as a calculator for energy values, the paper implements the entire Memetic Tabu Search loop as a single kernel launch. The CPU launches one kernel, and the GPU handles initialization, recombination, mutation, local search, and replacement without returning to the host. This eliminates the high latency penalty of PCIe data transfer (Host-Device switching) between iterations.

2. **Two-level Parallelism:** The paper explicitly divides labor between GPU Blocks and Threads to balance exploration and exploitation. Each CUDA Thread Block runs a completely independent replica of the algorithm. Within a single block (replica), the threads work together to parallelize the computationally heavy Tabu Search.

3. **Shared Memory Data Structures:** Global memory is too slow, but Shared Memory (L1) is small (e.g., 164 KB per SM on an A100). To fit the necessary data into the target ~5 KB per block (allowing more active blocks), the paper used bit vectors to represent the population and correlation matrices and sparse storage of the correlation matrix.

We will plan to implement as many of the strategies they mentioned as possible, given our time constraints. # TODO: mention t4 pretest Since the paper has shown effectiveness using A100 GPUs, we will run our final benchmarks on the A100 machines in Brev.

However, we notice that 

We will check our implementation against known correct implementations, provided in Bošković et al.

### Hardware Targets
* **Dev Environment:** We will use Qbraid CPUs for initial testing and code verification, T4 for verifying GPU path usage, and Brev L4 for initial GPU testing. Our budget will be 8 hours of testing at $0.85/hr. Total budget will be $6.9.
* **Production Environment:** We will use the Brev A100-80GB for final benchmarks. Our budget will be 7 hours of benchmark at $1.50/hr. Total budget will be $10.50.

Total amount will be $17.40. We will leave $2.60 buffer in case of extra benchmarking needed or for idle runs.

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

All tests could be found in the folder `tests/`.

### Verification goals (Definition of Done)
We consider the solver "verified" when it satisfies **correctness**, **reproducibility**, and **GPU/CPU consistency** requirements below:

- **Correctness**: Energy computation and local-search updates are correct (verified by unit + property tests), and end-to-end solvers return valid sequences with consistent energies.
- **Reproducibility**: Given fixed random seeds and hyperparameters, the CPU reference implementation produces deterministic outputs.
- **GPU/CPU parity (when applicable)**: GPU-accelerated components match the CPU reference on small deterministic cases (same initialization + same number of steps), and never produce invalid states (values must remain in \(\{-1, +1\}\)).
- **Performance sanity**: GPU path actually runs on GPU (device utilization) and shows a measurable speedup over the CPU baseline on a benchmark suite (see Section 5 metrics).

### Testing Environments
We will split verification by environment to control cost and isolate GPU-specific issues:

| Verification level | CPU (Qbraid) | T4 | L4 | A100 | Notes |
| --- | --- | --- | --- | --- | --- |
| Unit tests (fast) | ✅ | optional | optional | ❌ | Always run on CPU in CI; GPU optional locally |
| CPU regression tests (golden seeds) | ✅ | ✅ | ✅ | ✅ | Confirms determinism and guards against refactors |
| GPU/CPU parity tests (small N) | ❌ | ✅ | ✅ | ✅ | Exact match expected for deterministic kernels |
| End-to-end smoke (N~64, short budget) | ✅ | ✅ | ✅ | ✅ | Confirms no crashes + energy improves |
| Final benchmarks (large N / many walkers) | ❌ | ❌ | optional | ✅ | Only on A100 near final submission |

### Unit Testing Strategy
* **Framework:** `pytest`
* **Style:** deterministic, seed-controlled tests
* **Target modules:** CPU and GPU MTS implementation; CPU and GPU quantum LABS solver implementation

#### What we unit test (CPU reference)
We will test the smallest building blocks because GPU acceleration will reuse the same logic:

- **Energy correctness** (`mts.energy`):
  - known hand-checkable sequences (e.g., all-ones length 4)
  - single-vs-batch consistency (vectorized vs scalar path)
  - invariances (see "Gauge symmetries" below)
- **Incremental update correctness** (`mts._delta_energy_for_flip`, `mts._apply_flip_in_place`):
  - delta energy matches full recomputation for every 1-bit flip
  - autocorrelation vector updates match recomputation after flips
- **Genetic operators** (`mts.combine`, `mts.mutate`):
  - crossover produces prefix/suffix from parents (cut in \([1, N-1]\))
  - mutation probability extremes \(p=0\) (no change) and \(p=1\) (all flipped)
- **Local search** (`mts.tabu_search`):
  - returned best energy is non-worse than the initial sequence
  - returned energy matches `mts.energy(best_s)`
- **End-to-end determinism** (`mts.MTS`):
  - repeated runs with the same seed produce identical best sequence + best energy

These tests already exist in `tests/test_mts.py`, and will be extended as we add features (e.g., quantum-seeded population).

#### GPU unit + parity tests (small deterministic cases)
Because GPUs introduce additional failure modes (indexing, race conditions, integer overflow, silent wrong answers), we will add **parity tests** that compare GPU outputs to CPU outputs for small \(N\) and short runs:

- **Bit validity**: GPU always returns sequences in \(\{-1, +1\}\) and finite energies.
- **Parity on toy problems**: for \(N \in \{8, 16, 32\}\) and fixed seeds, compare:
  - initial energy (after initialization),
  - energy after a fixed small number of tabu steps,
  - best energy reported by the kernel.

If the final GPU kernel is intentionally stochastic / non-deterministic across architectures, parity tests will still enforce **invariants** (valid bits, monotone best-energy tracking, and energy recomputation on CPU matches the reported energy).

#### AI hallucination guardrails
We will treat AI-generated code as **untrusted** until it passes all checks below:

- For any new optimization (especially GPU kernels), we add/extend a unit/parity test that would fail for common indexing or sign bugs.
- **Property checks before merge**: invariances + delta-energy consistency must pass on random seeds for several small \(N\).
- **Cross-implementation checks**: GPU results must be revalidated by recomputing energy on CPU from the GPU-returned bitstring.

### Integration & system verification
- **Pipeline smoke test**: run the full classical solver end-to-end (generate/init → combine/mutate → tabu search → replacement) and confirm:
  - no crashes, correct output shapes/types,
  - best energy is non-increasing over iterations (history sanity),
  - result energy matches an independent recomputation.
- **Quantum-seeded integration (when added)**:
  - verify the quantum module produces a population of valid bitstrings,
  - verify `mts.MTS(..., population0=...)` consumes it and improves over the initial best (or at minimum does not degrade under identical compute budget).

### Core Correctness Checks
We will have the following correctness checks for our LABS solvers.

1. **Comparison to known references**
    * **Barker sequences**: for \(N=7, 11, 13\), verify our energy and autocorrelation calculations reproduce the expected near-perfect behavior (for Barker lengths where applicable) and match known reference energies.
    * **Exhaustive optimum for small N**: for small \(N\) (e.g., \(N \le 11\) where feasible), compute the exact optimum energy by exhaustive search and confirm our solver reaches it reliably (or at least never reports an energy below the proven optimum).
    * **External benchmark suite**: Bošković et al. provide reference results; we will compare energies and best-known sequences for basic correctness and regression checks.

2. **Physics / invariance checks (Ising gauge symmetries)**
    * **Gauge symmetries**: for any sequence \(S\), the following transformations must yield the exact same energy:
        * Reversal: \(s_1, s_2, \dots, s_N \to s_N, s_{N-1}, \dots, s_1\)
        * Inversion: \(s_i \to -s_i\)
        * Alternating inversion: \(s_i \to (-1)^i s_i\)
    * **Magnetization sanity** (manual / exploratory): as \(N\) grows, we expect magnetization (average spin) to be near 0 for good solutions.
    * **Correlation distribution sanity** (manual / exploratory): the autocorrelation values over distances should be centered around 0 without obvious structure.

For initial correctness checks, we will use both numerical and non-numerical tests (magnetization/correlation sanity). For large \(N\), we will primarily rely on automated numerical tests (reference comparisons, invariances, parity checks).


---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Success Metrics
* **Metric 1 (Approximation):** [e.g., Target Ratio > 0.9 for N=30]
* **Metric 2 (Speedup):** [e.g., 10x speedup over the CPU-only Tutorial baseline]
* **Metric 3 (Scale):** [e.g., Successfully run a simulation for N=40]

### Visualization Plan
* **Plot 1:** [e.g., "Time-to-Solution vs. Problem Size (N)" comparing CPU vs. GPU]
* **Plot 2:** [e.g., "Convergence Rate" (Energy vs. Iteration count) for the Quantum Seed vs. Random Seed]

### Agentic Workflow
We use AI agents (ChatGPT) as development accelerators, not autonomous decision-makers.vAI assistance is used for:
* Translating mathematical expressions into code,
* Generating boilerplate CUDA-Q kernels and classical routines,
* Drafting unit tests and identifying likely implementation errors,
* Clarifying algorithmic details and relevant literature.

All AI-generated outputs are manually reviewed and validated against physical intuition (spin models, Pauli operators), known LABS properties, and unit tests on small problem instances.
Critical components (energy evaluation, Hamiltonian construction, and MTS transitions) are verified independently to mitigate hallucinations or logic errors.

### Success Metrics
Success metrics are defined separately for correctness, convergence behavior, and runtime performance.

#### Phase 1: Correctness (Small N)
**Objective**: Verify correctness of both classical MTS and quantum-enhanced MTS implementations.

**Metrics**: Exact recovery of known minimal energies for LABS instances with $N\leq 10$; agreement between classical and quantum-enhanced energies within numerical tolerance; unit-test coverage of energy functions, Hamiltonian terms, and quantum kernels.

**Expectation**:
Both approaches should perform equivalently in this regime.

#### Performance and Scaling (Moderate N)
**Baseline**: Classical MTS initialized with random populations.

**Comparison**: Quantum-enhanced MTS initialized using samples from the quantum circuit.

**Measured Metrics**:
* Time-to-Threshold:
Time required to reach a fixed target energy or merit factor.

* Convergence Rate:
Energy as a function of MTS iterations.

* Initialization Quality:
Mean and best energy of the initial population.

* Run-to-Run Variability:
Variance of final energy across repeated runs.

**Expected Outcomes (Conservative)**:
* For small and moderate problem sizes (N≲30):
No asymptotic advantage is expected. Quantum-enhanced initialization may reduce convergence time or variance.

* For larger N explored experimentally:
Any observed improvement will be reported empirically.
No claims of polynomial or exponential quantum speedup will be made.
Results will be presented as comparative empirical measurements, not theoretical performance guarantees.

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC 

* **Plan:** [How will you avoid burning all your credits?]
    * *Example:* "We will develop entirely on Qbraid (CPU) until the unit tests pass. We will then spin up a cheap L4 instance on Brev for porting. We will only spin up the expensive A100 instance for the final 2 hours of benchmarking."
    * *Example:* "The GPU Acceleration PIC is responsible for manually shutting down the Brev instance whenever the team takes a meal break."

    To prevent unnecessary cloud usage, we employ a staged execution strategy.
Planned Credit Allocation (~$20):
L4 GPU (development and debugging): ~5 hours (~$5)
A100 GPU (benchmarking): ~4 hours (~$8)
Buffer for reruns and debugging: ~$4–5
Operational Controls:
Regular manual checks to ensure no idle instances remain active.
Explicit shutdown of GPU instances after each run.
Scaling to larger GPUs only after correctness is verified on smaller backends.

## 7. Detailed Tasking and Scheduling

- classical algs schedule (Chengyi)
    -
    * write tests (15:00)
    * run tests on qbraid (16:00)
    * write gpu code (17:00)
    * run tests on t4 (17:30)
    * create benchmarking code for classical (22:00)
    * test small benchmark on t4 (23:00)

- quantum algs schedule (everyone else)
    -
    * finish notebook (15:00)
    * decide which algorithm to use (18:00)
    * write tests (19:00)
    * write documentation for algorithm (23:00)
    * finish code for quantum algorithm (23:00)
    * iterate if needed

    * run tests on qbraid (0:00)
    * write gpu code (1:30)
    * synthesize classical and quantum codebase (2:15)
        * quantum produces population that feeds to classical
    * run tests on t4 (2:45)
    * create benchmarking code for quantum (22:00 from Chengyi)
    * test small benchmark on t4 (3:30)
    * run entire benchmark on a100 (8:45)

- presentation schedule (Sofi)
    - 
    * need to write documentation throughout
    * record AI usage and write vibe log
        * record good/bad ai usage
    
    * presentation plan
    * fill in details
    * practice presentation

- 2:30 Finish all coding. Start final benchmarking. Begin working on presentation material.

- 8:00 Finish benchmarking, start final work on presentations.

- 9:00 Finish writing presentation. Start practicing for presentation. Final checks for code styling and provide documentation for any unclear parts.

- 9:50 Final submission.