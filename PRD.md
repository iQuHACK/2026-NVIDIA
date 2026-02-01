# Product Requirements Document (PRD)

**Project Name:** GPU-Accelerated Quantum-Enhanced Optimization for MTS  
**Team Name:** Entangled  
**GitHub Repository:** `https://github.com/nadakhatab/2026-NVIDIA`

---

## 1. Team Roles & Responsibilities
*(Per the template: you can DM judges this info instead of committing it.)*

| Role | Name | GitHub | Discord |
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Nada Ali | [nadakhatab](https://github.com/nadakhatab) | `nada_alii` |
| **GPU Acceleration PIC** (Builder) | Tushar Pandey | [pandey-tushar](https://github.com/pandey-tushar) | `tusharpandey` |
| **Quality Assurance PIC** (Verifier) | Tommy Nguyen | [NguyenHTommy](https://github.com/NguyenHTommy) | `tommtommbom` |
| **Technical Marketing PIC** (Storyteller) | Anirudh Raman | [AnirudhRaman3277](https://github.com/AnirudhRaman3277) | `Loaded_Diaper3277` |

**Additional Contributor:** Shah Md Khalil Ullah — [Kodarr](https://github.com/Kodarr) — Discord: `.nerd23`  
*(General development support and experimentation.)*

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm
- **Algorithm:** **QAOA** (initially \(p=1\)–\(p=3\), tuned based on feasibility) for a qubit encoding of the LABS objective / Hamiltonian.

- **How it fits the workflow (Quantum Seed → Classical MTS):**
  - Run QAOA to produce a measurement distribution over bitstrings of length \(N\).
  - **Sample bitstrings** (shots), map them to \(\pm 1\) sequences, compute their LABS energies, and keep the **top-\(K\)** unique lowest-energy samples.
  - Use these samples to **seed the initial population** for Classical MTS (fill remaining population slots with random and/or mutated variants for diversity).
  - Run MTS from this seeded population and report best energy and time-to-solution.

### Motivation
- **Metric-driven:** QAOA naturally outputs candidate solutions via sampling. We hypothesize QAOA samples yield a **better-than-random initial population**, improving MTS convergence and time-to-solution under a fixed compute budget.
- **Engineering-driven:** QAOA is straightforward to implement and benchmark in CUDA-Q and integrates cleanly with a sampling→population seeding interface.

### Literature Review
- **Reference 1:** *Scaling advantage with quantum-enhanced memetic tabu search for LABS* (`https://arxiv.org/html/2511.04553v1`)  
  - **Relevance:** Motivates hybrid workflows where a quantum routine provides a seed population for MTS and suggests benchmarking scaling and time-to-solution.
- **Reference 2:** *Parallel MTS by JPMorgan Chase* (`https://arxiv.org/pdf/2504.00987`)  
  - **Relevance:** Supports that MTS-like heuristics have strong opportunities for parallelization/acceleration (e.g., batch evaluation / parallel neighborhood search).
- **Reference 3:** Farhi et al., *A Quantum Approximate Optimization Algorithm* (`https://arxiv.org/abs/1411.4028`)  
  - **Relevance:** Establishes QAOA as a standard variational method for combinatorial objectives with solutions obtained through measurement samples.

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC (Tushar Pandey)

### Quantum Acceleration (CUDA-Q)
- **Strategy:**
  - **CPU validation first** (qBraid): verify correctness of kernels + sampling and confirm the pipeline works for small \(N\).
  - **GPU next** (Brev): move to CUDA-Q GPU backends to accelerate simulation/sampling throughput.
- **What we will accelerate:**
  - Sampling throughput (shots/sec).
  - Parameter sweeps / multi-start runs (to reduce sensitivity to initialization).

### Classical Acceleration (MTS)
- **Strategy:** GPU-accelerate the highest-cost operations in MTS:
  - Vectorize and GPU-accelerate **energy/correlation** computations (e.g., `cupy`).
  - **Batch neighbor evaluation**: evaluate many candidate flips in parallel on GPU (instead of sequential evaluation).
- **Goal:** Reduce time spent in repeated energy/delta evaluation while maintaining correctness.

### Hardware Targets
- **Dev Environment:** qBraid CPU for logic + Brev **L4/T4** for early GPU porting/benchmarks.
- **Benchmark Environment:** Brev **A100** for final scaling experiments (subject to budget).

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC (Tommy Nguyen)

### Unit Testing Strategy
- **Framework:** `pytest`
- **AI Hallucination Guardrails:**
  - No AI-generated solver/kernels are accepted unless **all tests pass**.
  - Any refactor affecting energy evaluation must retain correctness via **small-N brute force cross-checks**.
  - Benchmarks are separated from tests (tests must be fast and deterministic).

### Core Correctness Checks (committed)
- **Check 1 (Symmetry):**
  - Assert `E(s) == E(-s)` (global sign flip invariance)
  - Assert `E(s) == E(s[::-1])` (reversal invariance)
- **Check 2 (Ground truth):**
  - For small \(N\) (e.g., \(N \le 12\) or feasible), brute force optimum is computed and used as ground truth for validation.
- **Check 3 (Delta correctness, if using delta updates):**
  - For random sequences and indices \(i\), assert `E(flip_i(s)) == E(s) + dE(i)` for the incremental update implementation.
- **Check 4 (Quantum sampling validity):**
  - Sampled bitstrings have length \(N\), map cleanly to \(\pm 1\), energies are within expected bounds, and results are reproducible under fixed seeds/config.

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC (Anirudh Raman)

### Agentic Workflow
- Split implementation into parallel tracks:
  - **Quantum seed track:** QAOA kernel + sampling + seeding.
  - **Classical acceleration track:** GPU acceleration of energy / neighbor evaluation.
  - **QA track:** continuous testing + regression prevention.
  - **Analysis track:** logging + plotting + PRD/report clarity.

### Success Metrics (quantifiable)
- **Metric 1 (Seeding benefit):** For fixed \(N\), pop size, and compute budget, QAOA-seeded MTS has a better initial population quality than random seeding (e.g., lower best/median energy at iteration 0).
- **Metric 2 (Time-to-solution):** Achieve measurable runtime reduction relative to CPU baseline at comparable solution quality for a selected benchmark \(N\). *(Exact target TBD after initial profiling.)*
- **Metric 3 (Scale):** Demonstrate runs at larger \(N\) than Phase 1 baseline once GPU backends are enabled. *(Target \(N\) TBD.)*

### Visualization Plan
- **Plot 1:** Runtime vs \(N\) (CPU vs GPU; classical MTS vs QAOA-seeded MTS refinement)
- **Plot 2:** Best energy / energy gap vs \(N\) (include exact optimum where feasible)
- **Plot 3:** Convergence curves (best energy vs iteration/time): random-seeded vs QAOA-seeded

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC (Tushar Pandey)

- **Plan:**
  - Do all correctness work on CPU first (qBraid/local) before using Brev GPU time.
  - Use Brev GPU only for scheduled port validation + benchmarking windows.
  - GPU instances must be shut down during breaks and after runs (“no zombie instances”).
- **Budget (rough; adjust once instance pricing is confirmed):**
  - L4/T4: **TBD** hours for development + profiling
  - A100: **TBD** hours for final benchmarks
  - Buffer: **TBD**
- **Benchmark logging requirement:** Every benchmark run records GPU type, CUDA-Q target, \(N\), shots, QAOA depth \(p\), MTS pop size/generations, seeds, and wall-clock runtime.