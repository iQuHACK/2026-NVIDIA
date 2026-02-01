# Product Requirements Document (PRD)

**Project Name:** Quantum LaBros
**Team Name:** QuantumBros
**GitHub Repository:** [Insert Link Here]

---

> **Note to Students:** The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist. 
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have a solid plan, not just to fill in boxes.

---

## 1. Team Roles & Responsibilities [You can DM the judges this information instead of including it in the repository]

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | kel404 | @kel404x | @kel404_ |
| **GPU Acceleration PIC** (Builder) | Meghaj Kabra | @MeghajBUD | @thechief1739 |
| **Quality Assurance PIC** (Verifier) | Catomakyto | @Catomakyto | @cate_96047 |
| **Technical Marketing PIC** (Storyteller) | Catomakyto | @Catomakyto| @cate_96047 |

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm
* **Algorithm:** QAOA with the LABS Hamiltonian (G2/G4 from the tutorial) and the standard X-mixer, in CUDA-Q. QAOA samples seed the initial population for classical MTS (QE-MTS). We use X-mixer first; Grover mixer is an optional upgrade if time permits.

* **Motivation:** QAOA is a different algorithm from the tutorial’s counteradiabatic, so it satisfies Phase 2’s custom quantum seed. We reuse the same LABS cost formulation (G2/G4). The X-mixer matches CUDA-Q examples and is quick to implement, so we can deliver a working, GPU-accelerated QE-MTS pipeline with lower risk.

### Literature Review
* **Reference:** “Scaling advantage with quantum-enhanced memetic tabu search for LABS,” Martínez et al., [arXiv:2511.04553](https://arxiv.org/abs/2511.04553).
* **Relevance:** This is the tutorial’s source. It defines the LABS Hamiltonian (G2/G4), the QE-MTS hybrid (quantum seed → MTS), and notes that QAOA can be used as the quantum subroutine; we follow that with an X-mixer QAOA implementation.

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Strategy:** Develop and test on Qbraid (CPU) first; then run CUDA-Q on a single GPU (e.g. L4) for validation. For larger N, target the `nvidia-mgpu` backend to distribute circuit simulation across multiple GPUs if needed.

### Classical Acceleration (MTS)
* **Strategy:** The standard MTS evaluates neighbors one by one. We will use `cupy` to evaluate LABS energy over batches of neighbor flips on the GPU so MTS can scale.

### Hardware Targets
* **Dev Environment:** Qbraid (CPU) for logic and unit tests; Brev L4 (or local GPU) for initial GPU runs.
* **Production Environment:** Brev L4 or A100 for final benchmarks and larger N.

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** pytest (e.g. `tests.py` for energy functions and quantum kernels).
* **AI Hallucination Guardrails:** AI-generated kernels must pass property checks (e.g. energy within theoretical bounds) before integration.

### Core Correctness Checks
* **Check 1 (Symmetry):** LABS sequence $S$ and its negation $-S$ must have identical energies. Assert `energy(S) == energy(-S)`.
* **Check 2 (Ground Truth):** For $N=3$, optimal energy is 1.0. Assert our code returns 1.0 for the sequence `[1, 1, -1]`.

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow
* **Plan:** Use Cursor as the IDE. QA Lead runs tests; on failure, paste error log back for refactor. Optionally maintain a skills or docs snippet for CUDA-Q so generated code stays consistent.

### Success Metrics
* **Metric 1 (Approximation):** Target approximation ratio > 0.9 for chosen N (e.g. N=30).
* **Metric 2 (Speedup):** Meaningful speedup over CPU-only tutorial baseline (e.g. 10x where applicable).
* **Metric 3 (Scale):** Successfully run the full QE-MTS workflow for larger N (e.g. N=40).

### Visualization Plan
* **Plot 1:** Time-to-Solution vs. problem size N (CPU vs. GPU).
* **Plot 2:** Convergence (energy vs. iteration) for quantum-seeded vs. random-seeded MTS.

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC

* **Plan:** Develop on Qbraid (CPU) until unit tests pass, then use a low-cost GPU (e.g. Brev L4) for porting and validation. Reserve expensive instances (e.g. A100) for final benchmarking only. Use local GPU (e.g. RTX 5080) when possible for testing and tuning. GPU Acceleration PIC shuts down cloud instances when not in use (e.g. breaks). Goal: minimize credit burn during prototyping; spend credits on final runs and key experiments.
