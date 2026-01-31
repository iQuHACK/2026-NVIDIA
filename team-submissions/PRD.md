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
* **Reference:** "New Improvements in Solving Large LABS Instances Using Massively Parallelizable Memetic Tabu Search" by Zhiwei Zhang, Jiayu Shen, Niraj Kumar, and Marco Pistoia (2025) - [https://arxiv.org/html/2504.00987v2](https://arxiv.org/html/2504.00987v2)
* **Relevance:** This paper provides the classical benchmark for our quantum-classical hybrid approach. It demonstrates:
    * State-of-the-art classical MTS implementation achieving up to 26× speedup on GPU compared to CPU implementations
    * Best-known LABS solutions for N=92-120, with new merit factor records for 16 problem sizes
    * Efficient GPU parallelization strategies (block-level and thread-level) that inform our classical acceleration design
    * Evidence that general-purpose solvers outperform skew-symmetry-constrained methods, validating our unrestricted quantum search approach
    * Serves as the performance baseline we aim to match or exceed with our DCQO + GPU-accelerated MTS hybrid approach

---

## 3. The Acceleration Strategy

<!-- ### Quantum Acceleration (CUDA-Q)
* **Strategy:** [How will you use the GPU for the quantum part?]
    * *Example:* "After testing with a single L4, we will target the `nvidia-mgpu` backend to distribute the circuit simulation across multiple L4s for large $N$."
 

### Classical Acceleration (MTS)
* **Strategy:** [The classical search has many opportuntities for GPU acceleration. What will you chose to do?]
    * *Example:* "The standard MTS evaluates neighbors one by one. We will use `cupy` to rewrite the energy function to evaluate a batch of 1,000 neighbor flips simultaneously on the GPU."

### Hardware Targets
* **Dev Environment:** [e.g., Qbraid (CPU) for logic, Brev L4 for initial GPU testing]
* **Production Environment:** [e.g., Brev A100-80GB for final N=50 benchmarks] -->

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
| Dev & Debug | L4 | $0.60-0.87 | Test quantum circuits (N≤30), verify MTS logic | 5 | $4.00 |
| Quantum Scale-Up | A100 40GB | $2.00-2.50 | Test quantum algorithm N=35-40 | 2 | $5.00 |
| MTS Acceleration | A100 40GB | $2.00-2.50 | Parallelize classical MTS, benchmark N>50 | 3 | $7.00 |
| Final Benchmark | A100 80GB | $3.00-4.00 | Production runs for presentation plots | 1 | $3.50 |
| Buffer | - | - | Handle failures, re-runs | - | $0.50 |
| **Total** | | | | **11** | **$20.00** |

**Anti-Zombie Protocol (Resource Protection Plan):**
1. **Timer Alarms:** Set 30-minute timer alarms during active development sessions
2. **Auto-Shutdown:** Use Brev auto-shutdown feature (max 2-hour session without activity)
3. **Test-Then-Migrate:** Test all code on L4 before migrating to A100 instances
4. **Session Planning:** Pre-commit to A100 session plan with written checklist of experiments before spinning up
5. **Active Monitoring:** GPU Acceleration PIC monitors Brev dashboard every hour during active sessions
6. **Team Coordination:** All team members must announce in Discord before spinning up any GPU instance and when shutting it down
7. **Meal Break Protocol:** Mandatory instance shutdown during team meal breaks or extended discussions
