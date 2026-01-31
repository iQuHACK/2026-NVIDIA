# Product Requirements Document (PRD)

**Project Name:** [GQE-MTS]
**Team Name:** [QQQ]
**GitHub Repository:** [https://github.com/Jim137/2026-NVIDIA]

---

## 1. Team Roles & Responsibilities [You can DM the judges this information instead of including it in the repository]

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Jiun-Cheng Jiang | Jim137 | mrjiang |
| **GPU Acceleration PIC** (Builder) | [Name] | [@handle] | [@handle] |
| **GPU Acceleration PIC** (Builder) | [Name] | [@handle] | [@handle] |
| **Quality Assurance PIC** (Verifier) | YuChao Hsu | Astor-Hsu | Yuchao0520 |
| **Technical Marketing PIC** (Storyteller) | Yi-Kai Lee | leon53660713 | asdtaiwan |

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
* **Strategy:** [The classical search has many opportuntities for GPU acceleration. What will you chose to do?]
    * *Example:* "The standard MTS evaluates neighbors one by one. We will use `cupy` to rewrite the energy function to evaluate a batch of 1,000 neighbor flips simultaneously on the GPU."

### Hardware Targets
* **Dev Environment:** [e.g., Qbraid (CPU) for logic, Brev L4 for initial GPU testing]
* **Production Environment:** [e.g., Brev A100-80GB for final N=50 benchmarks]

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** [ruff, mypy]

### Physical and Algorithmic Correctness Checks

To validate both the physical fidelity of the LABS objective and the correctness of the accelerated evaluation pipeline, we design a collection of deterministic and symmetry-aware validation tests. These checks jointly verify analytical correctness, invariance properties, and algorithmic sanity across classical and quantum components.

* **Check 1 (Global Spin-Flip Invariance):**
  * **Principle:**  
    The LABS Hamiltonian is invariant under a global inversion of all spins,
    $S_i \rightarrow -S_i \quad \forall i,$
    leading to a $\mathbb{Z}_2$ symmetry of the energy landscape.
  * **Test:**  
    For any sampled spin sequence $S$, its inverted configuration $-S$ must yield the same energy.
  * **Assertion:**  
    $E(S) = E(-S).$
  * **Implementation:**  
    We explicitly evaluate both configurations and enforce
    ```python
    assert energy(S) == energy(-S)
    ```

* **Check 2 (Exact Energy Evaluation for Small Systems):**
  * **Principle:**  
    For small sequence lengths, the LABS energy can be computed analytically or via exhaustive enumeration, providing exact reference values.
  * **Test:**  
    For $N=3$, the energies of representative sequences are known:
      $
      E([1,1,1]) = 5,\quad
      E([1,1,-1]) = 1,\quad
      E([1,-1,1]) = 5.
      $
  * **Assertion:**  
    The evaluation kernel must reproduce these exact values without numerical deviation.
  * **Implementation:**  
    The test suite performs direct comparisons against analytically derived energies.
    
* **Check 3 (Index and Correlation Structure Validation):**
  * **Principle:**  
    The LABS energy computation relies on a deterministic generation of correlation indices.
  * **Test:**  
    For fixed system sizes (e.g., $N=6$ and $N=8$), the number and structure of generated correlation groups must match theoretical expectations.
  * **Assertion:**  
    Index generation outputs are checked against known reference patterns.

* **Check 4 (Sequence Reversal Symmetry):**
  * **Principle:**  
    The LABS objective depends only on pairwise correlations at fixed distances and is invariant under sequence reversal.
  * **Test:**  
    For any sequence $S$, its reversed sequence $S^{\mathrm{rev}}$ must yield the same energy.
  * **Assertion:**  
    $E(S) = E(S^{\mathrm{rev}}).$
  * **Implementation:**  
    Random sequences are reversed and validated via
    ```python
    assert energy(S) == energy(S[::-1])
    ```





---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow
* **Plan:** [How will you orchestrate your tools?]
    * *Example:* "We are using Cursor as the IDE. We have created a `skills.md` file containing the CUDA-Q documentation so the agent doesn't hallucinate API calls. The QA Lead runs the tests, and if they fail, pastes the error log back into the Agent to refactor."

### Success Metrics
* **Metric 1 (Approximation):** [e.g., Target Ratio > 0.9 for N=30]
* **Metric 2 (Speedup):** [e.g., 10x speedup over the CPU-only Tutorial baseline]
* **Metric 3 (Scale):** [e.g., Successfully run a simulation for N=40]

### Visualization Plan
* **Plot 1:** [e.g., "Time-to-Solution vs. Problem Size (N)" comparing CPU vs. GPU]
* **Plot 2:** [e.g., "Convergence Rate" (Energy vs. Iteration count) for the Quantum Seed vs. Random Seed]

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC 

* **Plan:** [How will you avoid burning all your credits?]
    * *Example:* "We will develop entirely on Qbraid (CPU) until the unit tests pass. We will then spin up a cheap L4 instance on Brev for porting. We will only spin up the expensive A100 instance for the final 2 hours of benchmarking."
    * *Example:* "The GPU Acceleration PIC is responsible for manually shutting down the Brev instance whenever the team takes a meal break."
