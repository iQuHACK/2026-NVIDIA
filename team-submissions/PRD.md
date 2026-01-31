# Product Requirements Document (PRD)

**Project Name:** Quantum-enhanced Generative Quantum Eigensolver with Tensor Networks and Multi-Trajectory Search for LABS Problem

**Team Name:** QQQ

**GitHub Repository:** https://github.com/Jim137/2026-NVIDIA

---

## 1. Team Roles & Responsibilities [You can DM the judges this information instead of including it in the repository]

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Jiun-Cheng Jiang | Jim137 | mrjiang |
| **GPU Acceleration PIC1** (Builder) | [Name] | [@handle] | [@handle] |
| **GPU Acceleration PIC2** (Builder) | [Name] | [@handle] | [@handle] |
| **Quality Assurance PIC** (Verifier) | YuChao Hsu | Astor-Hsu | Yuchao0520 |
| **Technical Marketing PIC** (Storyteller) | Yi-Kai Lee | leon53660713 | asdtaiwan |

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm
* **Algorithm:**
    * Generative Quantum Eigensolver (GQE) combined with Multi-Trajectory Search (MTS).
    * Using tensor network (TN) methods to efficiently simulate quantum circuits within energy evaluations.
    * Incorporating Quantum-inspired Kolmogorov-Arnold Networks (QKAN) as part of the transformer model, making it as QKANsformer to enhance the GQE framework.

* **Motivation:**
    * The GQE-MTS hybrid algorithm is selected to leverage the strengths of both quantum and classical optimization techniques for solving the LABS problem. The GQE component allows us to generate low-energy candidate solutions using quantum circuits, while the MTS component efficiently explores the solution space classically.
    * The integration of tensor network methods aims to optimize the simulation of quantum circuits, making it feasible to handle larger problem sizes. 
    * QKANsformer is empirically shown to have better expressivity, less computational resource requirements, and improved training efficiency compared to classical transformer models, which can enhance the performance of the GQE framework.
   

### Literature Review

* The generative quantum eigensolver (GQE) and its application for ground state search, Nakaji et al., https://arxiv.org/abs/2401.09253
    * This paper introduces the GQE algorithm, which combines quantum circuit-based state generation with classical optimization techniques. It is relevant to our project as it provides a framework for leveraging quantum resources to find low-energy configurations in combinatorial optimization problems like LABS. The insights from this paper will guide our implementation of the quantum component of our hybrid algorithm.
* Validating Large-Scale Quantum Machine Learning: Efficient Simulation of Quantum Support Vector Machines Using Tensor Networks, Chen et al., https://arxiv.org/abs/2405.02630
    * This paper discusses efficient simulation techniques for quantum machine learning algorithms, specifically quantum support vector machines (QSVMs). The relevance to our project lies in the potential application of tensor network methods to optimize the classical-quantum hybrid approach we are adopting. The techniques outlined in this paper may help us improve the efficiency of our classical optimization routines when integrated with quantum state generation.
* Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks, Jiang et al., https://arxiv.org/abs/2509.14026
    * This paper explores the use of quantum variational activation functions within neural network architectures, specifically Kolmogorov-Arnold networks. The relevance to our project is in the potential for using QKAN in transformer models to enhance the GQE framework. The insights from this paper will inform our approach to integrating quantum components into classical machine learning models for improved performance in solving the LABS problem.

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
      
      E([1,1,1]) = 5,E([1,1,-1]) = 1,E([1,-1,1]) = 5.
      
  * **Assertion:**  
    The evaluation kernel must reproduce these exact values without numerical deviation.
  * **Implementation:**  
    The test suite performs direct comparisons against analytically derived energies.
    
* **Check 3 (Index and Correlation Structure Validation):**
  * **Principle:**  
    The LABS energy computation relies on a deterministic generation of correlation indices.
  * **Test:**  
    For fixed system sizes $N=6$, the number and structure of generated correlation groups must match theoretical expectations.
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
* **IDE:** VS Code


### Success Metrics
* **Metric 1 (Approximation):** Obtain Merit Factor F = N² / (2E) > 6.0 for N=40.
* **Metric 2 (Speedup):** Achieve 10× speedup for classical MTS. 
* **Metric 3 (Scalability):** Successfully execute GQE-MTS for N =35,40,45.
* **Metric 4 (Quantum Advantage):** Ｑuantum seed demonstrates advantages over random initialization.
### Visualization Plan
* **Plot 1:** Solution Time as a Function of Problem Size (N) Across CPU and GPU Architectures.
* **Plot 2:** Convergence Rate (Energy vs Iteration) for Quantum vs Random vs Classical.

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC 

* **Plan:** [How will you avoid burning all your credits?]
    * *Example:* "We will develop entirely on Qbraid (CPU) until the unit tests pass. We will then spin up a cheap L4 instance on Brev for porting. We will only spin up the expensive A100 instance for the final 2 hours of benchmarking."
    * *Example:* "The GPU Acceleration PIC is responsible for manually shutting down the Brev instance whenever the team takes a meal break."
