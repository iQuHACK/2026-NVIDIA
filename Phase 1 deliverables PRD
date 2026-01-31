# Product Requirements Document (PRD)

**Project Name:** TINS-LABS (Topology-Informed Neural Seeding for LABS)  
**Team Name:** Schrodingers Qat
**GitHub Repository:** 

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm

* **Algorithm:** Digitized Counterdiabatic Quantum Optimization (DCQO) with 2-local counterdiabatic terms, followed by a novel **Topology-Informed Neural Seeding (TINS)** pipeline before MTS.

* **Motivation (Scientific Risk):** 
  - DCQO achieves 6x circuit depth reduction over QAOA (Hegade et al., PRR 2022), enabling practical GPU simulation at N=32+
  - The Kipu QE-MTS paper (Cadavid et al., 2025) demonstrated O(1.24^N) scaling using energy-ranked quantum seeds
  - **Our hypothesis:** Quantum samples encode richer landscape information than energy alone. By extracting topological structure (basin connectivity) and learned features (GNN-predicted MTS success), we can achieve better seed curation than naive energy ranking.
  - **Risk acknowledged:** This is untested for LABS. The landscape may be too rugged for learned seeding to help. A valid negative result would still contribute to understanding hybrid quantum-classical optimization.

### Novel Contribution: Topology-Informed Neural Seeding (TINS)

I propose a three-stage filtering pipeline between quantum sampling and classical MTS:

```
DCQO Samples → TDA Basin Analysis → GNN Quality Prediction → Diverse Seed Selection → MTS
```

**Stage 1: Topological Basin Analysis**
- Compute H₀ persistent homology of DCQO samples in Hamming metric
- Persistence diagram reveals basin structure without requiring training data
- Long-persisting components → stable solution clusters (exploitation targets)
- Short-persisting components → isolated outliers (exploration targets)

**Stage 2: Neural Seed Quality Prediction**
- Lightweight GNN trained to predict MTS convergence iterations from sample features
- Input features: energy, persistence-based cluster ID, local neighborhood structure
- Physics-inspired architecture following Schuetz et al. (2022)

**Stage 3: Diversity-Aware Selection**
- Combine TDA clustering with GNN ranking via weighted ensemble
- Ensure population diversity by sampling across persistence-identified basins
- Fallback: If GNN underperforms, use TDA-only or energy-only selection

### Literature Review

| Reference | Relevance |
|-----------|-----------|
| **Hegade et al., "Digitized-Counterdiabatic Quantum Optimization", PRR 2022** | Foundation for DCQO algorithm. Demonstrates polynomial enhancement over adiabatic QO with 2-local CD terms. We adopt their circuit structure. |
| **Cadavid et al., "Scaling advantage with quantum-enhanced memetic tabu search for LABS", arXiv:2511.04553 (2025)** | Direct predecessor. Establishes QE-MTS baseline with O(1.24^N) scaling. We extend their pipeline with intelligent seed selection. |
| **Schuetz et al., "Combinatorial Optimization with Physics-Inspired GNNs", Nature Machine Intelligence 2022** | Shows GNNs can solve QUBO/PUBO (which includes LABS) by treating Hamiltonian as differentiable loss. Informs our GNN architecture. |
| **Cappart et al., "Combinatorial Optimization and Reasoning with GNNs", JMLR 2023** | Comprehensive survey on GNN+local search hybridization. Key insight: GNNs can generate diverse candidates that seed classical search via hindsight loss. |
| **Lloyd et al., "Quantum algorithms for topological and geometric analysis of data", Nature Communications 2016** | Foundational quantum TDA paper. Establishes that persistent homology reveals multi-scale topological features. We apply classically to quantum samples. |
| **Karimi-Mamaghan et al., "ML at the Service of Meta-heuristics", EJOR 2022** | Taxonomy of ML integration points in metaheuristics: initialization, fitness evaluation, evolution. TINS targets the initialization phase. |
| **Packebusch & Mertens, "Low autocorrelation binary sequences", J. Phys. A 2016** | Landscape analysis of LABS. Shows exponentially many local minima, justifying need for intelligent seeding over random initialization. |

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)

* **Strategy:** 
  1. Implement DCQO circuit using CUDA-Q kernels with parameterized Trotter steps
  2. Use `nvidia` single-GPU backend for N≤30 (state vector fits in ~16GB)
  3. Target `nvidia-mgpu` backend for N>30 if time permits
  4. Batch sampling: Generate 1000+ samples per DCQO run to amortize circuit compilation overhead

* **Key optimization:** Pre-compute LABS Hamiltonian coefficients and store as GPU-resident tensors to avoid CPU-GPU transfer during sampling.

### Classical Acceleration (MTS)

* **Strategy:**
  1. **Vectorized energy computation:** Rewrite autocorrelation sums using CuPy to evaluate entire population simultaneously
  2. **Batched neighbor evaluation:** Instead of sequential single-flip checks, evaluate all N possible flips for all population members in parallel (population_size × N matrix operation)
  3. **TDA acceleration:** Use GPU-accelerated ripser (giotto-tda with CUDA backend) for persistent homology on large sample sets

* **Expected speedup:** 10-50x over CPU baseline for energy evaluation (the MTS bottleneck)

### Hardware Targets

| Phase | Environment | Hardware | Purpose |
|-------|-------------|----------|---------|
| Development | qBraid | CPU | Logic validation, unit tests |
| GPU Porting | Brev | L4 (24GB) | Initial GPU testing, small N |
| Benchmarking | Brev | A100 (80GB) | Final runs at N=32-40 |

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy

* **Framework:** `pytest` with property-based testing via `hypothesis`
* **AI Hallucination Guardrails:**
  1. All CUDA-Q kernels must pass energy bounds test: `0 ≤ E(s) ≤ N(N-1)/2`
  2. All generated sequences verified as valid binary: `s ∈ {-1, +1}^N`
  3. Cross-validate CUDA-Q energy against pure NumPy reference implementation
  4. Require 100% test pass before any GPU credit expenditure

### Core Correctness Checks

| Check | Description | Implementation |
|-------|-------------|----------------|
| **Symmetry** | `energy(S) == energy(-S)` for all sequences | Property test over 1000 random sequences |
| **Reflection** | `energy(S) == energy(reverse(S))` | Property test |
| **Ground Truth N=13** | Optimal energy E=4, merit factor F=14.08 | Assert solver finds known optimum |
| **Ground Truth N=21** | Optimal energy E=12, merit factor F=18.38 | Assert solver finds known optimum |
| **Hamiltonian Consistency** | `<ψ|H|ψ>` from CUDA-Q matches classical energy | Compare for 100 random product states |
| **TDA Sanity** | H₀ Betti number at ε=0 equals sample count | Verify ripser output |

### Verification Gates

```
Gate 1: All unit tests pass on CPU → Proceed to GPU porting
Gate 2: GPU results match CPU within 1e-6 → Proceed to benchmarking  
Gate 3: MTS finds known optima for N≤21 → Proceed to large-N experiments
```

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow

* **Plan:**
  1. Claude as primary code generator with CUDA-Q documentation in context
  2. Test-driven development: Write tests first, then implementation
  3. Verification loop: Generate → Test → Fix → Commit
  4. All code reviewed against literature before integration

* **Context management:** Maintain `skills.md` with CUDA-Q API reference and LABS-specific constraints to prevent hallucination

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Correctness** | 100% test pass | pytest report |
| **Quantum advantage** | TINS-seeded MTS converges faster than random-seeded MTS | Iterations-to-solution ratio |
| **TDA value** | TDA-filtered seeds outperform energy-only seeds | A/B comparison |
| **GPU speedup** | ≥10x over CPU baseline | Wall-clock time ratio |
| **Scale** | Successfully run N=32 | Completion without OOM |
| **Approximation ratio** | ≥0.95 for N≤28 | E_found / E_optimal |

### Visualization Plan

| Plot | Purpose |
|------|---------|
| **Iterations-to-Solution vs N** | Compare: Random seeds, Energy-ranked seeds, TINS seeds |
| **Persistence Diagram** | Visualize DCQO sample topology at different N |
| **Energy Distribution** | Histogram of DCQO samples vs random samples |
| **GPU vs CPU Runtime** | Scaling comparison |
| **MTS Convergence Curves** | Energy vs iteration for different seeding strategies |

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC

### Credit Conservation Strategy

| Rule | Rationale |
|------|-----------|
| **CPU-first development** | All logic validated on qBraid (free) before touching Brev |
| **L4 before A100** | Port to cheap GPU first, only use A100 for final benchmarks |
| **Strict shutdown policy** | Terminate Brev instance during any break >15 minutes |
| **Batch experiments** | Queue all benchmark runs before starting GPU, minimize idle time |
| **Checkpoint frequently** | Save intermediate results to avoid re-running on crash |

### Budget Allocation (assuming $20 credit)

| Phase | Estimated Cost | Duration |
|-------|----------------|----------|
| L4 GPU porting | $3 | 1 hour |
| L4 small-N benchmarks | $2 | 30 min |
| A100 large-N benchmarks | $10 | 1.5 hours |
| Buffer for debugging | $5 | - |

### Contingency

If TINS components (TDA/GNN) prove too slow or complex:
1. **Fallback 1:** TDA-only seeding (drop GNN)
2. **Fallback 2:** Energy-stratified seeding (drop TDA)
3. **Fallback 3:** Standard QE-MTS replication (Kipu baseline)

All fallbacks still demonstrate working quantum→MTS pipeline with GPU acceleration.

---

## 7. Deliverables & Expected Outputs
**Owner:** Project Lead

### Code Deliverables

| File | Description |
|------|-------------|
| `labs_hamiltonian.py` | LABS Hamiltonian construction and energy computation (CPU + GPU) |
| `dcqo_circuit.py` | CUDA-Q kernel implementing DCQO with counterdiabatic terms |
| `tda_filter.py` | Persistent homology analysis of quantum samples using ripser |
| `gnn_scorer.py` | Graph neural network for seed quality prediction |
| `tins_pipeline.py` | Full TINS integration: DCQO → TDA → GNN → seed selection |
| `mts_solver.py` | GPU-accelerated Memetic Tabu Search |
| `hybrid_solver.py` | Complete quantum-enhanced MTS pipeline |
| `benchmark.py` | Timing and quality benchmarks across configurations |
| `tests/` | Unit tests for all modules |

### Data Outputs

| Output | Format | Description |
|--------|--------|-------------|
| `results/energies_N{n}.csv` | CSV | Best energies found per configuration per N |
| `results/timing_N{n}.csv` | CSV | Wall-clock time and iterations to solution |
| `results/persistence_diagrams/` | PNG | H₀ persistence diagrams for DCQO samples |
| `results/convergence_curves/` | PNG | MTS energy vs iteration plots |

### Visualization Outputs

| Figure | Description |
|--------|-------------|
| `figures/iterations_vs_N.png` | Bar chart: Random vs Energy vs TINS seeding |
| `figures/speedup_gpu_cpu.png` | Line plot: GPU acceleration factor vs N |
| `figures/sample_topology.png` | Persistence diagram showing basin structure |
| `figures/energy_histogram.png` | DCQO samples vs random samples distribution |
| `figures/convergence_comparison.png` | MTS convergence for different seeding strategies |

### Documentation Outputs

| Document | Description |
|----------|-------------|
| `PRD.md` | This document - planning and architecture |
| `RESEARCH.md` | Technical findings, analysis, and conclusions |
| `README.md` | Setup instructions and usage guide |
| `PRESENTATION.pdf` | 5-10 min presentation slides for judging |

### Expected Quantitative Results

| Metric | Baseline (Random MTS) | Target (TINS-QE-MTS) |
|--------|----------------------|----------------------|
| Iterations to optimal (N=24) | ~10,000 | <5,000 |
| Iterations to optimal (N=28) | ~50,000 | <25,000 |
| GPU speedup factor | 1x | ≥10x |
| Approximation ratio (N=32) | 0.85 | ≥0.95 |

### Negative Result Outputs (If Hypothesis Fails)

If TINS does not outperform energy-only seeding:
- Detailed analysis of why (e.g., DCQO samples lack topological structure)
- Comparison of basin sizes across seeding strategies
- Recommendations for when topological seeding may/may not help
- This constitutes a valid scientific contribution

---

## Appendix: Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| TDA shows no structure in DCQO samples | Medium | Medium | Valid negative result; fallback to energy seeding |
| GNN fails to train in time | Medium | Low | Use TDA-only; GNN is additive, not critical |
| CUDA-Q compilation errors | Low | High | Extensive unit testing; fallback to qiskit simulation |
| GPU OOM at large N | Medium | Medium | Reduce batch size; use state vector chunking |
| Credits exhausted early | Low | High | Strict budget gates; CPU-first development |  

Here are the direct links to all referenced papers:

## **Primary References**

**1. Hegade et al., "Digitized-Counterdiabatic Quantum Optimization", PRR 2022**  
[https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.013015](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.013015)  
**arXiv**: [https://arxiv.org/abs/2110.00903](https://arxiv.org/abs/2110.00903)

**2. Cadavid et al., "Scaling advantage with quantum-enhanced memetic tabu search for LABS", arXiv:2511.04553 (2025)**  
[https://arxiv.org/abs/2511.04553](https://arxiv.org/abs/2511.04553)  
**HTML version**: [https://arxiv.org/html/2511.04553v1](https://arxiv.org/html/2511.04553v1)

## **GNN + Optimization References**

**3. Schuetz et al., "Combinatorial Optimization with Physics-Inspired GNNs", Nature Machine Intelligence 2022**  
**Primary**: [https://www.nature.com/articles/s42256-022-00468-6](https://www.nature.com/articles/s42256-022-00468-6)  
**arXiv**: [https://arxiv.org/abs/2107.01188](https://arxiv.org/abs/2107.01188)  
**AWS Blog**: [https://aws.amazon.com/blogs/quantum-computing/combinatorial-optimization-with-physics-inspired-graph-neural-networks/](https://aws.amazon.com/blogs/quantum-computing/combinatorial-optimization-with-physics-inspired-graph-neural-networks/)

**4. Cappart et al., "Combinatorial Optimization and Reasoning with GNNs", JMLR 2023**  
[http://jmlr.org/papers/v24/22-0913.html](http://jmlr.org/papers/v24/22-0913.html)  
**arXiv**: [https://arxiv.org/abs/2210.18080](https://arxiv.org/abs/2210.18080)

## **Foundational Works**

**5. Lloyd et al., "Quantum algorithms for topological and geometric analysis of data", Nature Communications 2016**  
[https://www.nature.com/articles/ncomms13389](https://www.nature.com/articles/ncomms13389)  
**arXiv**: [https://arxiv.org/abs/1805.10941](https://arxiv.org/abs/1805.10941)

**6. Karimi-Mamaghan et al., "ML at the Service of Meta-heuristics", EJOR 2022**  
[https://www.sciencedirect.com/science/article/pii/S037722172100796X](https://www.sciencedirect.com/science/article/pii/S037722172100796X)

**7. Packebusch & Mertens, "Low autocorrelation binary sequences", J. Phys. A 2016**  
[https://iopscience.iop.org/article/10.1088/1751-8113/49/20/205001](https://iopscience.iop.org/article/10.1088/1751-8113/49/20/205001)  
**arXiv**: [https://arxiv.org/abs/1601.02012](https://arxiv.org/abs/1601.02012)

***

**Most important for your LABS work**: 
- **Cadavid et al. (arXiv:2511.04553)** ← Direct QE-MTS baseline
- **Hegade et al. (PRR 2022)** ← DCQO algorithm foundation
- **Schuetz et al. (Nature MI 2022)** ← GNN physics inspiration



---

*Document prepared for MIT iQuHACK 2026 NVIDIA Challenge - Phase 1 Submission*
