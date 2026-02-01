# Product Requirements Document (PRD)

**Project Name:** LABS-Solv-QE (Quantum-Enhanced LABS Solver)  
**Team Name:** QuantumVibes  
**GitHub Repository:** [Insert Link Here]

---

## 1. Team Roles & Responsibilities

| Role | Name | GitHub Handle | Discord Handle |
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | [Sezer] | [@JuL-sezaR] | [@julsezar] |
| **GPU Acceleration PIC** (Builder) | [Nicholas] | [@GB102] | [@def__neo__] |
| **Quality Assurance PIC** (Verifier) | [Asadullah] | [@AsadullahGalib007] | [@asadullah_galib] |
| **Technical Marketing PIC** (Storyteller) | [Sezer] | [@JuL-sezaR] | [@julsezar] |

---

## 2. The Architecture
**Owner:** Project Lead (Sezer)

### Choice of Quantum Algorithm

* **Algorithm:** **Hybrid Counteradiabatic + Dynamic Circuit Optimization**
    * **Primary Approach:** Digitized Counteradiabatic Quantum Algorithm (from Milestone 1 tutorial)
    * **Novel Enhancement:** Dynamic qubit reuse with measurement-and-reset optimization inspired by arXiv:2511.22712
    * **Ancilla Management:** Strategic uncomputation based on SQUARE methodology (Ding et al., arXiv:2004.08539)

* **Motivation:** 

    **Metric-Driven Reasoning:**
    - The counteradiabatic approach from arXiv:2511.04553 demonstrates O(1.24^N) scaling vs. classical O(1.34^N) and QAOA O(1.46^N)
    - For N=20-40, this represents a significant computational advantage
    - Our initial validation (Milestone 1) shows ~25-30% better initial population energy vs. random initialization
    
    **Innovation Beyond Tutorial:**
    - **Dynamic Circuit Integration:** While the tutorial uses static ancilla allocation, we will implement measurement-and-reset (M&R) to aggressively reclaim qubits mid-circuit
    - **Qubit Reuse Strategy:** Combining SQUARE's uncomputation with dynamic circuit primitives will reduce physical qubit requirements by an estimated 30-50% based on results from arXiv:2511.22712 for similar quantum algorithms
    - **GPU-Friendly Design:** The modular structure of counteradiabatic circuits maps naturally to parallel GPU execution
    
    **Skills-Driven Reasoning:**
      - Our team has already validated the baseline counteradiabatic implementation in Milestone 1
      - The enhancement provides clear learning objectives: dynamic circuits, qubit reuse, and GPU optimization
      - Modular design allows parallel development (quantum optimization + classical acceleration)
      
  **Alternative Approach Considered and Rejected:**
      We considered using Matrix Product State (MPS) tomography (Kurmapu et al., PRX Quantum 2023) as an alternative quantum approach. While MPS successfully reconstructs 20-qubit states with ~4000 parameters and works well for 1D chains with area-law entanglement, we rejected this approach for the following reasons:
      
   1. **Different Problem Domain:** MPS tomography solves the inverse problem (state reconstruction from measurements), whereas LABS requires the forward problem (generating states that encode good solutions). The paper focuses on characterizing what states were produced, not producing states to solve optimization problems.
   
   2. **Unvalidated for Optimization:** While MPS excels at state reconstruction, there is no literature evidence it provides advantage for LABS optimization. This introduces high implementation risk compared to our validated counteradiabatic baseline.
   
   3. **Incompatible Metrics:** MPS tomography optimizes for fidelity with unknown states, while LABS requires minimizing combinatorial energy. The optimization landscapes are fundamentally different.
   
   4. **Timeline Risk:** Implementing MPS-based optimization from scratch would require abandoning our validated Milestone 1 work, with no guarantee of success within the hackathon timeline.
   
   However, the MPS paper provides valuable validation for our approach: it demonstrates that 20-qubit 1D systems with short-range interactions (exactly LABS's structure) can be efficiently represented and manipulated, supporting our choice of counteradiabatic methods for this problem class.

### Algorithmic Innovations

#### Innovation 1: Symmetry-Exploiting Hybrid Sampling
**What:** Extend the tutorial's basic sampling with our 4-strategy framework (amplitude, energy, diversity, hybrid)

**Why:** LABS has 4-fold symmetry (bit-flip, time-reversal, combined). For top 25% of quantum samples, we generate all symmetric variants "for free"

**Expected Impact:** 4× effective sampling efficiency without additional quantum shots

**Reference:** Our own creative contribution from Milestone 1 validation

#### Innovation 2: Adaptive Ancilla Reclamation 
**What:** Implement cost-effective reclamation (CER) from SQUARE paper with LABS-specific modifications

**Why:** The tutorial uses static ancilla. SQUARE shows up to 9.6× reduction in active quantum volume (AQV) for modular arithmetic

**Expected Impact:** 
- Reduce ancilla qubits by 40-60% 
- Lower gate noise through better locality
- Enable larger N on same hardware

**Reference:** SQUARE paper (Ding et al., 2020), Section IV-D and Figure 10

#### Innovation 3: Dynamic Circuit Qubit Reuse
**What:** Integrate measurement-and-reset optimization from arXiv:2511.22712

**Why:** CUDA-Q supports dynamic circuits. For LABS, we can measure and reset ancilla mid-circuit after energy evaluation of sub-sequences

**Expected Impact:**
- QPE-style circuits achieve 30-50% qubit reduction (arXiv:2511.22712)
- LABS modular structure is similar to QPE → expect comparable savings
- Reduces hardware requirements for N>30

**Reference:** "Qubit Reuse Beyond Reorder and Reset" (arXiv:2511.22712)

### Other Improvements
- Multiple GPUs for Tabu Search parallelization and increased qubit count.
- CuPY for CUDA Core Acceleration
- Reduced Floating Point Precision for less VRAM Usage
- Statevector Compression for less VRAM Usage

### Literature Review

**Reference 1:** "Scaling advantage with quantum-enhanced memetic tabu search"  
**Authors:** [LABS paper authors]  
**Link:** https://arxiv.org/html/2511.04553v1  
**Relevance:** 
- Establishes counteradiabatic approach as superior to QAOA for LABS
- Provides theoretical foundation for O(1.24^N) scaling
- Demonstrates quantum advantage for N≥40
- **Key Insight:** We validated their tutorial implementation and will now enhance it

**Reference 2:** "SQUARE: Strategic Quantum Ancilla Reuse"  
**Authors:** Yongshan Ding et al. (University of Chicago)  
**Link:** arXiv:2004.08539  
**Relevance:**
- Demonstrates 1.5X-9.6X reduction in active quantum volume for modular arithmetic
- Provides locality-aware allocation (LAA) and cost-effective reclamation (CER) heuristics
- Shows that strategic uncomputation can *reduce* swap gates despite adding uncompute gates
- **Key Insight:** "Surprisingly, adding gates for uncomputation creates ancilla with better locality" - applicable to LABS circuits

**Reference 3:** "Qubit Reuse Beyond Reorder and Reset"  
**Authors:** [arXiv:2511.22712 authors]  
**Link:** https://arxiv.org/abs/2511.22712  
**Relevance:**
- Demonstrates 30-50% qubit reduction for QPE, QFT, VQE through dynamic circuits
- Introduces classically-controlled gates for aggressive qubit reuse
- LABS circuits have similar modular structure to QPE
- **Key Insight:** "Moving measurements and introducing dynamic circuit primitives forges entirely new pathways for qubit reuse"

### Technical Approach Summary

```
┌─────────────────────────────────────────────────────────┐
│  QUANTUM SEED GENERATION (Enhanced Counteradiabatic)   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Trotterized Evolution (validated in Milestone 1)   │
│     • G2 interactions (90 for N=20)                    │
│     • G4 interactions (498 for N=20)                   │
│     • Counteradiabatic angles θ(t)                     │
│                                                         │
│  2. ANCILLA MANAGEMENT (NEW - SQUARE inspired)         │
│     • Locality-Aware Allocation (LAA)                  │
│     • Cost-Effective Reclamation (CER)                 │
│     • Reduces AQV by 1.5X-9X                           │
│                                                         │
│  3. DYNAMIC QUBIT REUSE (NEW - arXiv:2511.22712)      │
│     • Mid-circuit measurement & reset                   │
│     • Classically-controlled gates                      │
│     • Reduces qubit count by 30-50%                    │
│                                                         │
│  4. SYMMETRY EXPLOITATION (NEW - our contribution)     │
│     • Generate 4 variants from top samples             │
│     • 4× effective sampling efficiency                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  HYBRID SAMPLING (4 strategies from Milestone 1)       │
├─────────────────────────────────────────────────────────┤
│  • Amplitude-based                                      │
│  • Energy-based (greedy)                               │
│  • Diversity-aware                                      │
│  • Hybrid adaptive (best overall)                      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  CLASSICAL MTS REFINEMENT (GPU-accelerated)            │
└─────────────────────────────────────────────────────────┘
```

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC (Nicholas)

### Quantum Acceleration (CUDA-Q)

* **Strategy:** Multi-tier GPU acceleration with dynamic circuit optimization

**Tier 1: Baseline GPU Execution (Weeks 1-2)**
- Migrate validated Milestone 1 code to CUDA-Q `nvidia-mgpu` backend
- Target: L4 GPU for development, A100 for production benchmarks
- Distribute circuit simulation across multiple GPUs for N>20

**Tier 2: Ancilla Optimization (Week 2-3)**  
- Implement SQUARE's LAA/CER algorithms in CUDA-Q
- Use `qubit_release()` and `qubit_reset()` APIs for ancilla management
- **Key Optimization:** Reduce active qubit count → smaller state vector → faster simulation
- Expected: 2-3× speedup from smaller quantum state space

**Tier 3: Dynamic Circuit Integration (Week 3-4)**
- Implement mid-circuit measurement and classically-controlled gates
- Use CUDA-Q's dynamic circuit features: `mz()` measurement + conditional gates
- **Key Optimization:** Qubit reuse reduces memory footprint exponentially (2^n_qubits)
- Expected: Enable N=35-40 on A100 (vs N=25-30 without dynamic circuits)

**Implementation Plan:**
```python
# Pseudocode for enhanced circuit
@cudaq.kernel
def enhanced_trotterized_circuit(N, G2, G4, ...):
    # Standard initialization
    reg = cudaq.qvector(N)
    h(reg)
    
    # NEW: Track ancilla with SQUARE-style allocation
    ancilla_heap = []  # Manage reusable qubits
    
    for step in range(n_steps):
        # Apply G2/G4 rotations (from Milestone 1)
        apply_two_body_gates(reg, G2, thetas[step])
        apply_four_body_gates(reg, G4, thetas[step])
        
        # NEW: Dynamic ancilla management
        if should_reclaim(step):  # SQUARE CER heuristic
            # Uncompute + reset for reuse
            uncompute_ancilla(ancilla_subset)
            for anc in ancilla_subset:
                mz(anc)  # Measure
                reset(anc)  # Reset to |0⟩
                ancilla_heap.append(anc)
        
        # NEW: Allocate from heap when available
        if ancilla_needed:
            if ancilla_heap:
                reuse_qubits(ancilla_heap.pop())
            else:
                allocate_new_qubits()
```

**Hardware Targets:**
- **Dev Environment:** Qbraid (CPU) for logic testing and unit tests
- **Initial GPU Testing:** Brev L4 (8 hours @ $1.60/hr = $12.80 budget)
- **Production Benchmarks:** Brev A100-80GB (4 hours @ $3.09/hr = $12.36 budget)
- **Budget Allocation:** $25.16 / $20 available → need 20% optimization or reduce hours

### Classical Acceleration (MTS)

* **Strategy:** GPU-accelerated batch evaluation with CuPy

**Baseline MTS Issues:**
- Sequential evaluation: tests one neighbor at a time
- LABS energy calculation is O(N²) per sequence
- For N=30, each MTS iteration evaluates ~30 neighbors → 30 × 900 = 27K operations serially

**Our GPU Acceleration Plan:**

**Phase 1: Batch Energy Evaluation**
```python
import cupy as cp

def gpu_batch_labs_energy(sequences_batch, N):
    """
    Evaluate LABS energy for batch of sequences on GPU.
    
    Args:
        sequences_batch: CuPy array of shape (batch_size, N)
        N: Sequence length
    
    Returns:
        Energies: CuPy array of shape (batch_size,)
    """
    batch_size = sequences_batch.shape[0]
    energies = cp.zeros(batch_size)
    
    # Parallelize over lags k and batch
    for k in range(1, N):
        # Vectorized autocorrelation: (batch_size, N-k)
        autocorr = sequences_batch[:, :-k] * sequences_batch[:, k:]
        # Sum and square for each sequence
        autocorr_sum = cp.sum(autocorr, axis=1)
        energies += autocorr_sum ** 2
    
    return energies
```

**Expected Impact:**
- Evaluate 1000 neighbors in parallel on GPU
- For N=30: 1000 parallel evaluations vs. 1000 serial
- Expected speedup: 50-100× for energy evaluation step
- Overall MTS speedup: 10-20× (energy is bottleneck)

**Phase 2: GPU-Accelerated Neighbor Generation**
- Generate all bit-flip neighbors on GPU in parallel
- Use CUDA kernels for efficient neighbor enumeration
- Expected: Additional 2-3× speedup

**Phase 3: Memory Optimization**
- Stream sequences to/from GPU to avoid memory bottleneck
- Use cupy memory pool for efficient allocation
- Target: Support batches of 10,000+ neighbors

### Hardware Targets
* **Dev Environment:** Qbraid (CPU) for logic validation
* **GPU Testing:** Brev L4 for initial CuPy port (cheap: $0.20/hr)
* **Production:** Brev A100 for final benchmarks (maximum parallelism)

**Compute Budget Breakdown:**
```
Quantum Development (L4):     8 hours × $0.20/hr = $1.60
Quantum Production (A100):    4 hours × $3.09/hr = $12.36
Classical Development (L4):   10 hours × $0.20/hr = $2.00
Classical Production (A100):  2 hours × $3.09/hr = $6.18
Buffer:                                            = -$2.14 (OVER)
─────────────────────────────────────────────────────────
TOTAL:                                             = $22.14

OPTIMIZATION NEEDED: Reduce by 2 hours or improve efficiency
```

**Risk Mitigation:**
- Complete ALL logic validation on free Qbraid CPU before GPU
- Use L4 for extensive testing (5× cheaper than A100)
- Reserve A100 only for final benchmarking runs
- Implement automatic instance shutdown after 15 min idle

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC (Asadullah)

### Unit Testing Strategy

* **Framework:** `pytest` with `hypothesis` for property-based testing
* **Coverage Target:** >90% for critical paths (quantum circuits, energy calculation, GPU kernels)

### AI Hallucination Guardrails

**Strategy:** Three-layer verification before integrating AI-generated code

**Layer 1: Property Tests (Automated)**
```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=-1, max_value=1), min_size=3, max_size=20))
def test_energy_bounds(sequence):
    """Energy must be within theoretical bounds."""
    N = len(sequence)
    energy = calculate_labs_energy(sequence, N)
    
    # Worst case: all same → maximum autocorrelation
    worst_case = sum((N-k)**2 for k in range(1, N))
    
    assert 0 <= energy <= worst_case, f"Energy {energy} out of bounds [0, {worst_case}]"

@given(st.lists(st.integers(min_value=-1, max_value=1), min_size=3, max_size=20))
def test_symmetry_invariance(sequence):
    """LABS energy must respect all symmetries."""
    N = len(sequence)
    e_orig = calculate_labs_energy(sequence, N)
    
    # Test bit-flip symmetry
    e_flip = calculate_labs_energy([-s for s in sequence], N)
    assert e_flip == e_orig, "Bit-flip symmetry violated"
    
    # Test time-reversal symmetry
    e_rev = calculate_labs_energy(sequence[::-1], N)
    assert e_rev == e_orig, "Time-reversal symmetry violated"
```

**Layer 2: Known Ground Truth (Manual)**
- N=3: Optimal energy = 1 (sequence: [1,1,-1])
- N=4: Optimal energy = 2 (known from literature)
- Validation suite with 10+ known optimal solutions for N≤10

**Layer 3: Cross-Validation (Automated)**
- Compare quantum circuit output against classical simulation
- Verify GPU results match CPU results (CuPy vs NumPy)
- Check ancilla management: all reclaimed qubits return to |0⟩ state

### Core Correctness Checks

**Check 1: LABS Symmetries (Physics)**
```python
def test_labs_symmetries():
    """All four LABS symmetries must be preserved."""
    for N in [5, 10, 15, 20]:
        for trial in range(100):
            seq = generate_random_sequence(N)
            e_base = labs_energy(seq)
            
            # Four symmetric variants must have same energy
            assert labs_energy([-s for s in seq]) == e_base  # Bit-flip
            assert labs_energy(seq[::-1]) == e_base  # Time-reversal
            assert labs_energy([-s for s in seq[::-1]]) == e_base  # Combined
            assert labs_energy(seq) == e_base  # Identity (sanity)
```

**Check 2: Quantum Circuit Correctness**
```python
def test_quantum_circuit_produces_valid_bitstrings():
    """All quantum measurements must be valid N-bit strings."""
    N = 10
    result = sample_quantum_circuit(N, shots=1000)
    
    for bitstring, count in result.items():
        assert len(bitstring) == N, f"Invalid length: {len(bitstring)} != {N}"
        assert all(b in '01' for b in bitstring), f"Invalid characters in {bitstring}"
        assert count > 0, f"Invalid count: {count}"
```

**Check 3: Ancilla Reclamation (SQUARE validation)**
```python
def test_ancilla_properly_reclaimed():
    """Reclaimed ancilla must return to |0⟩ state."""
    # Track qubit states throughout circuit
    states = simulate_with_tracking(enhanced_circuit, N=10)
    
    for qubit_id, state_history in states.items():
        if qubit_is_ancilla(qubit_id):
            # After reclamation, state must be |0⟩
            final_state = state_history[-1]
            assert np.allclose(final_state, [1, 0]), \
                f"Ancilla {qubit_id} not properly reset: {final_state}"
```

**Check 4: GPU-CPU Equivalence**
```python
def test_gpu_cpu_equivalence():
    """GPU and CPU implementations must produce identical results."""
    N = 20
    test_sequences = generate_test_suite(N, num_sequences=100)
    
    cpu_energies = [cpu_labs_energy(seq, N) for seq in test_sequences]
    gpu_energies = gpu_batch_labs_energy(cp.array(test_sequences), N).get()
    
    assert np.allclose(cpu_energies, gpu_energies, rtol=1e-10), \
        "GPU results differ from CPU baseline"
```

**Check 5: Dynamic Circuit Validity**
```python
def test_dynamic_circuit_preserves_output():
    """Dynamic qubit reuse must not change final output distribution."""
    N = 15
    
    # Sample with and without dynamic reuse
    static_result = sample_static_circuit(N, shots=5000)
    dynamic_result = sample_dynamic_circuit(N, shots=5000)
    
    # Distribution should be statistically similar
    kl_divergence = calculate_kl(static_result, dynamic_result)
    assert kl_divergence < 0.1, f"Dynamic circuit changes distribution: KL={kl_divergence}"
```

### Continuous Integration Plan

**Pre-Commit Hooks:**
- Run fast unit tests (<10s)
- Check symmetry properties
- Validate code formatting

**CI Pipeline (GitHub Actions):**
1. **CPU Tests** (every commit)
   - All unit tests
   - Property-based tests (100 samples)
   - Known ground truth validation
   
2. **GPU Tests** (before GPU time allocation)
   - GPU-CPU equivalence
   - Performance benchmarks
   - Memory usage checks

3. **Integration Tests** (weekly)
   - Full workflow: quantum → sampling → MTS
   - Compare against Milestone 1 baseline
   - Regression tests

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow

**Plan:** Cursor IDE with custom skills and validation loops

**Workflow Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│  HUMAN (Project Lead)                                   │
│  • Defines high-level requirements                      │
│  • Reviews AI-generated code                            │
│  • Makes architectural decisions                        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│  AGENT A (Code Generator)                               │
│  • Context: CUDA-Q docs in skills.md                    │
│  • Task: Generate quantum circuits & GPU kernels        │
│  • Constraint: Must pass unit tests before commit       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│  AGENT B (Test Generator - QA PIC)                      │
│  • Context: Property testing best practices             │
│  • Task: Generate property tests for new code           │
│  • Constraint: Must achieve >90% coverage               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────┐
│  VALIDATION LOOP                                        │
│  • Run pytest suite                                      │
│  • Check GPU-CPU equivalence                            │
│  • Verify symmetries and ground truth                   │
└────────────────┬────────────────────────────────────────┘
                 │
         ┌───────┴───────┐
         │               │
         ↓               ↓
    ✅ PASS         ❌ FAIL
         │               │
         │               ↓
         │     ┌─────────────────┐
         │     │ Paste error log  │
         │     │ to Agent A       │
         │     │ (refactor loop)  │
         │     └─────────┬─────────┘
         │               │
         └───────────────┘

```

**Key Practices:**
1. **Context Management:** 
   - Create `skills/cuda_q_docs.md` with CUDA-Q API reference
   - Create `skills/square_paper.md` with ancilla reuse algorithms
   - Create `skills/labs_theory.md` with problem symmetries
   
2. **Incremental Development:**
   - Start: "Generate a test for LABS symmetry preservation"
   - Then: "Generate function that passes this test"
   - Then: "Optimize this function for GPU with CuPy"
   
3. **Validation-First:**
   - QA PIC generates tests BEFORE features are implemented
   - AI must pass all tests before code review
   - No manual "looks good" checks - only automated validation

### Success Metrics

**Metric 1: Approximation Quality**
- **Target:** Approximation Ratio > 0.90 for N=30
- **Definition:** Final energy / Known optimal energy
- **Baseline:** Random initialization achieves ~0.75-0.80
- **Measurement:** Average over 10 independent runs

**Metric 2: Time-to-Solution Speedup**
- **Target:** 10× speedup over CPU-only Tutorial baseline
- **Components:**
  - Quantum: 2-3× from ancilla optimization (smaller state space)
  - Classical: 10-20× from GPU batch evaluation
  - Combined: 10× overall (classical MTS is bottleneck)
- **Measurement:** Wall-clock time for N=20, 25, 30

**Metric 3: Scalability**
- **Target:** Successfully solve N=35-40 (vs tutorial's N≤25)
- **Enabler:** Dynamic circuits reduce qubit requirements by 30-50%
- **Validation:** Produce valid solutions with energy < 1.2× best known

**Metric 4: Resource Efficiency**
- **Target:** 40% reduction in active qubits through SQUARE + dynamic circuits
- **Measurement:** Compare qubit usage vs. static allocation baseline
- **Reference:** SQUARE achieves 1.5X-9.6X reduction for arithmetic circuits

**Metric 5: GPU Utilization**
- **Target:** >80% GPU utilization during MTS refinement
- **Measurement:** `nvidia-smi` monitoring during runs
- **Validation:** Ensure batch sizes are large enough to saturate GPU

### Visualization Plan

**Plot 1: Time-to-Solution vs. Problem Size**
```
┌─────────────────────────────────────────┐
│ Time-to-Solution vs N                   │
│                                         │
│    ╔════════╗ CPU Baseline             │
│    ║        ║                           │
│    ║    ╔═══╩══╗ GPU (Ours)            │
│    ║    ║      ║                       │
│ ───╨────╨──────╨────────────            │
│   N=20  N=25  N=30  N=35               │
└─────────────────────────────────────────┘
X-axis: Problem size N
Y-axis: Wall-clock time (seconds, log scale)
Lines: CPU baseline, L4 GPU, A100 GPU
```

**Plot 2: Quantum Seed Quality**
```
┌─────────────────────────────────────────┐
│ Initial Population Energy Distribution  │
│                                         │
│  Random:     ▁▂▃▅▇▅▃▂▁                 │
│                                         │
│  Quantum:  ▁▂▅▇▅▂▁                     │
│           ←better                       │
└─────────────────────────────────────────┘
X-axis: LABS energy
Y-axis: Frequency
Overlay: Random vs. Quantum distributions
```

**Plot 3: Convergence Comparison**
```
┌─────────────────────────────────────────┐
│ MTS Convergence: Quantum vs Random     │
│                                         │
│  ╲                                     │
│   ╲──── Random seed                   │
│    ╲                                   │
│     ╲╲                                 │
│      ╲╲─── Quantum seed               │
│       ╲──────────                     │
└─────────────────────────────────────────┘
X-axis: MTS iteration
Y-axis: Best energy found (log scale)
Lines: Random vs. Quantum initialization
```

**Plot 4: Qubit Usage Over Time**
```
┌─────────────────────────────────────────┐
│ Active Qubits: Static vs Dynamic       │
│                                         │
│ ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁  Static (baseline)    │
│                                         │
│ ▁▂▃▄▅▄▃▂▁▂▃▄▃▂▁  Dynamic (ours)       │
│                                         │
└─────────────────────────────────────────┘
X-axis: Circuit execution time
Y-axis: Active qubits
Compare: Static allocation vs. our dynamic reuse
Highlight: Area under curve = Active Quantum Volume
```

**Plot 5: GPU Acceleration Breakdown**
```
┌─────────────────────────────────────────┐
│ Speedup Components                      │
│                                         │
│ Quantum:  ████ 2.5×                    │
│ Classical:██████████ 15×               │
│ Combined: █████ 10×                    │
│                                         │
└─────────────────────────────────────────┘
Bar chart showing speedup from each component
```

### Deliverables Checklist

- [ ] **Code:** GitHub repo with all implementations
- [ ] **Tests:** pytest suite with >90% coverage
- [ ] **Documentation:** README, API docs, setup instructions
- [ ] **Benchmarks:** CSV files with timing and quality metrics
- [ ] **Visualizations:** All 5 plots with analysis
- [ ] **Report:** Technical write-up explaining innovations
- [ ] **Presentation:** 10-minute demo with live results

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC (Nicholas)

### Budget Allocation Strategy

**Total Budget:** $20.00 in Brev credits

**Phase 1: Development & Testing (CPU + L4)**
```
Week 1-2: Logic validation on Qbraid CPU         FREE
Week 2-3: L4 testing (10 hours @ $0.20/hr)      $2.00
Week 3-4: L4 optimization (10 hours @ $0.20/hr) $2.00
──────────────────────────────────────────────────────
Subtotal Development:                             $4.00
```

**Phase 2: Production Benchmarks (A100)**
```
Final quantum benchmarks (4 hours @ $3.09/hr)    $12.36
Final classical benchmarks (2 hours @ $3.09/hr)  $6.18
──────────────────────────────────────────────────────
Subtotal Production:                              $18.54
```

**Phase 3: Buffer & Contingency**
```
Remaining buffer:                                 $1.46
Emergency re-runs (if needed):                    −$3.54 
──────────────────────────────────────────────────────
RISK: Slightly over budget
```

### Budget Optimization Strategies

**Strategy 1: Reduce A100 Time (Priority 1)**
- Batch ALL production runs into single 5-hour session
- Pre-compute all parameters and test scripts on CPU/L4
- Use automated scripts to run benchmark suite unattended
- **Savings:** Eliminate instance startup/shutdown overhead

**Strategy 2: Maximize L4 Usage (Priority 2)**
- L4 is 15× cheaper than A100 ($0.20 vs $3.09/hr)
- Validate that L4 is sufficient for N≤25
- Only use A100 for N>25 and final presentation runs
- **Savings:** ~$10 if most work stays on L4

**Strategy 3: Checkpoint-Resume (Priority 3)**
- Save intermediate results every 30 minutes
- If instance crashes, resume from checkpoint
- Avoid wasting compute on failed runs
- **Savings:** Insurance against wasted GPU time

**Strategy 4: Instance Monitoring (Priority 4)**
- Set up automatic shutdown after 15 minutes idle
- Nicholas (GPU PIC) responsible for monitoring Brev dashboard
- Team alarm every 2 hours during GPU sessions
- **Savings:** Prevent "zombie instances" (biggest risk)

### Execution Timeline

```
┌─────────────────────────────────────────────────────┐
│ Week 1: CPU Development (Qbraid - FREE)            │
├─────────────────────────────────────────────────────┤
│ Mon-Tue: Implement SQUARE ancilla management       │
│ Wed-Thu: Add dynamic circuit support               │
│ Fri:     Unit tests and validation                 │
├─────────────────────────────────────────────────────┤
│ Week 2: L4 GPU Testing ($2.00)                     │
├─────────────────────────────────────────────────────┤
│ Mon-Tue: Port quantum circuit to CUDA-Q GPU        │
│ Wed:     Validate quantum results vs CPU           │
│ Thu-Fri: Optimize quantum circuit (ancilla reuse)  │
├─────────────────────────────────────────────────────┤
│ Week 3: Classical GPU Acceleration ($2.00)         │
├─────────────────────────────────────────────────────┤
│ Mon-Tue: Implement CuPy batch energy evaluation    │
│ Wed:     GPU-CPU equivalence testing               │
│ Thu-Fri: Optimize MTS with GPU acceleration        │
├─────────────────────────────────────────────────────┤
│ Week 4: Production Benchmarks ($18.54)             │
├─────────────────────────────────────────────────────┤
│ Mon:     Final validation on L4                    │
│ Tue-Wed: A100 benchmarking session (6 hours)       │
│ Thu:     Generate visualizations and analysis      │
│ Fri:     Final presentation preparation            │
└─────────────────────────────────────────────────────┘
```

### Risk Mitigation

**Risk 1: Budget Overrun**
- **Probability:** Medium
- **Impact:** High (cannot complete benchmarks)
- **Mitigation:**
  - Validate ALL code on CPU before GPU time
  - Use L4 for extensive testing
  - Reserve A100 for final 6-hour session only
  - If over budget: reduce A100 time or seek extension

**Risk 2: Code Doesn't Work on GPU**
- **Probability:** Low (we have validated baseline)
- **Impact:** High (major refactoring needed)
- **Mitigation:**
  - Incremental GPU migration (one component at a time)
  - GPU-CPU equivalence tests at each step
  - Keep CPU fallback working throughout

**Risk 3: Dynamic Circuits Not Supported**
- **Probability:** Low (CUDA-Q supports dynamic circuits)
- **Impact:** Medium (lose 30-50% qubit reduction)
- **Mitigation:**
  - Test dynamic circuit APIs early (Week 1)
  - Fall back to SQUARE-only optimization if needed
  - Still achieve significant improvement from SQUARE

**Risk 4: Zombie Instance**
- **Probability:** Medium (human error)
- **Impact:** High (waste entire budget in one night)
- **Mitigation:**
  - **MANDATORY:** Set calendar alarms every 2 hours during GPU sessions
  - **RULE:** Nicholas must confirm instance shutdown before leaving computer
  - **AUTOMATION:** Script to shutdown after 15 minutes idle
  - **BACKUP:** Brev email notifications for long-running instances

### Emergency Protocols

**If Budget Runs Out Early:**
1. Switch to CPU-only benchmarks (slower but free)
2. Focus on quality over scale (N=20-25 instead of N=30-40)
3. Request budget extension with justification
4. Use team member's personal Brev credits if approved

**If Instance Crashes:**
1. Check checkpoint files immediately
2. Resume from last checkpoint if available
3. Document crash in GitHub issue
4. Report to judges if affects deliverables

**If Results Don't Meet Targets:**
1. Analyze failure modes with QA PIC
2. Run ablation study: which optimizations work?
3. Iterate on working components
4. Adjust success metrics if needed (document reasoning)

---

## 7. Summary & Innovation Statement

### What Makes This Project Stand Out?

**1. Three-Way Innovation Fusion:**
- **Paper 1 (LABS):** Validated counteradiabatic baseline
- **Paper 2 (SQUARE):** Strategic ancilla reuse for locality
- **Paper 3 (Dynamic):** Measurement-and-reset for qubit reduction
- **Our Contribution:** First implementation combining all three for LABS

**2. Concrete, Measurable Goals:**
- Not "try to improve" - specific targets: 10× speedup, N=35-40, 40% qubit reduction
- Clear baseline from Milestone 1 validation
- Every metric has measurement plan

**3. Professional Development Process:**
- AI-assisted but human-validated
- Property-based testing catches hallucinations
- Incremental GPU migration with checkpoints
- Detailed resource management plan

**4. Learning-Focused:**
- Team gains experience with: CUDA-Q, dynamic circuits, GPU optimization, quantum-classical hybrid algorithms
- Builds on validated foundation (Milestone 1)
- Clear roles and accountability

### Expected Outcomes

**If Successful:**
- Demonstrate quantum advantage for N=30-40 LABS instances
- Achieve 10× speedup over tutorial baseline
- Publish open-source implementation for community
- Contribute novel insights on combining ancilla reuse strategies

**If Partially Successful:**
- Even without dynamic circuits, SQUARE provides significant improvement
- Even without full GPU acceleration, quantum-enhanced populations work
- Modular design allows publishing working components

**Learning Value:**
- Real experience with quantum-classical hybrid algorithms
- GPU optimization for both quantum and classical code
- Research skills: literature review → implementation → validation
- Team collaboration under resource constraints

---

## Appendix: Technical References

### Key Equations

**LABS Energy Function:**
```
E = Σ_{k=1}^{N-1} [Σ_{i=0}^{N-k-1} s_i · s_{i+k}]²
```

**Active Quantum Volume (SQUARE):**
```
AQV = Σ_{q∈Q} Σ_{(ti,tf)∈Tq} (tf - ti)
```

**Cost-Effective Reclamation (SQUARE):**
```
C1 = Nactive × Guncomp × S × 2^ℓ  (cost of uncompute)
C0 = Nanc × Gp × S × √((Nactive + Nanc)/Nactive)  (cost of not uncomputing)

Decision: Uncompute if C1 ≤ C0
```

### Code Repositories

- **Milestone 1 Validation:** [Link to completed notebook]
- **CUDA-Q Examples:** https://nvidia.github.io/cuda-quantum/
- **CuPy Documentation:** https://docs.cupy.dev/
- **SQUARE Reference Implementation:** (adapt from paper algorithms)

### Contact Information

**Project Lead (Sezer):** [sezeraptourachman@gmail.com] - Architecture & strategy  
**GPU PIC (Nicholas):** [ng.contact.secure@gmail.com] - GPU optimization & resource management  
**QA PIC:** [abrgalib@gmail.com] - Testing & validation  
**Marketing PIC:** [sezeraptourachman@gmail.com] - Visualizations & presentation

---

**Last Updated:** [31-01-2026]  
**Version:** 1.0  
**Status:** Ready for Review
