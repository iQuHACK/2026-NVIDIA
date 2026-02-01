## 1. The Workflow

In this project, we used AI as an **engineering assistant rather than an autonomous code generator**.  
The overall workflow was **iterative and strictly human-in-the-loop**, with AI used to accelerate development, debugging, and optimization while all final decisions remained under human control.

### Role of AI (GPT)

GPT was primarily used to accelerate the development of the `classical_mts` (classical memory tabu search) implementation, with the following responsibilities:

- Translating high-level algorithmic ideas (tabu memory, neighborhood exploration, acceptance rules) into concrete and modular Python code.
- Assisting with **program debugging**, including interpreting runtime errors, diagnosing subtle logical bugs (e.g., incorrect tabu updates or abnormal convergence), and suggesting minimal, targeted fixes.
- Supporting **code refactoring and performance optimization**, such as restructuring the codebase for clarity and identifying bottlenecks like redundant energy computations.
- Helping design **verification logic** aligned with known algorithmic and physical invariants.

GPT’s outputs were treated strictly as suggestions. All AI-assisted changes were reviewed and validated by the human developers before being integrated.

### Role of the Human Developers

The human developers retained full control over:

- Algorithmic decisions, including the exact formulation of the `classical_mts` update rules, tabu constraints, aspiration criteria, and stopping conditions.
- Performance decisions, such as which parts of the algorithm to optimize and which optimizations were theoretically safe.
- Final validation of correctness through unit tests, physical invariants, and comparison against known baselines.
- Deciding when AI suggestions were incorrect or inappropriate and rewriting them manually.

No AI-generated code was accepted blindly; every modification was tested or revised based on observed behavior.

---

## 2. Verification Strategy

All AI-generated or AI-assisted code was validated using a combination of **unit tests, physical invariants, and baseline cross-checks**.  
The explicit goal was to catch AI hallucinations, incorrect assumptions, or silent logic errors.

The following unit tests were implemented in `tests.py`.

