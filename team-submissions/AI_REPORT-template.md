

 1. The Workflow (How we organized AI agents)

 Roles / agent separation
    We used Co-pilot for bug fixing so we can fix runtime errors quickly (missing imports, undefined symbols, cell-order issues)
    and to implement workflow edits (CUDA-Q kernels, QAOA loop, seeding pipeline, logging).
    We used CharGPT for:
    - how to add symmetry canonicalization without breaking correctness,
    - how to evaluate improvement fairly (multi-run win rate, matched MTS budgets).


How we avoided stepping on each other:
 We have a strict order for things:
  1, make it run (compile + sample),
  2, make it correct (unit tests),
  3, make it better (improved search, algorithm...),
  4, prove it’s better (benchmark).


2. Verification Strategy (How we validated AI-written code)
 we primarly focused on:

   (1) Symmetry invariance tests (LABS physics/property checks)
   -Global flip invariance: E(s) == E(-s)
   - Reversal invariance E(s) == E(reverse(s))
   - Combined invariance: E(s) == E(-reverse(s))


   (2) Bitstring ↔ sequence conversion tests
  - Assert output length matches N.
  - Assert values are only in {+1, -1}.
  - Assert mapping is consistent and stays consistent across the notebook.


    (3). Canonicalization correctness tests (equivalence class)
  - For random s, define the set: {s, -s, rev(s), -rev(s)}
  - Assert canon(x) is identical for every x in that set.
  - Assert canon(canon(s)) == canon(s) (idempotence).
  - Assert canonicalization does not change the energy: E(canon(s)) == E(s).

    (4) Brute-force ground truth (small N)
     For N ≤ 12:
       - Enumerate all 2^N sequences.
       - Compute exact optimum energy and known best sequences.
     - Assert:
       - reference energy function returns the known optimum on the known best,
       - heuristic outputs are within a reasonable bound under fixed budgets (optional but useful).

    (5) QAOA objective consistency test
   - We added a test/check:
  - Take chosen params (γ, β),
  - sample circuit to bitstrings to sequences,
  - compute sample mean of compute_energy,
  - assert it matches the objective value used by the optimizer (within shot noise).

3. The Vibe Log

Win:
We used the coding agent to refactor the notebook into:
- a QAOA seeding pipeline,
- symmetry canonicalization,
- CVaR-style objective option,
which saved us a lot of time

Learn:
We used Ai to generate prompts for Ai:
which gave us a structured prompt with:
explicit constraints, exact deliverables and acceptance checks.


Fail:
Missing imports (random) / missing helper (get_interactions)
   -we fixed it by:
     centralizing imports at the top,
     creating a “Run me first” cell with definitions and imports.

