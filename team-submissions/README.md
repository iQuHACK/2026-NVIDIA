## Deliverables Checklist

**Step 0: (Due by noon eastern Sat Jan 31)**
The Project Lead should fork this repository and share the forked repository link with the NVIDIA judges in a DM on discord (`Monica-NVIDIA`, `Linsey-NV Mentor`, and `Iman_nvidia`).  

**Phase 1 Submission (Due 10pm eastern Sat Jan 31):**
(To be judged to obtain access to GPUs with Brev credits)
* [ ] **Tutorial Notebook:** Completed [01_quantum_enhanced_optimization_LABS.ipynb](https://github.com/iQuHACK/2026-NVIDIA/blob/main/tutorial_notebook/01_quantum_enhanced_optimization_LABS.ipynb) including your "Self-Validation" section.
* [ ] **PRD:** `PRD.md` defining your plan.  See [Milestone 1 in the challenge description]([LABS-challenge-Phase1.md](https://github.com/iQuHACK/2026-NVIDIA/blob/main/LABS-challenge-Phase1.md)) and the [PRD-template.md](PRD-template.md) file.
* [ ] **Notify the judges in discord:** DM `Monica-NVIDIA`, `Linsey-NV Mentor`, and `Iman_nvidia` that your phase 1 deliverables are ready to be judged.   

**Phase 2 Submission (Due 10am eastern Sun Feb 1):**

* [ ] **Final Code:** Notebooks and scripts for the Milestone 3 implementation. Add these files to a folder in this directory.  Include a README.md to guide the judges if you have multiple files.
* [ ] **Test Suite:** `tests.py` or other documentation or scripts used for verifying your work. 
* [ ] **AI Report:** `AI_REPORT.md` (See [AI_REPORT-template.md](AI_REPORT-template.md)).
* [ ] **Presentation:** Slides (Live) or MP4 (Remote).

## Evaluation Criteria: How You Will Be Graded

This challenge is graded on **Rigorous Engineering**, not just raw speed. We are simulating a professional delivery pipeline. Your score is split between your planning (40%) and final product(60%).

### Phase 1: The Plan (40% - Early Submission)
*Goal: Prove you have a viable strategy before we release GPU credits.*

* **The Self-Validation from Milestone 1 [5 points]:** Did you prove your tutorial baseline was correct? We look for rigorous checks (e.g., symmetry verification, small N brute-force) rather than just "it looks right."
* **The PRD Quality from Milestone 2 [35 points]:**
    * **Architecture:** Did you cite literature to justify your quantum algorithm choice? Deep research drives high scores. We explicitly reward teams that take Scientific Risks over those who play it safe.
    * **Acceleration:** Do you have a concrete plan for GPU memory management and parallelization?
    * **Verification:** Did you commit to specific physical correctness checks (e.g., `energy(x) == energy(-x)`)?
    * **Success Metrics:** Did you define quantifiable targets?


### Phase 2: The Product (60% - Final Submission)
*Goal: Prove your solution works, scales, and is verified.*

* **Performance, Scale, and Creativity [20 points]:**
    * Does the solution scale to larger $N$?
    * Did you successfully accelerate the *Classical* component (e.g., using `cupy` for batch neighbor evaluation) or did you only accelerate the quantum circuit?
    * We reward the **rigorous implementation of creative ideas**. If your novel experiment fails to beat the baseline, document *why* in your report. A "Negative Result" backed by great engineering is a success.
    * Make sure to document which GPUs you used for each of your results for fair comparisons
* **Verification [20 points]:**
    * How much of your code is covered by the `tests.py` suite?
    * Does the test suite catch physical violations?
* **Communication & Analysis [20 points]:**
    * **Visualizations:** We expect professional data plotting. Do not just paste screenshots of code. We want to see generated plots (Time vs. N, Approximation Ratio vs. N) with clearly labeled axes comparing CPU vs. GPU performance.
    * **The Narrative:** Your presentation must tell the story of "The Plan & The Pivot." Did you identify *why* you had to change your strategy?
    * **The AI Report:** Your `AI_REPORT.md` must demonstrate *how* you verified AI-generated code, including specific examples of "Wins" and "Hallucinations."

> For submissions, please place the deliverables in the `teams-submissions` directory of your forked repository and notify the judges.
