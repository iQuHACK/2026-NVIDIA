# NVIDIA Challenge: Solving the LABS problem with a Quantum Enhanced and GPU Accelerated Workflow
## Phase 1
---

### Milestone 1: The Ramp Up (Scaffolded Tutorial)

*Goal: Understand the LABS problem and the baseline algorithms.*

Your first task is to complete the provided scaffolded Jupyter Notebook [01_quantum_enhanced_optimization_LABS.ipynb](https://github.com/iQuHACK/2026-NVIDIA/blob/main/tutorial_notebook/01_quantum_enhanced_optimization_LABS.ipynb). This notebook guides you through:

1. Understanding the LABS symmetry and problem definition.
2. Running a classical Memetic Tabu Search (MTS).
3. Implementing a **Digitized Counterdiabatic Quantum Algorithm** using CUDA-Q.
4. Creating a hybrid workflow where the quantum algorithm seeds the Classical MTS.

*The Verification Challenge:* We are not providing an answer key. In scientific R&D, you rarely know the ground truth. You must prove to yourself (and us) that your code works before you build upon it.

*The Requirement:* You must add a "Self-Validation" section to the notebook. In this section, explain how you verified your results. Did you calculate solutions by hand for small N? Did you create unit tests?  Did you cross-reference your Quantum energy values against your Classical MTS results? Did you check known symmetries?

**Deliverable 1:** A completed, executable version of the `01_quantum_enhanced_optimization_LABS.ipynb` notebook. This must include your new "Self-Validation" section containing code or text explaining how you verified your baseline results (e.g., manual calculation for N=3, unit tests for symmetries, or brute-force comparison for small N).

---

### Milestone 2: Research and Plan

*Goal: Define the Custom Solver, Plan the Acceleration, and Assign Roles.*

Your objective for the remainder of the challenge is to evolve the tutorial code in Milestone 1 into a high-performance, custom solver. You must plan for two specific technical projects:

1. **The Custom Quantum Seed:** Identify and implement a *different* quantum algorithm (e.g., QAOA, VQE, a custom Ansatz, etc.) or a variation of the Counteradiabatic approach to replace the Counterdiabatic approach from Milestone 1 using CUDA-Q.
2. **Full GPU Acceleration:** Design a workflow that accelerates **both** the Quantum Algorithm (using CUDA-Q) and the Classical MTS on NVIDIA GPUs.

**Warning:** Do not rush this phase. The Delveriables from Milestone 1 along with the **Product Requirements Document (PRD)** you submit accounts for **40% of your final grade**.

### **Section 1: The Artifact (What is a PRD?)**

Think of your team as an early-stage startup. Your provided GPU credits are your Seed Funding. You have a finite "runway." If you burn through your budget on inefficient experiments, or the common novice mistake of leaving a "zombie" GPU instance running overnight, your venture halts before you can deliver your product.

In this challenge, your Product Requirements Document (PRD) is your safeguard against that fate. It is your business case for compute resources. Because your GPU credits are finite, you cannot afford to "move fast and break things." You must "plan fast and build correctly."

Your PRD defines exactly what you are building and how you will verify it works. It demonstrates that you are ready to use high-performance infrastructure like a professional: with intent, precision, and efficiency.

**A Note on Agility:** While you will submit this PRD early in the hackathon to unlock your compute credits and to be graded, adjustments to the PRD later in the hackathon may be required to ensure on-time delivery. If you pivot, you do not need to resubmit a new plan to the judges, simply communicate those changes to your teammates, and reflect on these pivots in the final presentation.

### **Section 2: Assign Your Technical Roles**

High-performance engineering teams do not work in silos. While every member will write code, debug, and contribute to the strategy, you need clear lines of accountability to prevent chaos.

Assign the following roles to act as the "Person in Charge" (PIC) for each domain. Being a lead does not mean you work alone; it means you orchestrate the team's effort in that area.  If you have more or less than 4 members of the team, some individuals may take on more than one role or some teammates may share a role.

* **Project Lead**
     * **Role:** You make the final decision on the algorithmic approach. You manage time, resources, and team communication. You are responsible for keeping the PRD updated if the strategy changes.
     * **Deliverable:** You own the **PRD** (Milestone 2) and are responsible for the **Final Submission Logistics** (ensuring all files, reports, and checklists are present and submitted on time).


* **GPU Acceleration PIC**
     * **Role:** You lead the GPU acceleration. You are the bridge between the code and the hardware.
     * **Deliverable:** You own the migration to Brev and the selection of GPU architectures. Crucially, you are responsible for Resource Management: you must decide which GPU to use (e.g., L4 vs. A100) based on cost-efficiency and ensure the team does not burn through the $20 credit by leaving instances running idle ("Zombie Instances").


* **Quality Assurance PIC**
     * **Role:** You are responsible for verifying that the human and the AI-generated code are correct. You protect the team from "AI Hallucinations."
     * **Deliverable:** You own the **Verification Strategy** and the **Unit Test Suite** (`tests.py`).


* **Technical Marketing PIC**
     * **Role:** You are the Analyst. You translate raw logs into insight. You are responsible for analyzing the run-data and proving the team's success through data visualization.
     * **Deliverable:** You own the **Final Presentation** and the **Success Metric Visualizations** (Milestone 4).



### **Section 3: Define Your Verification Strategy**

**Great engineering is not about guessing; it is about intent.** As part of your PRD, the QA PIC must commit to a verification strategy *before* you prompt your AI agents to write a single line of code.

> **A Note for the QA PIC: From "Sanity Check" to "System Check"**
> 
> In Milestone 1, we asked you to explain *how* you convinced yourself your tutorial code was correct. You may have conducted ad-hoc "Sanity Checks" (e.g., printing a small array and nodding your head that the result looked correct).
> **Now, you must professionalize that instinct.**
> In the next milestone (Build), you cannot rely on manual print statements. You must implement automated **Unit Tests**.
> * **For Example:** Instead of manually looking at `N=3` output, you will write a script that *asserts* `calculate_energy([1, -1, 1]) == 1.0` or `energy([1, -1]) == energy([-1, 1])`.
> * **Why?** You are about to use AI to generate code. If you don't have a reliable test suite, you will have no way to distinguish a brilliant code from a subtle hallucination. You need a way to check your work every time you make a change.
> * **Resource:** [Getting Started with Testing in Python](https://realpython.com/python-testing/).
> 
> 

### **Section 4: The Research Requirement**

To fill out your PRD, you must do the homework. You cannot simply guess a solution. You must answer:

* **Choice of Quantum Algorithm and Motivation:** Why did you choose this specific algorithm to challenge the tutorial's Counterdiabatic approach? Your reasoning can be metric-driven (expecting better Time to Solution, Approximation Ratio, circuit depth, etc.) or exploratory (e.g., selecting a foundational algorithm like QAOA to gain mastery, or testing a novel ansatz). Regardless of the motivation, you must cite papers to explain the theoretical interest of this algorithm.
* **GPU Acceleration:** How exactly will you accelerate the quantum and/or classical portions of the algorithm? (e.g., Will you use CUDA-Q's GPU-accelerated backends? Will you replace `numpy` with `cupy`? Will you explore batch-processing neighbors?) 

### **Section 5: Define Execution Tactics**

Finally, define the operational details of your workflow:

* **Agentic Workflow:** How will you orchestrate your AI agents? (e.g., "Agent A generates tests; Agent B refactors for CUDA").
* **Success Metrics:** Define your targets. (e.g., "We are targeting an Approximation Ratio > 0.85 for N=20," or "We aim to reduce the Classical Tabu Search runtime by 50% using CuPy for N>30").
* **Resource Management:** How will you manage your workflow so that you do not deplete your Brev credits? (e.g., "We'll set an alarm every 30 minutes to check that we don't have any instances running idly in the background." "We will test on an L4 GPU before attempting to run on an A100"). How will you allocate your $20 credit? Create a rough estimate. (e.g., "5 hours of dev on L4 ($5.00) + 4 hours of heavy benchmarking on A100 ($8.00) + Buffer ($4.00).")

---
## References

* [Scaling advantage with quantum-enhanced memetic tabu search](https://arxiv.org/html/2511.04553v1)
* [Parallel MTS by JPMorgan Chase](https://arxiv.org/pdf/2504.00987)
