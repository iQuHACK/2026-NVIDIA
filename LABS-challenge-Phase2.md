# NVIDIA Challenge: Solving the LABS problem with a Quantum Enhanced and GPU Accelerated Workflow
## Phase 2
---

### Milestone 3: Build

*Goal: Implement, Test, and Accelerate.*

This milestone is broken into three three steps:

#### Step A: CPU Validation

Implement your chosen custom quantum algorithm (from the PRD in Milestone 2) using CUDA-Q, running on a CPU backend in .

* **Requirement:** Demonstrate that your custom quantum-enhanced workflow finds valid solutions for small (e.g. N=3 to N=10).
* **Verification:** You must include a rigorous test suite (`tests.py`) covering your energy functions and quantum kernels.

#### Step B: GPU Acceleration and Hardware Migration

Now that your logic is proven on qbraid's CPU backend, it is time to scale.

**The Migration:** Provision an instance on Brev. You must select a specific NVIDIA GPU architecture (we recommend starting with an L4 or T4) to serve as your target backend.
Next, migrate your code, switch your CUDA-Q target to the NVIDIA GPU backend, and begin your acceleration benchmarks.

Optional:  Rewrite your code for the quantum algorithm to exploit additional opportunities for parallelization.  Experiment with different GPUs or multiple GPUs.

* **Requirement:** Scale up. Run your simulation for larger N.  
* **Metrics:** Compute your success metrics as defined in your PRD and compare them to the CPU run (e.g., the time-to-solution and approximation ratio)

#### Step C: GPU Acceleration of the classical algorithm

Carry out your GPU acceleration strategy to accelerate the classical MTS algorithm.  This might have been to replace numpy with cupy, or to rewrite the code to exploit further opportunities for parallelization.  Put this together with your work in step B.

* **Goal:** A fully GPU-accelerated workflow with comparisons made to the CPU-only runs and to the example from Milestone 1.

* **Tip:** Occasionally, refer back to your PRD.md to stay on track and/or make changes to your overall strategy based on progress.

---

### Milestone 4: Showcase & Retrospective

*Goal: Communicate your results and your AI strategy.*

The final phase is about packaging your work. You will submit your code, a presentation, and a detailed report on how you utilized Artificial Intelligence.

#### Part A: The AI Post-Mortem Report

We want to see how you leveraged AI to accelerate your development and, crucially, how you verified it.

**Format:** `AI_REPORT.md`
> **Note to Students:** > The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist. 
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have employed AI agents in a thoughtful way.

**Required Sections:**

1. **The Workflow:** How did you organize your AI agents? (e.g., "We used a Cursor agent for coding and a separate ChatGPT instance for documentation").
2. **Verification Strategy:** How did you validate code created by AI?
* *Requirement:* You must describe specific **Unit Tests** you wrote to catch AI hallucinations or logic errors.


3. **The "Vibe" Log:**
* *Win:* One instance where AI saved you hours.
* *Learn:* One instance where you altered your prompting strategy (provided context, created a skills.md file, etc) to get better results from your interaction with the AI agent.
* *Fail:* One instance where AI failed/hallucinated, and how you fixed it.
* *Context Dump:* Share any prompts, `skills.md` files, etc. that demonstrate thoughtful prompting.



#### Part B: The Presentation

You must produce a cohesive story of your project.

* **If In-Person:** A live 5-10 minute presentation to the judges.
* **If Remote:** A 5-10 minute voice-over slide deck (MP4 format) included in your final submission.

**Example Presentation Agenda:**

1. The Plan & The Pivot:

* Briefly outline your original strategy from the PRD. Did you stick to the plan? If you had to change your algorithm or acceleration strategy mid-stream, explain why. Engineering is about adaptation; tell us about the obstacles you overcame.

2. Results: 

* Present your results based on the Success Metrics defined in your PRD. For example, use charts to visualize your Time-to-Solution and Approximation Ratio compared to the baseline.

3. The Retrospective:

* Conclude with a personal touch. Each team member should share one specific technical or strategic takeaway from the challenge (e.g., "I learned that moving data to GPU is slower than computing on CPU if the batch size is too small").


