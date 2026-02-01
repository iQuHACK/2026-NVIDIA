# NVIDIA iQuHACK 2026 Challenge

## Overview

The Low Autocorrelation of Binary Sequences (LABS) problem is a notoriously difficult optimization challenge, critical for high-performance radar and telecommunications.

Your objective is to take the current classical state-of-the-art Memetic Tabu Search (MTS) and evolve it. Rather than jumping to a purely quantum solution, you will engineer a hybrid quantum-enhanced workflow where samples from a quantum algorithm are used to seed the classical MTS population. You must then push the limits of performance by GPU-accelerating both the quantum simulation and the classical search components.

**We want you to vibe code!** 

In modern R&D and this challenge, speed matters, but rigor and coordination matter more. We expect you to employ Agentic Strategies, utilizing AI tools that can reason across your codebase to act as your collaborators while you operate as the Technical Leadership Team. Your collective job is to decompose the problem, delegate tasks across your team and AI agents, and most importantly verify the work. As Leads, you must clearly communicate your planning, workflow, and solution, ensuring your team remains aligned and ready to pivot even as technical challenges shift your strategy.

## Logistics, Milestones, and Evaluation

![LABS Challenge Timeline](images/image-3.png)

In this challenge, you will mimic a real-world R&D pipeline, moving from rapid prototyping to high-performance deployment. Success in this challenge requires clear orchestration; review the section [Assign Your Technical Roles](https://github.com/iQuHACK/2026-NVIDIA/blob/main/LABS-challenge-Phase1.md) in the GitHub repository to define your team's PICs for Project Lead, GPU Acceleration, Quality Assurance, and Technical Marketing. These roles ensure your team remains aligned while managing both human and AI-agent workflows.

During the challenge, you will utilize two distinct platforms, each chosen for a specific phase of your development lifecycle. 

* **Phase 1 (Prototyping): qBraid** | [Specifications](https://github.com/iQuHACK/2026-NVIDIA/blob/main/LABS-challenge-Phase1.md)

    For the "Ramp Up" and initial CPU validation, you will work on Milestones 1 and 2 in [qBraid](https://account-v2.qbraid.com/). qBraid is your "Dev Environment" — a zero-setup, pre-configured cudaq sandbox that allows you to focus entirely on mastering the algorithm and logic without worrying about infrastructure overhead.

    ### Milestones:
    1. **Ramp Up**: Master the state-of-the-art for LABS via a scaffolded tutorial.
    
    2. **Research & Plan**: Perform due diligence to design a custom quantum strategy and acceleration plan.

* **Phase 2 (Acceleration): Brev** | [Specifications](https://github.com/iQuHACK/2026-NVIDIA/blob/main/LABS-challenge-Phase2.md)

    Once your logic is validated, you will "graduate" your code to [Brev](https://brev.nvidia.com/) to complete Milestone 3 and 4. Brev provides on-demand access to a wide variety of NVIDIA GPU architectures (L4s, T4s, A100s, ...). We have provided a pre-configured GPU environment for this environment called a **Launchable**. You will use Brev to test your solution across different hardware configurations and unlock full GPU acceleration.

    ### Milestones: 
    3. **Build**: Inmplement your algorithm in CUDA-Q and validate it on a CPU in qBraid, then migrate to Brev to deploy full GPU acceleration.

    4. **Showcase and Retrospective**: Present your solution, performance metrics, and your AI-driven workflow.

**Good luck. Let the agents build the code, you build the architecture.**

## Accessing Phase 1 of the Challenge with qBraid

<a href="https://account-v2.qbraid.com/explore/projects/iquhack-nvidia" target="_parent"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" alt="Launch On qBraid" width="150"/></a>

During the duration of the hackathon, you will have access to the new and improved version of the qBraid platform accessible through here: https://account-v2.qbraid.com/ 

If any issues occur, try deleting cache in your browser and refreshing the page.

### Steps for qBraid Environment Setup:

1. Click the `Launch on qBraid` button above and create an account 

2. Click on the `Launch on Lab` button on the right side of the explore page. This will automatically clone your repository and add the CUDA-Q environment. 

<img align="right" width= "28%" src="images/image.png">

3. Double check that you have `CUDA-Q (v0.13.0)` installed. 
    * If it's **NOT** installed follow Steps 4-6 
    * Otherwise, go directly to Step 7

4. Add the CUDA-Q environment by navigating to the ENVS tab in the right sidebar, and click on `+ ADD` 

5. Navigate to `CUDA-Q and GPU Quantum Environments` 

<img align="right" width ="28%" src="images/image-2.png">

6. Install CUDA-Q (v0.13.0)

7. Once installation is complete, open the challenge notebook `01_quantum_enhanced_optimization_LABS.ipynb` under the `tutorial_notebook` directory 

8. In the bottom left corner, make sure the kernel is set to `Python 3 [cuda q-v0.13.0]` 

9. Happy Hacking!


For any questions or additional assistance using qBraid, see the [v2 platform documentation](https://docs.qbraid.com/v2/home/introduction), or reach out to the qBraid team on discord.


## Accessing Phase 2 of the Challenge with Brev

<a href="https://brev.nvidia.com/launchable/deploy/now?launchableID=env-392R2ank8ts8FywogpAGLsQYqu8" target="_parent"><img src="https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg" width="150"/></a>

Congratulations on finishing the first part of the challenge! 

You will now get to run your code on real GPUs using NVIDIA's Brev Platform. Don't worry, you don't need to pay for anything! Once completing Phase 1 and your logic is validated, we will provide your team with a **$20 Brev coupon code**.

The desginated GPU Acceleration PIC on your team is responsible for majority of the qBraid to Brev migration. Go to the `GPU_PIC_Brev.pdf` document [here](https://github.com/iQuHACK/2026-NVIDIA/blob/main/GPU_PIC_Brev.pdf) and follow the instructions.

For any questions or additional assistance using Brev, see the [Brev Console Reference](https://docs.nvidia.com/brev/latest/console.html) or reach out to the NVIDIA team on discord.

## Submission Instructions and Deadlines


> **By 11am eastern time on Sat Jan 31: Team Intent** The Project Lead should fork this repository and share the link along with the team member's discord names with the judges in a DM on discord (do not post to the Discord channel unless you want all the other participants to see your work).  

> **By 10pm eastern on Sat Jan 31, Phase 1 is due.  Submit what you have finished by that point to have your work assessed for access to your team credits.  

> **By 10am eastern on Sun Feb 1, Phase 2 is due.** The Project Lead should ensure that the forked repository contains all of the deliverables and should DM the judges to confirm their project completion.  Judges will use the forked repository for grading. 

Please see the [Deliverables Checklist](/team-submissions/README.md) in the team-submissions folder for more information about the evaluation criteria of each deliverable.

## Resources
### NVIDIA Jan 30, 2026 Worskhop Presentation
* [Slide deck](https://github.com/iQuHACK/2026-NVIDIA/blob/main/NVIDIA-presentation-slides-MIT-IQuHack.pdf)

### CUDA-Q    

* If you are new to quantum computing, (e.g., you'd like a review of the definition of a qubit, quantum gates, and quantum circuits), then run through the first two notebooks of the [Quick Start to Quantum Computing](https://github.com/NVIDIA/cuda-q-academic/tree/main/quick-start-to-quantum) series.

* If you have quantum computing background and want a quick visual guide for translating a quantum circuit into a CUDA-Q kernel, check out this [hello world visualization tool](https://nvidia.github.io/cuda-q-academic/quick-start-to-quantum/interactive_widget/cudaq-hello-world.html).  For more in depth coverage of cuda-q syntax and examples, we recommend notebook 1 of the [QAOA for Max Cut series](https://github.com/NVIDIA/cuda-q-academic/tree/main/qaoa-for-max-cut) series as well as the [examples](https://nvidia.github.io/cuda-quantum/latest/using/examples/examples.html) and [applications](https://nvidia.github.io/cuda-quantum/latest/using/applications.html) in the CUDA-Q documentation, in particular the [QAOA example](https://nvidia.github.io/cuda-quantum/latest/applications/python/qaoa.html).  

* If you prefer to learn by watching, you can check out [minutes 30:40-38:28 of this demo](https://www.nvidia.com/en-us/on-demand/session/gtcdc25-dct51159/?playlistId=gtcdc25-quantum-computing-and-hpc&start=1840&end=2308) of description of cudaq kernels, sampling, and getting the state vector or watch [minutes 9:20-19:00 of this demo](https://www.youtube.com/live/DqPC-nlcXKA?si=ualhUnFYjW9BlbQz&t=560).

### CODA
* Conductor Quantum is providing 100 credits to their [CODA Platform](https://conductorquantum.substack.com/p/coda-natural-language-quantum-computing?utm_campaign=post&utm_medium=web&triedRedirect=true%2B) for the first 100 teams that complete Phase 1 of the NVIDIA challenge!

* CODA [Demo](https://youtu.be/MocBUZlQlmU?si=c2GXTKb1dosla4PW)
* CODA [MCP](https://pypi.org/project/coda-mcp/)

### Vibe Coding Tips and Resources
* Implement verification tests as part of your workflow (e.g., [unittests](https://docs.python.org/3/library/unittest.html), [pytest](https://docs.pytest.org/en/stable/)

* Introduce context into your prompting (e.g., [context7](https://context7.com), agent skills (`skills.md`), MCP, etc.)

* Use different [prompting strategies](https://www.promptingguide.ai/techniques)

* Keep prompts short; open up new chats occassionally to avoid the context window getting too large
* Meta vibe code. Take some pulse checks occassinoally as you're working by feeding your session back into an agent and ask them how you could improve your prompting.
* [Cursor & The "Vibe" Workflow](https://youware.medium.com/cursor-for-vibe-coding-a-complete-guide-b0863d4a2330)
* [Responsible AI Coding: Best Practices](https://cloud.google.com/blog/topics/developers-practitioners/five-best-practices-for-using-ai-coding-assistants)

## Accessing Material Post Challenge

Challenge materials can be accessed via https://account-v2.qbraid.com/ or in the coming weeks through https://account.qbraid.com/ once v2 is merged into the main platform.
Make sure to push your changes on qBraid to your respective repository forks as a backup.

If you completed Phase 2 of the challenge, materials can be accessed via https://brev.nvidia.com/.
