Context:

Problem first (one line).



The Lowest Autocorrelation Binary Sequence (LABS) problem asks for a sequence

si∈{+1,−1}s_i \in \{+1,-1\}

si

​∈{+1,−1}, i=1,…,Ni=1,\dots,N

i=1,…,N, that minimizes

E(s)=∑k=1N−1Ck2,Ck=∑i=1N−ksisi+k.E(s)=\sum_{k=1}^{N-1} C_k^2,

\qquad

C_k=\sum_{i=1}^{N-k} s_i s_{i+k}.

E(s)=k=1

∑

N−1

​Ck

2

​,Ck

​=i=1

∑

N−k

​si

​si+k

​.

Low autocorrelation means the sequence “looks random” at all shifts.

What “memetic tabu search” means here

Think of it as tabu search (strong local search) + evolutionary ideas (memes).



Tabu search: aggressively improves a single sequence by local moves, while avoiding cycles using memory.

Memetic: instead of one run, you keep a population of good sequences, combine them, then locally optimize again.

So: global exploration from recombination, local exploitation from tabu.

Core components

1. Representation

A candidate is just a length-NN

N vector of ±1\pm 1

±1.



2. Objective function

Energy E(s)E(s)

E(s) above.



Efficient implementations keep all CkC_k

Ck

​ updated incrementally so a flip is cheap.

Key fact: flipping bit sjs_j

sj

​ changes all correlations CkC_k

Ck

​ where index jj

j participates.



You can update EE

E in O(N)O(N)

O(N), not O(N2)O(N^2)

O(N2

).

Reference: Bernasconi model for LABS.

3. Local move: single-bit flip

Neighborhood = all sequences obtained by flipping one bit.

For each position jj

j, compute the energy change ΔEj\Delta E_j

ΔEj

​.

Pick the best allowed move.

4. Tabu mechanism

This is the heart.



When you flip bit jj

j, forbid flipping it back for TT

T iterations (tabu tenure).

Tabu list = short-term memory of recent moves.

Aspiration rule:



Ignore tabu if the move produces a new global best.

Why this matters:



Prevents short cycles.

Forces exploration of “worse first” regions to escape local minima.

Classic reference: Glover (1989).

5. Tabu search loop (inner optimizer)

For one candidate sequence:



Initialize ss

s (random or recombined).

Repeat for KK

K steps:

Evaluate all single-bit flips.

Choose lowest ΔE\Delta E

ΔE move not tabu (or aspiration).

Apply move, update tabu list.

Return best sequence found during the run.

This alone already works well for LABS.

Where the memetic part comes in

Pure tabu = one trajectory.



Memetic tabu = many trajectories + recombination.

6. Population

Maintain a small elite set (e.g. 10–50 sequences).

Each individual is already locally optimized by tabu search.

7. Recombination (the “meme”)

Typical LABS choices:



Uniform crossover: for each bit, pick parent A or B.

Block crossover: copy contiguous segments.

Majority vote (3 parents): bit = sign of sum.

This preserves low-energy structure (“good memes”).

After recombination, the child is not optimal → run tabu search again.

8. Replacement strategy

Keep the best sequences:



Remove duplicates or very similar sequences (Hamming distance).

Preserve diversity to avoid collapse.

Why this works especially well for LABS

LABS landscape is rugged with many deep local minima.

Tabu search is excellent at deep exploitation.

Recombination lets you jump between distant basins.

Together, they scale to N∼103N \sim 10^3

N∼103

, far beyond naive heuristics.

Analogy:



Tabu search is a skilled rock climber; memetics gives you a helicopter to new cliffs.

Canonical references

Bernasconi, Low autocorrelation binary sequences (1987).

Glover, Tabu Search (1989).

Gallardo et al., Finding low autocorrelation binary sequences with memetic algorithms (2007).

Prestwich, Generalized hill climbing for LABS.

One-sentence summary

Memetic tabu search for LABS alternates between tabu-optimized local minima and evolutionary recombination, giving both depth and reach in an extremely rugged energy landscape.