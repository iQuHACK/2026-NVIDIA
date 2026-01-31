'''“We verify correctness by (a) exact enumeration for small N, 
(b) symmetry tests mandated by the LABS Hamiltonian (global spin-flip, 
reversal, and their combination), 
and (c) cross-checks between classical MTS and the 
quantum algorithm on small instances. 
This mirrors standard practice in quantum-classical algorithm validation where 
invariants and small-instance exact checks are used as the ground-truth proxy 
when a solution key is not available.” '''
# CELL: LABS energy function (canonical)
import numpy as np

def labs_energy(s):
    """Compute LABS energy for a sequence s of +/-1 (list or 1D numpy array)."""
    s = np.asarray(s, dtype=int)
    N = s.size
    E = 0
    for k in range(1, N):
        Ck = int(np.sum(s[:N-k] * s[k:]))   # autocorrelation at shift k
        E += Ck * Ck
    return int(E)


# brute force enumeration for small N
import itertools, pandas as pd

def brute_force_labs(N):
    energies = {}
    for bits in itertools.product([-1, 1], repeat=N):
        energies[bits] = labs_energy(bits)
    # return sorted list of (sequence, energy)
    return sorted(energies.items(), key=lambda t: t[1])

# run for N=5
N = 5
bf = brute_force_labs(N)
print(f"Found {len(bf)} sequences; best energy = {bf[0][1]}")
# show top few
for seq, e in bf[:6]:
    print(seq, e)


# symmetry positive tests (must pass)
def test_global_flip(s):
    assert labs_energy(s) == labs_energy([-x for x in s]), "Global flip symmetry failed"

def test_reversal(s):
    assert labs_energy(s) == labs_energy(list(reversed(s))), "Reversal symmetry failed"

def test_flip_and_reverse(s):
    assert labs_energy(s) == labs_energy([-x for x in reversed(s)]), "Flip+reverse symmetry failed"

# test on random and canonical sequences
import random
for N in [3,4,5,6]:
    for _ in range(10):
        s = [random.choice([-1,1]) for _ in range(N)]
        test_global_flip(s)
        test_reversal(s)
        test_flip_and_reverse(s)
print("Positive symmetry tests passed on random samples.")

#negative tests to catch incorrect assumptions (eg, accidental cyclic autocor)
def cyclic_rotate(s, r):
    """Cyclic rotate right by r"""
    r = r % len(s)
    return s[-r:] + s[:-r]

# pick a random sequence and check rotation changes energy (for linear LABS)

s = [1, -1, 1, 1, -1]  # example
E_orig = labs_energy(s)
E_rot = labs_energy(cyclic_rotate(s, 1))
print("orig:",s,"E=",E_orig, "rotated E=", E_rot)
assert E_orig != E_rot, "Rotation did not change energy check whether you accidentally implemented circular autocor!"

# cross-check classical vs quantum (small N)
# integrate after phase 1 is completed
'''def compare_solvers(N):
    bf = brute_force_labs(N)
    best_exact_e = bf[0][1]
    # run MTS (classical baseline) - placeholder: use your notebook's MTS call
    mts_seq, mts_e = run_MTS(N)            # must return (sequence, energy)
    # run quantum procedure (placeholder)
    q_seq, q_e = run_quantum(N)            # must return (sequence, energy)
    print(f"N={N}: exact={best_exact_e}, MTS={mts_e}, Quantum={q_e}")
    return best_exact_e, mts_e, q_e

# Example call (only for N small where brute force is available)
N = 6
compare_solvers(N)
'''

# build orbit under dihedral-like ops
def dihedral_orbit(s):
    #the 'shape' is a 1D sequence, so we can't use  rotations
    s = list(s)
    ops = [
        lambda x: x,
        lambda x: [-a for a in x],             # flip
        lambda x: list(reversed(x)),           # reverse
        lambda x: [-a for a in reversed(x)]    # flip+reverse
    ]
    return {tuple(op(s)) for op in ops}

# example
s = [1,-1,1,1]
#get all transformations
orbit = dihedral_orbit(s)
print("orbit size:", len(orbit))
#print energy, should be the same
for seq in orbit:
    print(seq, labs_energy(seq))

# CELL: small test suite runner
def run_all_tests():
    # smoke tests for small Ns using brute force sequences
    for N in range(3,7):
        print(f"Testing all symmetries on all {2**N} sequences for N={N}")
        bf = brute_force_labs(N)
        for seq, e in bf:
            test_global_flip(seq)
            test_reversal(seq)
            test_flip_and_reverse(seq)
    # negative test sample
    s = [1, -1, 1, 1, -1, 1]
    assert labs_energy(s) != labs_energy(cyclic_rotate(s, 1))
    print("All tests passed.")
    
# Run the tests
run_all_tests()

