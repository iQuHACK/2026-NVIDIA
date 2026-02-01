import numpy as np

def labs_energy_pm1(s):
    """
    Calculates the energy of a binary sequence s in {+1, -1}.
    E(s) = sum_{k=1}^{N-1} (sum_{i=1}^{N-k} s_i * s_{i+k})^2
    """
    N = len(s)
    total_energy = 0
    for k in range(1, N):
        ck = 0
        for i in range(N - k):
            ck += s[i] * s[i+k]
        total_energy += ck**2
    return total_energy

def bits01_to_pm1(bits):
    """Converts a bitstring of 0s and 1s to +1s and -1s."""
    return 1 - 2 * np.array(bits)

def run_validation(N=7, quantum_sample_bitstring=None):
    print(f"--- Starting Validation Suite (N={N}) ---")
    
    # 1. Small N Hand-Calculation Verification
    # For N=4, the sequence [1, 1, 1, -1] is a known Barker sequence with Energy = 2.
    s_barker = np.array([1, 1, 1, -1])
    e_barker = labs_energy_pm1(s_barker)
    print(f"[TEST 1] Barker Sequence (N=4): Expected Energy 2, Got {e_barker}.")
    assert e_barker == 2, "Energy calculation failed on known N=4 case."

    # 2. Check Symmetries
    # The LABS problem is invariant under bit-flip (s -> -s) and reversal.
    test_seq = np.random.choice([-1, 1], size=N)
    e_orig = labs_energy_pm1(test_seq)
    
    e_flip = labs_energy_pm1(-test_seq)
    e_rev = labs_energy_pm1(test_seq[::-1])
    
    print(f"[TEST 2] Symmetry Check - Bit-flip: {'PASSED' if e_orig == e_flip else 'FAILED'}")
    print(f"[TEST 3] Symmetry Check - Reversal: {'PASSED' if e_orig == e_rev else 'FAILED'}")

    # 3. Quantum-Classical Cross-Reference
    if quantum_sample_bitstring:
        # Convert bitstring (e.g., '101') to numerical array
        bits = [int(b) for b in quantum_sample_bitstring]
        s_quantum = bits01_to_pm1(bits)
        e_quantum = labs_energy_pm1(s_quantum)
        print(f"[TEST 4] Quantum Cross-Ref: Bitstring {quantum_sample_bitstring} has Classical Energy {e_quantum}")
    else:
        print("[TEST 4] Quantum Cross-Ref: Skipped (No sample provided)")

if __name__ == "__main__":
    run_validation()