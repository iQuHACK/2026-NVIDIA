import argparse
import random
import numpy as np
import time
import math
import json



import cudaq
from cudaq import spin
import cudaq_solvers as solvers
# from cudaq_solvers.gqe_algorithm.gqe import get_default_config
from src.GQEMTS.gqe import get_default_config

parser = argparse.ArgumentParser(description="LABS: GQE-mts")
# parser.add_argument("--target", type=str, default="cpu", choices=["cpu", "gpu", "mqpu"])
parser.add_argument("--target", type=str, default="cpu", choices=["cpu", "gpu"])
parser.add_argument("--output", type=str, default="results.json")
parser.add_argument("--n_target", type=int, default=12)
parser.add_argument("--n_train", type=int, default=10)


parser.add_argument("--shots", type=int, default=1000)
parser.add_argument("--keep_ratio", type=float, default=0.5)
parser.add_argument("--depth_levels", type=int, default=4)
parser.add_argument("--gqe_iters", type=int, default=15)
parser.add_argument("--gqe_ngates", type=int, default=5)

parser.add_argument("--pop_size", type=int, default=30)
parser.add_argument("--mts_iters", type=int, default=1000)
parser.add_argument("--tabu_len", type=int, default=10)
parser.add_argument("--tabu_steps", type=int, default=1000)
parser.add_argument("--p_crossover", type=float, default=0.7)
parser.add_argument("--p_mut", type=float, default=None)
args = parser.parse_args()

if args.target == "cpu":
    print("[Setup] target=qpp-cpu")
    cudaq.set_target("qpp-cpu")
elif args.target == "gpu":
    print("[Setup] target=nvidia")
    try:
        cudaq.set_target("nvidia")
    except Exception as e:
        print(f"[Setup] nvidia unavailable ({e}); fallback to qpp-cpu")
        cudaq.set_target("qpp-cpu")
# elif args.target == "mqpu":
#     print("[Setup] target=nvidia-mqpu")
#     try:
#         cudaq.set_target("nvidia-mqpu")
#     except Exception as e:
#         print(f"[Setup] nvidia-mqpu unavailable ({e}); run with mpiexec and ensure MPI is installed.")
#         raise SystemExit(1)



# LABS Hamiltonian (Z2 + Z4)
def get_interactions(N: int):
    G2, G4 = [], []
    for i in range(N - 2):
        for k in range(1, (N - i) // 2 + 1):
            G2.append([i, i + k])
    for i in range(N - 3):
        for t in range(1, (N - i - 1) // 2 + 1):
            for k in range(t + 1, N - i - t):
                G4.append([i, i + t, i + k, i + k + t])
    return G2, G4


def build_labs_ham(N: int):
    G2, G4 = get_interactions(N)
    H = 0
    for i, j in G2:
        H += spin.z(i) * spin.z(j)
    for i, j, k, l in G4:
        H += spin.z(i) * spin.z(j) * spin.z(k) * spin.z(l)
    return H



# CD Pool 
def make_cd_pool(N: int, depth_levels: int = 4, keep_ratio: float = 1.0, seed: int = 42):
    """
    Build a CD operator pool for LABS:
      - 2-body motifs: YZ and ZY over (i, j) pairs
      - 4-body motifs: one Y with three Z's over (i, j, k, l) quartets
      - coefficients: ± pi / 2^t, t=0..depth_levels-1
      - optional random subsampling by keep_ratio

    Returns:
      pool: List[cudaq.SpinOperator]
      meta: List[dict]  (same length as pool, without the 'op' field)
    """
    pairs, quads = get_interactions(N)
    scale_levels = [math.pi / (2**t) for t in range(depth_levels)]

    records = []


    two_body_templates = []
    for (p, q) in pairs:
        dist = q - p
        two_body_templates.append(("YZ", dist, cudaq.SpinOperator(spin.y(p) * spin.z(q))))
        two_body_templates.append(("ZY", dist, cudaq.SpinOperator(spin.z(p) * spin.y(q))))

    for pattern, dist, base_op in two_body_templates:
        for c in scale_levels:
            records.append({"op":  c * base_op, "arity": 2, "d": dist, "coef":  c, "pattern": pattern})
            records.append({"op": -c * base_op, "arity": 2, "d": dist, "coef": -c, "pattern": pattern})


    for (i, j, k, l) in quads:
        ds = (j - i, k - j, l - k)

        four_templates = {
            "YZZZ": spin.y(i) * spin.z(j) * spin.z(k) * spin.z(l),
            "ZYZZ": spin.z(i) * spin.y(j) * spin.z(k) * spin.z(l),
            "ZZYZ": spin.z(i) * spin.z(j) * spin.y(k) * spin.z(l),
            "ZZZY": spin.z(i) * spin.z(j) * spin.z(k) * spin.y(l),
        }

        for pattern, spin_prod in four_templates.items():
            base_op = cudaq.SpinOperator(spin_prod)
            for c in scale_levels:
                records.append({"op":  c * base_op, "arity": 4, "ds": ds, "coef":  c, "pattern": pattern})
                records.append({"op": -c * base_op, "arity": 4, "ds": ds, "coef": -c, "pattern": pattern})


    if seed is not None:
        random.seed(seed)
    if keep_ratio < 1.0:
        keep_cnt = max(1, int(len(records) * keep_ratio))
        records = random.sample(records, keep_cnt)

    pool = [r["op"] for r in records]
    meta = [{k: v for k, v in r.items() if k != "op"} for r in records]
    return pool, meta




def tile_features(chosen_ids, meta_src, N_big: int):
    coeff_list, word_list = [], []
    """
    Take learned motif metadata (gap/pattern) and tile along the chain to N_big.
    Return (coeff_list, pauli_word_list) for exp_pauli().
    """
    for idx in chosen_ids:
        f = meta_src[idx]
        c = f["coef"]
        pat = f["pattern"]

        if f["arity"] == 2:
            gap = f["d"]
            for i in range(N_big - gap):
                j = i + gap
                s = ["I"] * N_big
                s[i] = pat[0]
                s[j] = pat[1]
                coeff_list.append(c)
                word_list.append(cudaq.pauli_word("".join(s)))

        elif f["arity"] == 4:
            g1, g2, g3 = f["ds"]
            span = g1 + g2 + g3
            for i in range(N_big - span):
                j = i + g1
                k = j + g2
                l = k + g3
                s = ["I"] * N_big
                s[i] = pat[0]
                s[j] = pat[1]
                s[k] = pat[2]
                s[l] = pat[3]
                coeff_list.append(c)
                word_list.append(cudaq.pauli_word("".join(s)))

    return coeff_list, word_list



# Classical MTS

def labs_energy_01(x):
    s = 2 * np.asarray(x) - 1
    N = len(s)
    E = 0
    for k in range(1, N):
        Ck = np.sum(s[: N - k] * s[k:])
        E += Ck**2
    return E


def combine(p1, p2):
    N = len(p1)
    k = np.random.randint(1, N)
    return np.concatenate([p1[:k], p2[k:]])


def mutate(x, p_mut):
    y = x.copy()
    for i in range(len(y)):
        if np.random.rand() < p_mut:
            y[i] ^= 1
    return y

def tabu_search(x0, tabu_len=10, max_steps=1000):
    x = x0.copy()
    E = labs_energy_01(x)
    tabu = []

    for _ in range(max_steps):
        best_move = None
        best_E = E

        for i in range(len(x)):
            if i in tabu:
                continue

            x_try = x.copy()
            x_try[i] ^= 1
            E_try = labs_energy_01(x_try)

            if E_try < best_E:
                best_E = E_try
                best_move = i

        if best_move is None:
            break

        x[best_move] ^= 1
        E = best_E
        tabu.append(best_move)
        if len(tabu) > tabu_len:
            tabu.pop(0)

    return x

def memetic_ts(
    N,
    pop_size=20,
    p_mut=None,
    p_crossover=0.7,
    iters=10000,
    init_population=None,
    tabu_len=10,
    tabu_steps=1000,
    verbose=True,
):
    if p_mut is None:
        p_mut = 1 / N

    if init_population is not None:
        pop = init_population.copy()
        pop_size = pop.shape[0]
    else:
        pop = np.random.randint(0, 2, size=(pop_size, N))

    fit = np.array([labs_energy_01(x) for x in pop])
    best_i = int(np.argmin(fit))
    best_x = pop[best_i].copy()
    best_e = float(fit[best_i])

    for t in range(iters):
        if np.random.rand() < p_crossover and pop_size >= 2:
            i, j = np.random.choice(pop_size, 2, replace=False)
            child = combine(pop[i], pop[j])
        else:
            child = pop[np.random.randint(pop_size)]

        child = mutate(child, p_mut)
        refined = tabu_search(child, tabu_len=tabu_len, max_steps=tabu_steps)
        e_ref = float(labs_energy_01(refined))

        if e_ref < best_e:
            best_x = refined.copy()
            best_e = e_ref
            if verbose:
                print(f"[MTS] iter={t} best_E={best_e}")

        r = np.random.randint(pop_size)
        if e_ref < fit[r]:
            pop[r] = refined
            fit[r] = e_ref

    return best_x, best_e
# Main 

def main():
    # rank gating (MQPU)
    rank0 = True
    # if hasattr(cudaq, "mpi") and cudaq.mpi.is_initialized():
    #     rank0 = (cudaq.mpi.rank() == 0)

    wall0 = time.time()
    N_small = args.n_train
    N_big = args.n_target

    # ----GQE on small N
    if rank0:
        print(f"=== GQE on N={N_small} ===")
    tA0 = time.time()

    H_small = build_labs_ham(N_small)
    pool_small, meta_small = make_cd_pool(
        N_small,
        depth_levels=args.depth_levels,
        keep_ratio=args.keep_ratio,
        seed=123,
    )

    def op_coeffs(op):
        return [term.evaluate_coefficient() for term in op]

    def op_words(op, n):
        return [term.get_pauli_word(n) for term in op]  # <- fixed spelling

    @cudaq.kernel
    def ansatz_small(n_qubits: int, coeffs: list[float], words: list[cudaq.pauli_word]):
        q = cudaq.qvector(n_qubits)
        h(q)
        for i in range(len(coeffs)):
            exp_pauli(coeffs[i], q, words[i])

    def objective_small(sampled_ops, **kwargs):
        coeffs = [op_coeffs(op)[0].real for op in sampled_ops]
        words = [op_words(op, N_small)[0] for op in sampled_ops]
        return cudaq.observe(ansatz_small, H_small, N_small, coeffs, words).expectation()

    cfg = get_default_config()
    cfg.max_iters = args.gqe_iters
    cfg.ngates = args.gqe_ngates
    cfg.verbose = False

    best_E_small, picked_ids = solvers.gqe(objective_small, pool_small, config=cfg)
    tA = time.time() - tA0
    if rank0:
        print(f"[A] time={tA:.4f}s  best_E(N={N_small})={best_E_small:.6f}")

    #  transfer (tiling)
    if rank0:
        print(f" Motif Transfer to N={N_big} ===")
    theta_big, words_big = tile_features(picked_ids, meta_small, N_big)

    #  quantum sampling seeds
    if rank0:
        print(" Quantum Seeding ===")
    tC0 = time.time()

    @cudaq.kernel
    def ansatz_big(n_qubits: int, coeffs: list[float], words: list[cudaq.pauli_word]):
        q = cudaq.qvector(n_qubits)
        h(q)
        for i in range(len(coeffs)):
            exp_pauli(coeffs[i], q, words[i])

    samples = cudaq.sample(ansatz_big, N_big, theta_big, words_big, shots_count=args.shots)
    tC = time.time() - tC0
    if rank0:
        print(f"[C] time={tC:.4f}s")

    if not rank0:
        return

    #  classical refinement
    print(f"=== Classical Refinement (MTS/Tabu) ===")
    tD0 = time.time()

    bitstrings = list(samples.keys())
    if len(bitstrings) == 0:
        raise RuntimeError("cudaq.sample returned no bitstrings; cannot seed classical search.")

    counts = np.array([int(samples[b]) for b in bitstrings], dtype=np.int64)
    total = int(counts.sum()) if int(counts.sum()) > 0 else 1
    probs = counts.astype(np.float64) / float(total)

    pop_size = args.pop_size
    chosen = np.random.choice(len(bitstrings), size=pop_size, replace=True, p=probs)

    init_pop = np.zeros((pop_size, N_big), dtype=np.int64)
    for r, idx in enumerate(chosen):
        bs = bitstrings[idx].zfill(N_big)
        init_pop[r] = np.fromiter((1 if c == "1" else 0 for c in bs), dtype=np.int64, count=N_big)

    best_bits, best_energy = memetic_ts(
        N_big,
        pop_size=pop_size,
        p_mut=args.p_mut,
        p_crossover=args.p_crossover,
        iters=args.mts_iters,
        init_population=init_pop,
        tabu_len=args.tabu_len,
        tabu_steps=args.tabu_steps,
        verbose=True,
    )

    tD = time.time() - tD0
    wall = time.time() - wall0

    print(f"[D] time={tD:.4f}s")
    print(f"[Total] time={wall:.4f}s")
    print(f"[Result] E*={best_energy}  bits={best_bits}")

    out = {
        "backend": args.target,
        "N_train": int(N_small),
        "N_target": int(N_big),
        "shots": int(args.shots),
        "time_gqe": float(tA),
        "time_seed": float(tC),
        "time_refine": float(tD),
        "time_total": float(wall),
        "E_train_best": float(best_E_small),
        "E_final": float(best_energy),
        "best_bits_01": best_bits.astype(int).tolist(),
        "picked_ops": [int(i) for i in picked_ids],
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=4)
    print(f"[Saved] {args.output}")


if __name__ == "__main__":
    main()
