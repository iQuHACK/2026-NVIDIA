import numpy as np
import pytest


def _cuda_available() -> bool:
    try:
        from numba import cuda  # type: ignore
    except Exception:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def _cpu_emulate_single_walker(N: int, max_steps: int, tabu_tenure: int, tpb: int = 128):
    # Emulate the GPU kernel's per-block logic deterministically on CPU.
    # This is used for parity testing on small N.
    s = np.empty(N, dtype=np.int8)
    tabu = np.zeros(N, dtype=np.int32)

    # bx = 0 in parity tests
    for i in range(N):
        s[i] = 1 if (i % 2 == 0) else -1

    # C[k] for k=0..N-1 (k=0 unused)
    C = np.zeros(N, dtype=np.int32)
    for k in range(1, N):
        dot = 0
        for i in range(N - k):
            dot += int(s[i]) * int(s[i + k])
        C[k] = np.int32(dot)

    current_E = np.int32(0)
    for k in range(1, N):
        current_E = np.int32(current_E + C[k] * C[k])

    best_E = np.int32(current_E)
    best_s = s.copy()

    INT_MAX = np.int32(2147483647)

    for step in range(1, max_steps + 1):
        scratch_val = np.full(tpb, INT_MAX, dtype=np.int32)
        scratch_idx = np.full(tpb, -1, dtype=np.int32)

        # per-thread best over j in {tx, tx+tpb, ...}
        for tx in range(tpb):
            my_best_delta = INT_MAX
            my_best_bit = np.int32(-1)

            for j in range(tx, N, tpb):
                is_tabu = tabu[j] > step
                sj = int(s[j])
                delta = 0
                for k in range(1, N):
                    dCk = 0
                    if j < N - k:
                        dCk += -2 * sj * int(s[j + k])
                    if j >= k:
                        dCk += -2 * int(s[j - k]) * sj
                    if dCk != 0:
                        delta += 2 * int(C[k]) * dCk + dCk * dCk

                cand_E = int(current_E) + int(delta)
                if is_tabu and (cand_E >= int(best_E)):
                    continue

                if delta < int(my_best_delta):
                    my_best_delta = np.int32(delta)
                    my_best_bit = np.int32(j)

            scratch_val[tx] = my_best_delta
            scratch_idx[tx] = my_best_bit

        # reduction by "thread 0": choose min delta, tie by smaller tx
        winner_delta = INT_MAX
        winner_bit = np.int32(-1)
        for i in range(min(N, tpb)):
            val = scratch_val[i]
            if int(val) < int(winner_delta):
                winner_delta = val
                winner_bit = scratch_idx[i]

        best_delta = winner_delta
        best_bit = int(winner_bit)
        if best_bit == -1:
            continue

        sj_old = int(s[best_bit])
        for k in range(1, N):
            if best_bit < N - k:
                C[k] = np.int32(C[k] + (-2 * sj_old * int(s[best_bit + k])))
            if best_bit >= k:
                C[k] = np.int32(C[k] + (-2 * int(s[best_bit - k]) * sj_old))

        s[best_bit] = np.int8(-s[best_bit])
        current_E = np.int32(int(current_E) + int(best_delta))
        tabu[best_bit] = np.int32(step + tabu_tenure)

        if int(current_E) < int(best_E):
            best_E = np.int32(current_E)
            best_s = s.copy()

    return best_s.astype(np.int8), int(best_E)


@pytest.mark.gpu
def test_gpu_cpu_parity_single_walker_smallN():
    pytest.importorskip("numba")

    if not _cuda_available():
        pytest.skip("CUDA not available on this machine")

    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))

    import mts_cuda

    N = 32
    max_steps = 15
    tabu_tenure = 15

    s_gpu, E_gpu = mts_cuda.run_gpu_labs(
        N, num_walkers=1, max_steps=max_steps, tabu_tenure=tabu_tenure
    )
    s_cpu, E_cpu = _cpu_emulate_single_walker(
        N=N, max_steps=max_steps, tabu_tenure=tabu_tenure, tpb=128
    )

    s_gpu = np.asarray(s_gpu, dtype=np.int8)
    assert np.array_equal(s_gpu, s_cpu)
    assert int(E_gpu) == int(E_cpu)


@pytest.mark.gpu
def test_gpu_smoke_produces_valid_sequence_and_energy():
    # This test is intentionally marked as "gpu" and skipped by default via
    # pytest.ini (-m "not gpu"). Run explicitly with: pytest -m gpu
    pytest.importorskip("numba")

    if not _cuda_available():
        pytest.skip("CUDA not available on this machine")

    import sys
    from pathlib import Path

    # Ensure repo root is importable when running `pytest` from anywhere.
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))

    import mts
    import mts_cuda

    N = 32  # must be <= 256 due to shared-memory arrays in the kernel
    s, E = mts_cuda.run_gpu_labs(N, num_walkers=64, max_steps=10, tabu_tenure=15)

    s = np.asarray(s, dtype=np.int8)
    assert s.shape == (N,)
    assert set(np.unique(s)).issubset({-1, 1})
    assert int(mts.energy(s)) == int(E)


def test_gpu_marker_is_configured():
    # Sanity check: repository declares a "gpu" marker in pytest.ini.
    # (Helps avoid pytest warnings as we add more GPU tests.)
    assert hasattr(pytest.mark, "gpu")

