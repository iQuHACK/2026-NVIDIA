import csv
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Keep imports robust when running from anywhere (mirrors tests).
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import classical.mts as mts  # noqa: E402


class _Timeout(Exception):
    pass


def _timeout_handler(_signum, _frame):
    raise _Timeout()


@dataclass(frozen=True)
class BenchmarkParams:
    k: int = 12
    max_iter: int = 1000
    p_sample: float = 0.5
    p_mutate: float = 0.02
    tabu_steps: int = 60
    tabu_tenure: int = 10
    seed: Optional[int] = 0


def _seq_pm1_to_pm(s: np.ndarray) -> str:
    # + for +1, - for -1 (GPU benchmark formatting).
    s = np.asarray(s, dtype=np.int8)
    return "".join("+" if int(x) == 1 else "-" for x in s.tolist())


def _merit_factor(N: int, E: int) -> Optional[float]:
    # LABS merit factor: F = N^2 / (2E). Undefined at E=0.
    if E <= 0:
        return None
    return (N * N) / (2.0 * float(E))


def _run_one(
    N: int, run_idx: int, params: BenchmarkParams
) -> Tuple[int, float, float, Optional[float], int, str]:
    """
    Returns:
      best_E, conv_time_s, total_time_s, merit_factor, generations, sequence
    """
    seed = (
        None
        if params.seed is None
        else (int(params.seed) + 10_000 * int(run_idx) + int(N))
    )

    t0 = time.perf_counter()
    best_s, best_E, _pop, _Es, hist, conv_time = mts.MTS(
        k=params.k,
        N=N,
        target=0,
        max_iter=params.max_iter,
        p_sample=params.p_sample,
        p_mutate=params.p_mutate,
        tabu_steps=params.tabu_steps,
        tabu_tenure=params.tabu_tenure,
        population0=None,
        seed=seed,
        record_time=True,
    )
    t1 = time.perf_counter()

    best_E = int(best_E)
    generations = max(0, int(len(hist)) - 1)
    seq = _seq_pm1_to_pm(best_s)
    merit = _merit_factor(N, best_E)
    conv_time_s = float(conv_time)
    total_time_s = float(t1 - t0)

    return best_E, conv_time_s, total_time_s, merit, generations, seq


def run_benchmark(
    N_min: int = 1,
    N_max_inclusive: int = 39,
    runs: int = 100,
    csv_filename: str = "cpu_benchmark_results.csv",
    timeout_s: float = 60.0,
    params: Optional[BenchmarkParams] = None,
) -> None:
    params = params or BenchmarkParams()

    print("Running CPU benchmark (classical/mts.py)...")
    print(
        f"{'Run':<5} {'N':<5} {'Best E':<10} {'Conv. Time':<12} {'Total Time':<12} {'Merit F.':<10} {'Generations':<12} {'Sequence':<20}"
    )
    print("-" * 100)

    print(f"Saving results to {csv_filename}")
    with open(csv_filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Run",
                "N",
                "Best E",
                "Conv. Time",
                "Total Time",
                "Merit F.",
                "Generations",
                "Sequence",
            ]
        )

        # Best-effort per-N timeout (mirrors GPU bench behavior).
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        try:
            for run_idx in range(1, int(runs) + 1):
                for N in range(int(N_min), int(N_max_inclusive) + 1):
                    try:
                        if timeout_s and timeout_s > 0:
                            # ITIMER_REAL supports sub-second timeouts.
                            signal.setitimer(signal.ITIMER_REAL, float(timeout_s))

                        best_E, conv_time_s, total_time_s, merit, generations, seq = (
                            _run_one(N, run_idx, params)
                        )

                        merit_str = "N/A" if merit is None else f"{merit:.6g}"
                        conv_str = f"{conv_time_s:.12f}"
                        total_str = f"{total_time_s:.6g}"

                        print(
                            f"{run_idx:<5} {N:<5} {best_E:<10} {conv_str:<12} {total_str:<12} {merit_str:<10} {generations:<12} {seq:<20}"
                        )
                        writer.writerow(
                            [
                                run_idx,
                                N,
                                best_E,
                                conv_str,
                                total_str,
                                merit_str,
                                generations,
                                seq,
                            ]
                        )
                        csv_file.flush()

                    except _Timeout:
                        print(f"{run_idx:<5} {N:<5} TIMEOUT")
                        writer.writerow(
                            [run_idx, N, "TIMEOUT", "N/A", "N/A", "N/A", "N/A", ""]
                        )
                        csv_file.flush()
                    except Exception as e:
                        print(f"{run_idx:<5} {N:<5} ERROR: {e}")
                        writer.writerow(
                            [run_idx, N, "ERROR", "N/A", "N/A", "N/A", "N/A", ""]
                        )
                        csv_file.flush()
                    finally:
                        # Cancel alarm for the next iteration.
                        signal.setitimer(signal.ITIMER_REAL, 0.0)
        finally:
            signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    run_benchmark()
