import argparse
import csv
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Keep imports robust when running from anywhere (mirrors cpu_benchmark.py/tests).
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import classical.mts as mts  # noqa: E402


class _Timeout(Exception):
    pass


def _timeout_handler(_signum, _frame):
    raise _Timeout()


@dataclass(frozen=True)
class QuantumParams:
    popsize: int = 100
    T: float = 1.0
    n_steps: int = 3
    shots_count: int = 1000
    target: Optional[str] = None  # CUDA-Q target name (optional)


@dataclass(frozen=True)
class MTSParams:
    max_iter: int = 1000
    p_sample: float = 0.5
    p_mutate: float = 0.02
    tabu_steps: int = 60
    tabu_tenure: int = 10
    seed: Optional[int] = 0


def _seq_pm1_to_pm(s: np.ndarray) -> str:
    # + for +1, - for -1 (CPU/GPU benchmark formatting).
    s = np.asarray(s, dtype=np.int8)
    return "".join("+" if int(x) == 1 else "-" for x in s.tolist())


def _merit_factor(N: int, E: int) -> Optional[float]:
    # LABS merit factor: F = N^2 / (2E). Undefined at E=0.
    if E <= 0:
        return None
    return (N * N) / (2.0 * float(E))


def _maybe_set_cudaq_target(target: Optional[str]) -> None:
    if not target:
        return

    try:
        import cudaq  # type: ignore  # pylint: disable=import-error
    except Exception as e:
        raise RuntimeError(
            f"CUDA-Q is required to set target={target!r}, but import failed: {e}"
        ) from e

    cudaq.set_target(target)


def _run_one(
    N: int, run_idx: int, qparams: QuantumParams, mparams: MTSParams
) -> Tuple[int, float, float, float, Optional[float], int, str]:
    """
    Returns:
      best_E, sample_time_s, conv_time_s, total_time_s, merit_factor, generations, sequence
    """
    # Import here so the benchmark script still imports even if CUDA-Q isn't installed.
    try:
        import quantum.qe_mts as qe_mts  # noqa: WPS433
    except Exception as e:
        raise RuntimeError(
            "Failed to import quantum.qe_mts. Ensure CUDA-Q and dependencies are installed."
        ) from e

    seed = (
        None
        if mparams.seed is None
        else (int(mparams.seed) + 10_000 * int(run_idx) + int(N))
    )

    t0 = time.perf_counter()

    # (1) Quantum sampling: generate a population (k, N) of {-1,+1} spins.
    t_sample0 = time.perf_counter()
    population_list = qe_mts.quantum_population(
        popsize=int(qparams.popsize),
        T=qparams.T,
        n_steps=int(qparams.n_steps),
        N=int(N),
        shots_count=int(qparams.shots_count),
    )
    t_sample1 = time.perf_counter()

    population0 = np.asarray(population_list, dtype=np.int8)

    # (2) Classical MTS seeded by quantum population.
    best_s, best_E, _pop, _Es, hist, conv_time = mts.MTS(
        k=int(qparams.popsize),
        N=int(N),
        target=0,
        max_iter=int(mparams.max_iter),
        p_sample=float(mparams.p_sample),
        p_mutate=float(mparams.p_mutate),
        tabu_steps=int(mparams.tabu_steps),
        tabu_tenure=int(mparams.tabu_tenure),
        population0=population0,
        seed=seed,
        record_time=True,
    )

    t1 = time.perf_counter()

    best_E = int(best_E)
    generations = max(0, int(len(hist)) - 1)
    seq = _seq_pm1_to_pm(best_s)
    merit = _merit_factor(N, best_E)
    sample_time_s = float(t_sample1 - t_sample0)
    conv_time_s = float(conv_time)  # excludes sampling (MTS-internal)
    total_time_s = float(t1 - t0)  # includes sampling + MTS + overhead

    return best_E, sample_time_s, conv_time_s, total_time_s, merit, generations, seq


def run_benchmark(
    N_min: int = 1,
    N_max_inclusive: int = 39,
    runs: int = 25,
    csv_filename: str = "quantum_benchmark_results.csv",
    timeout_s: float = 180.0,
    qparams: Optional[QuantumParams] = None,
    mparams: Optional[MTSParams] = None,
) -> None:
    qparams = qparams or QuantumParams()
    mparams = mparams or MTSParams()

    # Configure CUDA-Q target (optional).
    _maybe_set_cudaq_target(qparams.target)

    print("Running Quantum benchmark (quantum/qe_mts.py QE-MTS)...")
    if qparams.target:
        print(f"CUDA-Q target: {qparams.target}")
    print(
        f"{'Run':<5} {'N':<5} {'Shots':<7} {'Pop':<5} {'Best E':<10} {'Q Sample':<12} {'MTS Conv':<12} {'Total Time':<12} {'Merit F.':<10} {'Generations':<12} {'Sequence':<20}"
    )
    print("-" * 120)

    print(f"Saving results to {csv_filename}")
    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Run",
                "N",
                "CUDA-Q Target",
                "Shots",
                "Population",
                "T",
                "Trotter Steps",
                "Best E",
                "Q Sample Time",
                "MTS Conv. Time",
                "Total Time",
                "Merit F.",
                "Generations",
                "Sequence",
            ]
        )

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        try:
            for run_idx in range(1, int(runs) + 1):
                for N in range(int(N_min), int(N_max_inclusive) + 1):
                    try:
                        if timeout_s and timeout_s > 0:
                            signal.setitimer(signal.ITIMER_REAL, float(timeout_s))

                        best_E, sample_s, conv_s, total_s, merit, generations, seq = (
                            _run_one(N, run_idx, qparams, mparams)
                        )

                        merit_str = "N/A" if merit is None else f"{merit:.6g}"
                        sample_str = f"{sample_s:.6g}"
                        conv_str = f"{conv_s:.12f}"
                        total_str = f"{total_s:.6g}"

                        print(
                            f"{run_idx:<5} {N:<5} {int(qparams.shots_count):<7} {int(qparams.popsize):<5} {best_E:<10} {sample_str:<12} {conv_str:<12} {total_str:<12} {merit_str:<10} {generations:<12} {seq:<20}"
                        )
                        writer.writerow(
                            [
                                run_idx,
                                N,
                                qparams.target or "",
                                int(qparams.shots_count),
                                int(qparams.popsize),
                                float(qparams.T),
                                int(qparams.n_steps),
                                best_E,
                                sample_str,
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
                            [
                                run_idx,
                                N,
                                qparams.target or "",
                                int(qparams.shots_count),
                                int(qparams.popsize),
                                float(qparams.T),
                                int(qparams.n_steps),
                                "TIMEOUT",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "",
                            ]
                        )
                        csv_file.flush()
                    except Exception as e:
                        print(f"{run_idx:<5} {N:<5} ERROR: {e}")
                        writer.writerow(
                            [
                                run_idx,
                                N,
                                qparams.target or "",
                                int(qparams.shots_count),
                                int(qparams.popsize),
                                float(qparams.T),
                                int(qparams.n_steps),
                                "ERROR",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "",
                            ]
                        )
                        csv_file.flush()
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0.0)
        finally:
            signal.signal(signal.SIGALRM, old_handler)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QE-MTS quantum benchmark suite.")
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=39)
    p.add_argument("--runs", type=int, default=25)
    p.add_argument("--csv", type=str, default="quantum_benchmark_results.csv")
    p.add_argument("--timeout-s", type=float, default=180.0)

    p.add_argument("--target", type=str, default=None, help="CUDA-Q target name")
    p.add_argument("--popsize", type=int, default=100)
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--t", type=float, default=1.0, help="Total evolution time T")
    p.add_argument("--n-steps", type=int, default=3, help="Trotter steps")

    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--p-sample", type=float, default=0.5)
    p.add_argument("--p-mutate", type=float, default=0.02)
    p.add_argument("--tabu-steps", type=int, default=60)
    p.add_argument("--tabu-tenure", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-seed", action="store_true", help="Disable deterministic seeding")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    qparams_ = QuantumParams(
        popsize=args.popsize,
        T=args.t,
        n_steps=args.n_steps,
        shots_count=args.shots,
        target=args.target,
    )
    mparams_ = MTSParams(
        max_iter=args.max_iter,
        p_sample=args.p_sample,
        p_mutate=args.p_mutate,
        tabu_steps=args.tabu_steps,
        tabu_tenure=args.tabu_tenure,
        seed=None if args.no_seed else args.seed,
    )

    run_benchmark(
        N_min=args.n_min,
        N_max_inclusive=args.n_max,
        runs=args.runs,
        csv_filename=args.csv,
        timeout_s=args.timeout_s,
        qparams=qparams_,
        mparams=mparams_,
    )

