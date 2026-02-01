import subprocess
import time
import os
import csv
import re

# --- Configuration ---
N_RANGE = range(3, 26)  # N = 3 to 19
SHOTS = 100000
POP_SIZE_ONE_SHOT = 128
POP_SIZE_RUNOPT = 8192
STEPS = 10  # Fixed at 10 steps
VARIANTS = ["default", "jenga", "dna", "beyblade"] # Updated variants
LAMBDA_METHODS = ["trig"] # Fixed to trig only
RUNOPT_ITERS = "1M"

LOG_DIR = "logs"
WARM_START_DIR = "warm_starts"
METRICS_FILE = "variant_10steps_benchmark_metrics.csv"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(WARM_START_DIR, exist_ok=True)

def parse_runopt_metrics(output_text):
    """
    Parses 'Total Moves', 'Equivalent Evals', and 'Best Bitstring' from runopt stdout.
    """
    moves_match = re.search(r"Total Moves:\s+(\d+)", output_text)
    evals_match = re.search(r"Equivalent Evals:\s+(\d+)", output_text)
    bits_match = re.search(r"Best Bitstring:\s+([01]+)", output_text)
    
    total_moves = moves_match.group(1) if moves_match else "0"
    equiv_evals = evals_match.group(1) if evals_match else "0"
    best_bitstring = bits_match.group(1) if bits_match else "N/A"
    
    return total_moves, equiv_evals, best_bitstring

def run_evaluation():
    fieldnames = [
        "N", "variant", "lambda_method", "steps", "one_shot_time", "runopt_time", 
        "total_time", "one_shot_status", "runopt_status", 
        "total_moves", "equiv_evals", "best_bitstring", "warm_start_file"
    ]
    
    with open(METRICS_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for n in N_RANGE:
            for variant in VARIANTS:
                for method in LAMBDA_METHODS:
                    # Unique ID for files to prevent collisions
                    param_id = f"n{n}_st{STEPS}_v{variant}_m_{method}"
                    warm_start_path = os.path.join(WARM_START_DIR, f"warm_{param_id}.bin")
                    log_file_path = os.path.join(LOG_DIR, f"log_{param_id}.log")
                    
                    print(f"[*] Testing N={n} | Variant={variant} | Method={method} | Steps={STEPS}")

                    # --- Step 1: Quantum Simulation (one_shot.py) ---
                    one_shot_cmd = [
                        "python", "one_shot.py",
                        "--n", str(n),
                        "--shots", str(SHOTS),
                        "--popsize", str(POP_SIZE_ONE_SHOT),
                        "--steps", str(STEPS),
                        "--variant", variant,
                        "--lambda_method", method,
                        "--output", warm_start_path
                    ]
                    
                    start_os = time.perf_counter()
                    res_os = subprocess.run(one_shot_cmd, capture_output=True, text=True)
                    os_duration = time.perf_counter() - start_os

                    # --- Step 2: Local Search (runopt) ---
                    runopt_cmd = [
                        "./runopt",
                        str(n),
                        str(POP_SIZE_RUNOPT),
                        RUNOPT_ITERS,
                        warm_start_path
                    ]

                    start_ro = time.perf_counter()
                    res_ro = subprocess.run(runopt_cmd, capture_output=True, text=True)
                    ro_duration = time.perf_counter() - start_ro

                    # --- Parsing Metrics ---
                    total_moves, equiv_evals, best_bitstring = parse_runopt_metrics(res_ro.stdout)

                    # --- Logging Data ---
                    with open(log_file_path, "w") as log_file:
                        log_file.write(f"=== {variant.upper()} BENCHMARK ===\n")
                        log_file.write(f"N: {n}\nMethod: {method}\nSteps: {STEPS}\n")
                        log_file.write(f"--- SIMULATION STDOUT ---\n{res_os.stdout}\n")
                        log_file.write(f"--- SIMULATION STDERR ---\n{res_os.stderr}\n")
                        log_file.write(f"--- RUNOPT STDOUT ---\n{res_ro.stdout}\n")
                        log_file.write(f"--- RUNOPT STDERR ---\n{res_ro.stderr}\n")

                    # --- Write to CSV ---
                    writer.writerow({
                        "N": n,
                        "variant": variant,
                        "lambda_method": method,
                        "steps": STEPS,
                        "one_shot_time": f"{os_duration:.4f}",
                        "runopt_time": f"{ro_duration:.4f}",
                        "total_time": f"{(os_duration + ro_duration):.4f}",
                        "one_shot_status": res_os.returncode,
                        "runopt_status": res_ro.returncode,
                        "total_moves": total_moves,
                        "equiv_evals": equiv_evals,
                        "best_bitstring": best_bitstring,
                        "warm_start_file": warm_start_path
                    })
                    csvfile.flush()

    print(f"\n[!] Benchmarking Complete. Results saved to {METRICS_FILE}")

if __name__ == "__main__":
    run_evaluation()