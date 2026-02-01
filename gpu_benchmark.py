import subprocess
import sys
import re
from pathlib import Path
import time
import csv

def run_benchmark():
    # 1. Compile CUDA code
    print("Compiling CUDA code...")
    classical_dir = Path("classical")
    result = subprocess.run(["make", "-C", str(classical_dir)], capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation Failed:")
        print(result.stderr)
        return

    print("Compilation Successful.")
    print(f"{'N':<5} {'Best E':<10} {'Conv. Time':<12} {'Total Time':<12} {'Merit F.':<10} {'Generations':<12} {'Sequence':<20}")
    print("-" * 100)

    # 2. Run Benchmark for N=1..30
    csv_filename = "benchmark_results.csv"
    print(f"Saving results to {csv_filename}")
    csv_file = open(csv_filename, "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["N", "Best E", "Conv. Time", "Total Time", "Merit F.", "Generations", "Sequence"])

    try:
        for N in range(1,40):
        # Run ./labs_gpu <N>
        cmd = [str(classical_dir / "labs_gpu"), str(N)]
        
        # Capture output
        try:
            # Set a timeout just in case
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            output = proc.stdout
            
            # Parse Output
            best_e_match = re.search(r"Best Energy Found: (-?\d+)", output)
            merit_match = re.search(r"Merit Factor: ([\d\.]+)", output)
            time_match = re.search(r"Finished in ([\d\.]+)s", output)
            seq_match = re.search(r"Best Sequence: ([\+\-]+)", output)
            
            best_e = best_e_match.group(1) if best_e_match else "N/A"
            merit = merit_match.group(1) if merit_match else "N/A"
            total_time = time_match.group(1) if time_match else "N/A"
            seq = seq_match.group(1) if seq_match else ""
            
            # Identify convergence time (first time best energy was hit)
            # Parse history: "Cycle, Energy"
            # We need ClockRate
            rate_match = re.search(r"ClockRate: (\d+) Hz", output)
            clock_rate = float(rate_match.group(1)) if rate_match else 1.0
            
            # Parse Start Cycle
            start_match = re.search(r"Start Cycle: (\d+)", output)
            start_cycle = float(start_match.group(1)) if start_match else 0.0

            # Find the line with the best energy
            convergence_time = "N/A"
            
            # Parse all history lines
            # Regex to handle potential negative signs and scientific notation (optional but safer)
            # e.g., 1.23E9, -5
            history_lines = re.findall(r"(\d+), (-?\d+)", output)
            if history_lines and best_e != "N/A":
                target_e = int(best_e)
                best_cycle = -1
                
                # Check if sorted? The C++ sorts it.
                # Find first occurrence of target_e
                # Note: history_lines contains (timestamp, energy)
                for cyc_str, e_str in history_lines:
                    e = int(e_str)
                    cyc = float(cyc_str) # Could be large
                    if e == target_e:
                        best_cycle = cyc
                        break # Found first instance
                
                if best_cycle != -1 and clock_rate > 0:
                    # Time from kernel start
                    # Use relative to start_cycle if available, otherwise fallback to first log entry (not ideal)
                    
                    base_cycle = start_cycle
                    
                    conv_sec = (best_cycle - base_cycle) / clock_rate
                    # If conv_sec is effectively 0 but not exactly, show small number
                    if conv_sec < 0: conv_sec = 0
                    convergence_time = f"{conv_sec:.12f}"

            # Parse Total Generations
            gen_match = re.search(r"Total Generations: (\d+)", output)
            total_gen = gen_match.group(1) if gen_match else "N/A"

            print(f"{N:<5} {best_e:<10} {convergence_time:<12} {total_time:<12} {merit:<10} {total_gen:<12} {seq:<20}")
            writer.writerow([N, best_e, convergence_time, total_time, merit, total_gen, seq])
            csv_file.flush()

        except subprocess.TimeoutExpired:
            print(f"{N:<5} TIMEOUT")
        except Exception as e:
            print(f"{N:<5} ERROR: {e}")
            
    finally:
        csv_file.close()

if __name__ == "__main__":
    run_benchmark()