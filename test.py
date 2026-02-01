import subprocess
import sys
import re
from pathlib import Path
import time

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
    print(f"{'N':<5} {'Best E':<10} {'Time(s)':<10} {'Merit F.':<10} {'Sequence':<20}")
    print("-" * 70)

    # 2. Run Benchmark for N=1..30
    for N in range(1, 31):
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
            
            # Find the line with the best energy
            convergence_time = "N/A"
            
            # Parse all history lines
            # Regex to handle potential negative signs and scientific notation (optional but safer)
            # e.g., 1.23E9, -5
            history_lines = re.findall(r"(\d+), (-?\d+)", output)
            if history_lines and best_e != "N/A":
                target_e = int(best_e)
                first_cycle = -1
                best_cycle = -1
                
                # Check if sorted? The C++ sorts it.
                # Find first occurrence of target_e
                for cyc_str, e_str in history_lines:
                    e = int(e_str)
                    cyc = float(cyc_str) # Could be large
                    if first_cycle == -1: first_cycle = cyc
                    if e == target_e:
                        best_cycle = cyc
                        break # Found first instance
                
                if best_cycle != -1 and clock_rate > 0:
                    # Time from start of log (approx)
                    # Use relative to first cycle in log to match user intuition of "search time"
                    # ignoring setup time unless first log IS setup.
                    # Actually, if we improve immediately at T0, time is 0.
                    
                    conv_sec = (best_cycle - first_cycle) / clock_rate
                    # If conv_sec is effectively 0 but not exactly, show small number
                    if conv_sec < 0: conv_sec = 0
                    convergence_time = f"{conv_sec:.6f}"

            print(f"{N:<5} {best_e:<10} {convergence_time:<10} {merit:<10} {seq:<20}")

        except subprocess.TimeoutExpired:
            print(f"{N:<5} TIMEOUT")
        except Exception as e:
            print(f"{N:<5} ERROR: {e}")

if __name__ == "__main__":
    run_benchmark()