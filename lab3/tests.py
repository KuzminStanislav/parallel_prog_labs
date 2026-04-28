import subprocess
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import platform

MPI_INC = r'C:\Program Files (x86)\Microsoft SDKs\MPI\Include'
MPI_LIB = r'C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64'

sizes = [200, 400, 800, 1200, 1600]
cores = [1, 2, 4]
results = []

def get_system_info():
    info = {
        "Processor": platform.processor(),
        "Physical Cores": psutil.cpu_count(logical=False),
        "Total Cores": psutil.cpu_count(logical=True),
        "RAM GB": round(psutil.virtual_memory().total / (1024**3), 2)
    }
    return info

print("--- System Info ---")
sys_info = get_system_info()
for k, v in sys_info.items():
    print(f"{k}: {v}")

print("\nCompiling C++ code...")
cmd_comp = f'g++ -O3 matrix_mult.cpp -o matrix_mult.exe -I"{MPI_INC}" -L"{MPI_LIB}" -lmsmpi'
try:
    subprocess.run(cmd_comp, check=True, shell=True)
    print("Compilation successful.\n")
except subprocess.CalledProcessError as e:
    print(f"Compilation failed: {e}")
    exit(1)

for n in sizes:
    A = np.random.rand(n, n).astype(np.float64)
    B = np.random.rand(n, n).astype(np.float64)
    A.tofile("A.bin")
    B.tofile("B.bin")

    for p in cores:
        if n % p != 0:
            continue
            
        print(f"Running: N={n}, Cores={p}...")
        
        start_mem = psutil.virtual_memory().used
        
        cmd_run = f'mpiexec -n {p} matrix_mult.exe {n} A.bin B.bin C.bin'
        proc = subprocess.run(cmd_run, capture_output=True, text=True, shell=True)
        
        if proc.returncode == 0:
            try:
                t = float(proc.stdout.strip())
                mem_usage = (psutil.virtual_memory().used - start_mem) / (1024**2) 
                results.append({
                    "N": n, 
                    "Cores": p, 
                    "Time": t, 
                    "Mem_MB": round(max(0, mem_usage), 2)
                })
            except ValueError:
                print(f"Output parsing error: {proc.stdout}")
        else:
            print(f"MPI Error: {proc.stderr}")

df = pd.DataFrame(results)

if not df.empty:
    df.to_csv("performance_results.csv", index=False)
    print("\nResults saved to performance_results.csv")

    for f in ["A.bin", "B.bin", "C.bin"]:
        if os.path.exists(f):
            os.remove(f)
    print("Temporary binary files deleted.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for p in cores:
        subset = df[df["Cores"] == p]
        axes[0].plot(subset["N"], subset["Time"], marker='o', label=f'P={p}')
    axes[0].set_title("Execution Time")
    axes[0].set_ylabel("Seconds")
    axes[0].legend()

    for n in sizes:
        subset = df[df["N"] == n].sort_values("Cores")
        if 1 in subset["Cores"].values:
            t1 = subset[subset["Cores"] == 1]["Time"].values[0]
            axes[1].plot(subset["Cores"], t1 / subset["Time"], marker='s', label=f'N={n}')
    axes[1].plot(cores, cores, color='black', linestyle='--', label='Ideal')
    axes[1].set_title("Speedup (S)")
    axes[1].set_ylabel("T1 / Tp")
    axes[1].legend()

    for n in sizes:
        subset = df[df["N"] == n].sort_values("Cores")
        if 1 in subset["Cores"].values:
            t1 = subset[subset["Cores"] == 1]["Time"].values[0]
            speedup = t1 / subset["Time"]
            axes[2].plot(subset["Cores"], speedup / subset["Cores"], marker='^', label=f'N={n}')
    axes[2].set_title("Efficiency (E)")
    axes[2].set_ylabel("S / P")
    axes[2].set_ylim(0, 1.2)
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Size / Cores")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("analysis_plots.png")
    print("Analysis plots saved to analysis_plots.png")
else:
    print("No data to process.")