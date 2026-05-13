import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


sizes = [200, 400, 800, 1200, 1600, 2000]
threads_counts = [1, 2, 4, 8, 16]
results = []

cpp_source = "matrix_mult.cpp"
exe_name = "matrix_mult.exe"

print("C++ compilation")
subprocess.run(["g++", "-O3", "-fopenmp", cpp_source, "-o", exe_name], check=True, shell=True)

for n in sizes:
    A = np.random.rand(n, n).astype(np.float64)
    B = np.random.rand(n, n).astype(np.float64)
    A.tofile("A.bin")
    B.tofile("B.bin")
    
    for t in threads_counts:
        print(f"Running N={n:4}, Threads={t:2}...", end=" ", flush=True)

        process = subprocess.run(
            [exe_name, str(n), "A.bin", "B.bin", "C_out.bin", str(t)],
            capture_output=True, text=True, shell=True
        )

        if process.returncode != 0:
            print(f"Error! {process.stderr}")
            continue

        try:
            cpp_time = float(process.stdout.strip())
            print(f"Done: {cpp_time:.4f}s")

            t1_time_list = [r["Time"] for r in results if r["N"] == n and r["Threads"] == 1]
            t1_time = t1_time_list[0] if t1_time_list else cpp_time
            speedup = t1_time / cpp_time
            
            results.append({
                "N": n,
                "Threads": t,
                "Time": round(cpp_time, 4),
                "Speedup": round(speedup, 2)
            })
        except ValueError:
            print(f"Parsing error. Output: {process.stdout}")

df = pd.DataFrame(results)
df.to_csv("report_table.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for t in threads_counts:
    subset = df[df["Threads"] == t]
    if not subset.empty:
        ax1.plot(subset["N"], subset["Time"], marker='o', label=f'Threads: {t}')

ax1.set_title("Execution Time by Threads count")
ax1.set_xlabel("Matrix Size (N)")
ax1.set_ylabel("Seconds")
ax1.legend()
ax1.grid(True)

for n in sizes:
    subset = df[df["N"] == n]
    if not subset.empty:
        ax2.plot(subset["Threads"], subset["Speedup"], marker='s', label=f'N: {n}')

ax2.plot(threads_counts, threads_counts, 'k--', alpha=0.5, label='Ideal Speedup') 
ax2.set_title("Speedup Analysis")
ax2.set_xlabel("Number of Threads")
ax2.set_ylabel("Speedup (T1 / Tn)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("openmp_performance.png")
print("\nVisualization saved to 'openmp_performance.png'")

for file in ["A.bin", "B.bin", "C_out.bin"]:
    if os.path.exists(file):
        os.remove(file)