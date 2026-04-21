import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

sizes = [200, 400, 800, 1200, 1600, 2000]
cores_list = [1, 2, 4, 8]
results = []

print("Compiling MPI C++ program...")
subprocess.run(["mpicxx", "-O3", "matrix_mult.cpp", "-o", "matrix_mult.exe"], check=True, shell=True)

for n in sizes:
    print(f"\nMatrix Size N = {n}")
    A = np.random.rand(n, n).astype(np.float64)
    B = np.random.rand(n, n).astype(np.float64)
    A.tofile("A.bin")
    B.tofile("B.bin")

    for p in cores_list:
        if n % p != 0:
            continue

        process = subprocess.run(
            ["mpiexec", "-n", str(p), "matrix_mult.exe", str(n), "A.bin", "B.bin", "C_out.bin"],
            capture_output=True, text=True, shell=True
        )

        if process.returncode == 0:
            cpp_time = float(process.stdout.strip())
            
            if p == 1:
                C_cpp = np.fromfile("C_out.bin", dtype=np.float64).reshape((n, n))
                is_correct = np.allclose(C_cpp, np.dot(A, B), atol=1e-8)
                print(f"  Verification: {'OK' if is_correct else 'FAIL'}")

            gflops = (2 * (n**3) / cpp_time) / 1e9
            results.append({"N": n, "Cores": p, "Time": cpp_time, "GFLOPS": gflops})
            print(f"  P={p}: {cpp_time:.4f}s | {gflops:.2f} GFLOPS")

df = pd.DataFrame(results)
df.to_csv("mpi_performance_report.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for p in cores_list:
    subset = df[df["Cores"] == p]
    ax1.plot(subset["N"], subset["Time"], marker='o', label=f'{p} Cores')
ax1.set_title("Execution Time (Seconds)")
ax1.set_xlabel("Matrix Size N")
ax1.set_ylabel("Time (s)")
ax1.legend()
ax1.grid(True)

for n in sizes:
    subset = df[df["N"] == n]
    if not subset.empty:
        t1 = subset[subset["Cores"] == 1]["Time"].values[0]
        ax2.plot(subset["Cores"], t1 / subset["Time"], marker='s', label=f'N={n}')
ax2.plot(cores_list, cores_list, 'k--', alpha=0.5, label="Ideal")
ax2.set_title("Speedup Analysis (T1 / Tp)")
ax2.set_xlabel("Number of Cores")
ax2.set_ylabel("Speedup")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("mpi_results.png")
print("\nSuccess! Files 'mpi_performance_report.csv' and 'mpi_results.png' generated.")

for f in ["A.bin", "B.bin", "C_out.bin"]:
    if os.path.exists(f): os.remove(f)