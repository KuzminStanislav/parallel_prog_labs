import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


sizes = [200, 400, 800, 1200, 1600, 2000]
results = []

print("C++ compilation")
subprocess.run(["g++", "-O3", "matrix_mult.cpp", "-o", "matrix_mult.exe"], check=True, shell=True)

for n in sizes:
    print(f"\n--- run for N = {n} ---")

    A = np.random.rand(n, n).astype(np.float64)
    B = np.random.rand(n, n).astype(np.float64)

    A.tofile("A.bin")
    B.tofile("B.bin")

    process = subprocess.run(
        ["matrix_mult.exe", str(n), "A.bin", "B.bin", "C_out.bin"],
        capture_output=True, text=True, shell=True
    )

    cpp_time = float(process.stdout.strip())
    print(f"Duration (C++): {cpp_time:.4f} seconds")

    C_cpp = np.fromfile("C_out.bin", dtype=np.float64).reshape((n, n))

    C_py = np.dot(A, B)
    is_correct = np.allclose(C_cpp, C_py, atol=1e-8)
    print(f"Verification: {'DONE' if is_correct else 'Error'}")

    complexity = 2 * (n ** 3)
    gflops = (complexity / cpp_time) / 1e9 if cpp_time > 0 else 0

    results.append({
        "N": int(n),
        "Time (sec)": round(cpp_time, 4),
        "GFLOPS": round(gflops, 4),
        "Operations": f"{complexity:.2e}",
        "Efficiency": round(gflops / 10, 4)
    })
    

df = pd.DataFrame(results)
df.to_csv("report_table.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.plot(df["N"], df["Time (sec)"], marker='o', color='b', linewidth=2)
ax1.set_title("Execution Time")
ax1.set_xlabel("Matrix Size (N)")
ax1.set_ylabel("Seconds")
ax1.grid(True)

ax2.plot(df["N"], df["GFLOPS"], marker='s', color='r', linewidth=2)
ax2.set_title("Performance (GFLOPS)")
ax2.set_xlabel("Matrix Size (N)")
ax2.set_ylabel("GFLOPS")
ax2.grid(True)

plt.tight_layout()
plt.savefig("performance_analysis.png")

for file in ["A.bin", "B.bin", "C_out.bin"]:
    if os.path.exists(file):
        os.remove(file)