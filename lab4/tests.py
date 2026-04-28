import subprocess
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pynvml import *

def cleanup():
    for f in ["A.bin", "B.bin", "C_out.bin"]:
        if os.path.exists(f):
            os.remove(f)

try:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_name = nvmlDeviceGetName(handle)
except Exception:
    gpu_name = "NVIDIA GPU"
    print("NVML not initialized. Resource tracking might be limited.")

sizes = [512, 1024, 2048, 3072, 4096]
block_configs = [16, 32]
results = []
exe_path = "./matrix_mult.exe"

print(f"Testing on: {gpu_name}")
print("-" * 60)

for n in sizes:
    for bs in block_configs:
        print(f"Running: N={n}, Block={bs}...", end=" ", flush=True)

        A = np.random.rand(n, n).astype(np.float64)
        B = np.random.rand(n, n).astype(np.float64)
        A.tofile("A.bin")
        B.tofile("B.bin")

        try:
            proc = subprocess.Popen(
                [exe_path, str(n), "A.bin", "B.bin", "C_out.bin", str(bs)],
                stdout=subprocess.PIPE, text=True
            )
            
            max_load = 0
            while proc.poll() is None:
                try:
                    util = nvmlDeviceGetUtilizationRates(handle)
                    max_load = max(max_load, util.gpu)
                except:
                    pass
            
            stdout, _ = proc.communicate()
            cuda_time = float(stdout.strip())

            gflops = (2 * (n ** 3) / cuda_time) / 1e9
            
            results.append({
                "MatrixSize": n,
                "BlockSize": bs,
                "Time": cuda_time,
                "GFLOPS": gflops,
                "GPULoad": max_load
            })
            print(f"DONE. {gflops:.2f} GFLOPS | Load: {max_load}%")

        except Exception as e:
            print(f"FAILED: {e}")

df = pd.DataFrame(results)
df.to_csv("gpu_performance_report.csv", index=False)

fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.set_xlabel('Matrix Size (N x N)', fontsize=12)
ax1.set_ylabel('Performance (GFLOPS)', color='tab:blue', fontsize=12)
for bs in block_configs:
    subset = df[df['BlockSize'] == bs]
    ax1.plot(subset['MatrixSize'], subset['GFLOPS'], marker='o', linewidth=2, label=f'Perf (Block {bs})')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.set_ylabel('GPU Utilization (%)', color='tab:red', fontsize=12)
for bs in block_configs:
    subset = df[df['BlockSize'] == bs]
    ax2.plot(subset['MatrixSize'], subset['GPULoad'], linestyle='--', alpha=0.6, label=f'Load (Block {bs})')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 105)

plt.title(f'CUDA Matrix Multiplication: Performance vs Resource Usage\nDevice: {gpu_name}', fontsize=14)
fig.tight_layout()

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.savefig("full_resource_report.png", dpi=300)
print("-" * 60)
print("Plot saved: full_resource_report.png")
print("Data saved: gpu_performance_report.csv")

cleanup()

try:
    nvmlShutdown()
except:
    pass