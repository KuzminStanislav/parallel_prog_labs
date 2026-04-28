import subprocess
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

sizes = [512, 1024, 2048, 3072] 
block_configs = [16, 32]
results = []

exe_path = "./matrix_mult.exe"

print(f"Запуск тестов производительности CUDA...")
print(f"Используемый файл: {exe_path}\n")

for n in sizes:
    for block_size in block_configs:
        print(f"Тестирование: N={n}, Block={block_size}x{block_size}", end=" | ", flush=True)

        A = np.random.rand(n, n).astype(np.float64)
        B = np.random.rand(n, n).astype(np.float64)

        A.tofile("A.bin")
        B.tofile("B.bin")

        try:
            process = subprocess.run(
                [exe_path, str(n), "A.bin", "B.bin", "C_out.bin", str(block_size)],
                capture_output=True, text=True, check=True
            )

            cuda_time = float(process.stdout.strip())

            status = "Passed"
            if n <= 1024:
                C_gpu = np.fromfile("C_out.bin", dtype=np.float64).reshape((n, n))
                C_cpu_ref = np.dot(A, B)
                if not np.allclose(C_gpu, C_cpu_ref, atol=1e-7):
                    status = "FAILED"

            gflops = (2 * (n ** 3) / cuda_time) / 1e9

            results.append({
                "N": n,
                "BlockSize": block_size,
                "Time (sec)": round(cuda_time, 4),
                "GFLOPS": round(gflops, 2),
                "Check": status
            })
            print(f"Время: {cuda_time:.4f} сек, GFLOPS: {gflops:.2f}, Статус: {status}")

        except Exception as e:
            print(f"ОШИБКА при запуске: {e}")

df = pd.DataFrame(results)
df.to_csv("cuda_report.csv", index=False)
print("\nОтчет сохранен в cuda_report.csv")

plt.figure(figsize=(10, 6))
for b in block_configs:
    subset = df[df['BlockSize'] == b]
    plt.plot(subset['N'], subset['GFLOPS'], marker='s', label=f'Block Size {b}x{b}')

plt.title("Производительность CUDA: Умножение матриц")
plt.xlabel("Размер матрицы (N)")
plt.ylabel("Производительность (GFLOPS)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("cuda_performance_graph.png")

print("График сохранен в cuda_performance_graph.png")

for f in ["A.bin", "B.bin", "C_out.bin"]:
    if os.path.exists(f):
        os.remove(f)
