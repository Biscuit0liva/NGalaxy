import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Cargar el CSV con todos los experimentos
csv_path = '/home/benjamin/Desktop/GPU/T2/NGalaxy/analytics/summary_stats.csv'
df = pd.read_csv(csv_path)

# Función para extraer N de un nombre de archivo
import re
def extract_n(name):
    # Para nombres tipo "results_cpu_16384.csv"
    m_cpu = re.match(r"^results_cpu_(\d+)\.csv$", name)
    if m_cpu:
        return int(m_cpu.group(1))
    # Para nombres tipo "results_cuda_base_16384_256.csv" o "results_opencl_base_16384_256.csv"
    m_gpu = re.match(r"^results_(?:cuda_base|opencl_base)_(\d+)_256\.csv$", name)
    if m_gpu:
        return int(m_gpu.group(1))
    return None

# Extraer filas relevantes
cpu_df = df[df['filename'].str.match(r"^results_cpu_\d+\.csv")].copy()
cpu_df['N'] = cpu_df['filename'].apply(extract_n)

cuda_df = df[df['filename'].str.match(r"^results_cuda_base_\d+_256\.csv")].copy()
cuda_df['N'] = cuda_df['filename'].apply(extract_n)

opencl_df = df[df['filename'].str.match(r"^results_opencl_base_\d+_256\.csv")].copy()
opencl_df['N'] = opencl_df['filename'].apply(extract_n)

# Renombrar para evitar confusiones
cpu_df.rename(columns={'time_ms_mean': 'time_cpu'}, inplace=True)
cuda_df.rename(columns={'time_ms_mean': 'time_cuda'}, inplace=True)
opencl_df.rename(columns={'time_ms_mean': 'time_opencl'}, inplace=True)

# Hacer merge de las tres tablas en base a N
merged = cpu_df[['N', 'time_cpu']].merge(
    cuda_df[['N', 'time_cuda']], on='N'
).merge(
    opencl_df[['N', 'time_opencl']], on='N'
)

# Calcular speedup
merged['speedup_cuda'] = merged['time_cpu'] / merged['time_cuda']
merged['speedup_opencl'] = merged['time_cpu'] / merged['time_opencl']

# Ordenar por N
merged.sort_values('N', inplace=True)

# Graficar speedup
plt.figure(figsize=(8, 5))
plt.plot(merged['N'], merged['speedup_cuda'], marker='o', label='CUDA Speedup', color='C0')
plt.plot(merged['N'], merged['speedup_opencl'], marker='s', label='OpenCL Speedup', color='C1')
plt.xscale('log', base=2)
plt.xlabel('Número de partículas (N)')
plt.ylabel('Speedup (x)')
plt.title('Speedup de GPU respecto a CPU')
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show()


# Preparar datos para fitting: x = log2(N), y = speedup
x = np.log2(merged['N'].values)
y_cuda = merged['speedup_cuda'].values
y_opencl = merged['speedup_opencl'].values

# Definición de la función logística (sigmoid)
def logistic(x, L, k, x0, y0):
    return L / (1 + np.exp(-k*(x - x0))) + y0

# Estimaciones iniciales para CUDA
L0_cuda = max(y_cuda) - min(y_cuda)
k0_cuda = 1.0
x0_cuda = np.median(x)
y0_cuda = min(y_cuda)
p0_cuda = [L0_cuda, k0_cuda, x0_cuda, y0_cuda]

# Fit para CUDA
params_cuda, _ = curve_fit(logistic, x, y_cuda, p0=p0_cuda, maxfev=10000)

# Estimaciones iniciales para OpenCL
L0_opencl = max(y_opencl) - min(y_opencl)
k0_opencl = 1.0
x0_opencl = np.median(x)
y0_opencl = min(y_opencl)
p0_opencl = [L0_opencl, k0_opencl, x0_opencl, y0_opencl]

# Fit para OpenCL
params_opencl, _ = curve_fit(logistic, x, y_opencl, p0=p0_opencl, maxfev=10000)

# Generar valores ajustados
x_fit = np.linspace(min(x), max(x), 100)
y_fit_cuda = logistic(x_fit, *params_cuda)
y_fit_opencl = logistic(x_fit, *params_opencl)

# Graficar datos y ajuste
plt.figure(figsize=(8, 5))
plt.plot(merged['N'], y_cuda, 'o', label='CUDA Speedup (datos)', color='C0')
plt.plot(2**x_fit, y_fit_cuda, '-', label='Ajuste logístico CUDA', color='C0')
plt.plot(merged['N'], y_opencl, 's', label='OpenCL Speedup (datos)', color='C1')
plt.plot(2**x_fit, y_fit_opencl, '-', label='Ajuste logístico OpenCL', color='C1')

plt.xscale('log', base=2)
plt.xlabel('Número de partículas (N)')
plt.ylabel('Speedup (x)')
plt.title('Speedup y Ajuste Logístico')
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show()