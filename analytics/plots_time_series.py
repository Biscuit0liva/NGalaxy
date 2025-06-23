import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Carga de datos
df = pd.read_csv('/home/benjamin/Desktop/GPU/T2/NGalaxy/analytics/summary_stats.csv')

output_dir = 'plots_time_series2'
os.makedirs(output_dir, exist_ok=True)

def save_current_plot(output_dir='plots', dpi=300):
    """
    Guarda la figura actual en `output_dir`, usando su título como nombre de fichero.
    """
    # Asegura que exista la carpeta
    os.makedirs(output_dir, exist_ok=True)

    # Extrae y sanea el título
    title = plt.gca().get_title()
    cleaned = re.sub(r'[^0-9a-zA-ZáéíóúÁÉÍÓÚ() ]+', '', title)
    filename = cleaned.replace(' ', '_').lower() + '.png'

    # Guarda la figura
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=dpi)
    print(f"Guardado: {path}")

def parse_filename(fn):
    name = fn.replace('.csv', '')
    # Elimina contenido entre paréntesis
    name = re.sub(r'\(.*?\)', '', name)
    parts = [p for p in name.split('_') if p]
    
    # parts[0] == 'results'
    backend = parts[1]
    
    if backend == 'cpu':
        particles = int(parts[2])
        variation = 'cpu'
    else:
        # GPU: estructura results_{backend}_{var}_{particles}_{block}
        variation_base = parts[2].lower()  # '2d', 'base', 'global'
        particles = int(parts[3])
        block = parts[4]
        variation = f"{variation_base}_{block}"
    
    return backend, variation, particles

# Aplicar parsing
parsed = df['filename'].apply(parse_filename)
df[['backend', 'variation', 'particles']] = pd.DataFrame(parsed.tolist(), index=df.index)

# Ordenar por número de partículas
df = df.sort_values('particles')

# Graficar tiempo medio por backend y variación
plt.figure()
for (backend, variation), group in df.groupby(['backend', 'variation']):
    label = f"{backend} – {variation}"
    plt.plot(group['particles'], group['time_ms_mean'], marker='o', label=label)

plt.xlabel('Número de partículas')
plt.ylabel('Tiempo (ms) (media)')
plt.title('Tiempo medio vs Número de partículas')
plt.legend()
plt.grid(True)
plt.tight_layout()
save_current_plot(output_dir)
plt.show()

avg_df = df.groupby(['backend', 'particles'])['time_ms_mean'].mean().reset_index()

# Ordenar
avg_df = avg_df.sort_values('particles')

pivot = avg_df.pivot(index='particles', columns='backend', values='time_ms_mean').reset_index()

# Filtrar solo el rango de interés: 2^8 a 2^12
low, high = 2**8, 2**12
pivot_range = pivot[(pivot['particles'] >= low) & (pivot['particles'] <= high)]

# Encontrar el umbral donde el tiempo de CPU supera al de ambas GPU
threshold = None
for _, row in pivot_range.iterrows():
    # Asegurarse de que existan columnas 'cuda' y 'opencl' para comparar
    cuda_time = row['cuda'] if 'cuda' in row else float('inf')
    opencl_time = row['opencl'] if 'opencl' in row else float('inf')
    if row['cpu'] > cuda_time and row['cpu'] > opencl_time:
        threshold = int(row['particles'])
        break

# Graficar las tres curvas y marcar el umbral
plt.figure(figsize=(8, 5))
plt.plot(pivot_range['particles'], pivot_range['cpu'], marker='o', label='CPU', color='C2')
plt.plot(pivot_range['particles'], pivot_range['cuda'], marker='o', label='CUDA', color='C0')
plt.plot(pivot_range['particles'], pivot_range['opencl'], marker='s', label='OpenCL', color='C1')

if threshold is not None:
    # Línea vertical en el umbral
    plt.axvline(x=threshold, color='gray', linestyle='--', label=f'Threshold N={threshold}')
    # Etiqueta sobre la línea
    plt.text(
        threshold, 
        plt.ylim()[1] * 0.5, 
        f'N={threshold}', 
        rotation=90, 
        verticalalignment='center', 
        color='gray'
    )

plt.xlabel('Número de partículas')
plt.ylabel('Tiempo medio (ms)')
plt.title('Tiempo medio vs Número de partículas (Rango $2^8$ a $2^{12}$)')
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)
plt.xlim(low, high)
plt.tight_layout()

df_gpu = df[df['backend'].isin(['cuda', 'opencl'])]

# Ordenar por número de partículas
df_gpu = df_gpu.sort_values('particles')

# Graficar tiempo medio por variación de GPU
plt.figure()
for (backend, variation), group in df_gpu.groupby(['backend', 'variation']):
    label = f"{backend.upper()} – {variation}"
    plt.plot(group['particles'], group['time_ms_mean'], marker='o', label=label)

plt.xlabel('Número de partículas')
plt.ylabel('Tiempo (ms) (media)')
plt.title('Tiempo medio vs Número de partículas (GPU Variaciones)')
plt.legend()
plt.grid(True)
plt.tight_layout()
save_current_plot(output_dir)
plt.show()

# Filtrar solo CUDA y OpenCL
df_gpu = df[df['backend'].isin(['cuda', 'opencl'])]

# Agrupar y promediar tiempo medio
avg_df = df_gpu.groupby(['backend', 'particles'])['time_ms_mean'].mean().reset_index()
avg_df = avg_df.sort_values('particles')

# Graficar CUDA y OpenCL promedio
plt.figure()
for backend in ['cuda', 'opencl']:
    subset = avg_df[avg_df['backend'] == backend]
    plt.plot(subset['particles'], subset['time_ms_mean'], marker='o', label=backend.upper())

plt.xlabel('Número de partículas')
plt.ylabel('Tiempo medio (ms)')
plt.title('Tiempo medio vs Número de partículas (CUDA vs OpenCL Promedio)')
plt.legend()
plt.grid(True)
plt.tight_layout()
save_current_plot(output_dir)
plt.show()

# Variaciones únicas de GPU (excluye cpu)
variations = sorted(df[df['backend'] != 'cpu']['variation'].unique())

# Crear 4 plots, uno por cada variación
for var in variations:
    plt.figure()
    for backend in ['cuda', 'opencl']:
        subset = df[(df['backend'] == backend) & (df['variation'] == var)].sort_values('particles')
        if not subset.empty:
            plt.plot(subset['particles'], subset['time_ms_mean'], marker='o', label=backend.upper())
    plt.xlabel('Número de partículas')
    plt.ylabel('Tiempo medio (ms)')
    plt.title(f'Comparación CUDA vs OpenCL ({var})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_current_plot(output_dir)

# Mostrar todos los plots
plt.show()