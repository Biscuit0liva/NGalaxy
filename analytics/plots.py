#!/usr/bin/env python3
# analyze_results.py

import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).parent
plot_dir   = script_dir / 'plots'

# Ruta al CSV maestro
master_csv = Path('results') / 'master_stats.csv'
if not master_csv.exists():
    print(f"ERROR: no se encontró {master_csv}")
    exit(1)

# Leer el CSV maestro
df = pd.read_csv(master_csv)
df.columns = df.columns.str.strip()

# Extraer backend (cpu|cuda|opencl)
df['backend'] = df['file'].apply(lambda x: x.replace('results_','').split('_')[0].split('.')[0])

# Métricas
time_col = 'time_ms_mean'
ips_col  = 'interactions_per_sec_mean'

# — Comparación CPU vs GPU (CUDA+OpenCL) ——
cpu = df[df['backend']=='cpu']
gpu = df[df['backend'].isin(['cuda','opencl'])]

agg_cpu_gpu = pd.DataFrame({
    'time_ms_mean': [
        cpu[time_col].mean(),
        gpu[time_col].mean()
    ],
    'interactions_per_sec_mean': [
        cpu[ips_col].mean(),
        gpu[ips_col].mean()
    ]
}, index=['cpu','gpu'])

# Graficar
fig, axes = plt.subplots(1,2, figsize=(10,4))
agg_cpu_gpu['time_ms_mean'].plot(kind='bar', ax=axes[0], edgecolor='black')
axes[0].set(title='Tiempo medio (ms)', xlabel='Backend', ylabel='ms')
agg_cpu_gpu['interactions_per_sec_mean'].plot(
    kind='bar', ax=axes[1], edgecolor='black', color='orange'
)
axes[1].set(title='Interacciones/s promedio', xlabel='Backend', ylabel='IPS')
fig.suptitle('CPU vs GPU')
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(plot_dir / 'cpu_vs_gpu.png')
plt.show()

# — Comparación OpenCL vs CUDA ——
opencl = df[df['backend']=='opencl']
cuda   = df[df['backend']=='cuda']

agg_opencl_cuda = pd.DataFrame({
    'time_ms_mean': [
        opencl[time_col].mean(),
        cuda[time_col].mean()
    ],
    'interactions_per_sec_mean': [
        opencl[ips_col].mean(),
        cuda[ips_col].mean()
    ]
}, index=['opencl','cuda'])

fig, axes = plt.subplots(1,2, figsize=(10,4))
agg_opencl_cuda['time_ms_mean'].plot(kind='bar', ax=axes[0], edgecolor='black')
axes[0].set(title='Tiempo medio (ms)', xlabel='Backend', ylabel='ms')
agg_opencl_cuda['interactions_per_sec_mean'].plot(
    kind='bar', ax=axes[1], edgecolor='black', color='orange'
)
axes[1].set(title='Interacciones/s promedio', xlabel='Backend', ylabel='IPS')
fig.suptitle('OpenCL vs CUDA')
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(plot_dir / 'opencl_vs_cuda.png')
plt.show()

# — Comparación interna de variantes para CUDA y OpenCL ——
# Tomamos solo GPU (CUDA+OpenCL) ya parseados
# results_opencl_2d_32768_256_(8x8).csv,8.621808652246257,0.6477686249725014,0.01868390606037318,0.03662045587833143,125207675673.25706,8958033240.964167,258380917.3637885,506426598.03302544
gpu = gpu.copy()

def parse_gpu_filename(fname):
    name = Path(fname).name.replace('results_','').removesuffix('.csv')
    backend, variant, N, block, *rest = name.split('_')
    grid = None
    # si viene _(...), ignoramos
    if '(' in block:
        # significa que block era e.g. "256_(8x8)"; remove grid
        block, grid = block.split('_(')[0], None
    return pd.Series({
        'backend': backend,
        'variant': variant.lower(),
        'block': int(block),
    })

parsed = gpu['file'].apply(parse_gpu_filename)
gpu[['variant','block']] = parsed[['variant','block']]

# Para cada backend (cuda/opencl), pivot por block y variante
for backend in ['cuda','opencl']:
    dfb = gpu[gpu['backend']==backend]
    pivot_time = dfb.groupby(['block','variant'])[time_col]\
                    .mean().unstack().fillna(0)
    pivot_ips  = dfb.groupby(['block','variant'])[ips_col]\
                    .mean().unstack().fillna(0)

    # Graficar tiempos
    pivot_time.plot(kind='bar', figsize=(8,4))
    plt.title(f'{backend.upper()}: time_ms_mean por variante')
    plt.ylabel('time_ms_mean (ms)')
    plt.xlabel('block size')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{backend}_time_variants.png')
    plt.show()

    # Graficar IPS
    pivot_ips.plot(kind='bar', figsize=(8,4))
    plt.title(f'{backend.upper()}: IPS por variante')
    plt.ylabel('interactions_per_sec_mean')
    plt.xlabel('block size')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{backend}_ips_variants.png')
    plt.show()

# Filtrar solo variantes 2D de CUDA y OpenCL y parsear metadata con regex
mask2d = df['file'].str.contains(r'results_(?:cuda|opencl)_2[dD]_', regex=True)
df2d = df[mask2d].copy()

meta = df2d['file'].str.extract(
    r'results_(?P<backend>cuda|opencl)_2[dD]_\d+_(?P<block>\d+)_\((?P<grid>\d+x\d+)\)'
)

# Sólo unimos block y grid, para no duplicar backend
df2d = df2d.join(meta[['block','grid']])

df2d['block'] = df2d['block'].astype(int)
df2d[['gridX','gridY']] = df2d['grid'].str.split('x', expand=True).astype(int)
df2d['grid_str'] = df2d['grid']

# Columnas de interés
time_col = 'time_ms_mean'
min_col  = 'time_ms_min'
max_col  = 'time_ms_max'

# Diccionario con orden de grids por block size
grid_orders = {
    256: ['8x8', '4x16', '2x32', '1x64'],
    248: ['17x4', '34x2', '68x1']
}

# Loop para cada block size con su orden específico
for block_size, grid_order in grid_orders.items():
    subset = df2d[df2d['block'] == block_size]
    stats = subset.groupby(['backend','grid_str']).agg(
        mean_time=(time_col,'mean'),
        min_time=(min_col,'min'),
        max_time=(max_col,'max')
    ).reset_index()
    
    plt.figure(figsize=(8,5))
    for backend in ['cuda','opencl']:
        bstats = stats[stats['backend']==backend].set_index('grid_str').reindex(grid_order)
        y = bstats['mean_time']
        lower_err = y - bstats['min_time']
        upper_err = bstats['max_time'] - y
        plt.errorbar(grid_order, y, yerr=[lower_err, upper_err],
                     marker='o', label=backend, capsize=5)
    
    plt.title(f'Block {block_size}: time_ms_mean vs grid (2D)')
    plt.xlabel('Grid (gridX x gridY)')
    plt.ylabel('time_ms_mean (ms)')
    plt.legend(title='Backend')
    plt.tight_layout()
    plt.savefig(plot_dir / f'grid_vs_time_2d_block{block_size}.png')
    plt.show()
    plt.close()


