import pandas as pd
import glob
import os
import numpy as np


def calculate_stats(folder, output_csv):
    # Encuentra todos los archivos CSV en la carpeta
    files = glob.glob(os.path.join(folder, '*.csv'))
    if not files:
        print(f"No se encontraron archivos CSV en: {folder}")
        return

    results = []
    metrics = ['time_ms', 'interactions_per_sec']

    for file in files:
        df = pd.read_csv(file)
        # Normaliza nombres de columnas
        clean_cols = [c.strip().lower() for c in df.columns]
        df.columns = clean_cols

        stats = {'filename': os.path.basename(file)}
        for metric in metrics:
            if metric not in clean_cols:
                print(f"Advertencia: columna '{metric}' no encontrada en {os.path.basename(file)}.")
                # opcionalmente, continuar para la siguiente métrica
                continue

            data = df[metric].dropna()
            n = data.shape[0]
            if n == 0:
                print(f"Sin datos en '{metric}' para {os.path.basename(file)}")
                continue

            mean_val = data.mean()
            std_val = data.std(ddof=1)
            sem_val = std_val / np.sqrt(n)
            ci95 = sem_val * 1.96

            stats[f'{metric}_mean'] = mean_val
            stats[f'{metric}_std_dev'] = std_val
            stats[f'{metric}_sem'] = sem_val
            stats[f'{metric}_ci95'] = ci95

        results.append(stats)

    if not results:
        print("No se generaron estadísticas: revisa tus archivos CSV y columnas.")
        return

    # Crea DataFrame de resultados y guarda
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_csv, index=False)
    print(f"Resumen guardado en {output_csv}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Calcula media y barras de error para múltiples archivos CSV (tiempo y tasas)'
    )
    parser.add_argument(
        'folder', help='Carpeta que contiene los archivos CSV'
    )
    parser.add_argument(
        '--output', '-o', default='summary_stats.csv',
        help='Nombre del archivo de salida (CSV con estadísticas)'
    )
    args = parser.parse_args()

    calculate_stats(args.folder, args.output)

