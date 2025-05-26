import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

folder_path = Path('results')
pattern_name = os.path.join(folder_path, '*.csv')

csv_files = [f for f in folder_path.glob('*.csv') if f.name != 'master_stats.csv']

records = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    record = {'file': os.path.basename(file_path)}
    for col in df.columns:
        record[f'{col}_mean'] = df[col].mean()
        record[f'{col}_median'] = df[col].median()
        record[f'{col}_std'] = df[col].std()
        record[f'{col}_min'] = df[col].min()
        record[f'{col}_max'] = df[col].max()
    records.append(record)

# DataFrame maestro con las estadísticas
master_df = pd.DataFrame(records)

# Guardar en CSV maestro
output_path = os.path.join(folder_path, 'master_stats.csv')
master_df.to_csv(output_path, index=False)
