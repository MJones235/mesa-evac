import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from textwrap import wrap

meta = pd.read_csv("outputs/batch-20250619091336/metadata.csv")
results = []

variable = 'mean_evacuation_delay_m'
x_label = 'Rayleigh delay parameter (min)'

for _, row in meta.iterrows():
    model_path = os.path.join(row.output_path, os.path.basename(row.output_path) + ".model.csv")
    if os.path.exists(model_path):
        df = pd.read_csv(model_path)
        final = df.iloc[-1]
        evacuated = final['number_evacuated']
        to_evacuate = final['number_to_evacuate']
        percent_evacuated = 100 * evacuated / to_evacuate if to_evacuate > 0 else 0

        results.append({
            'variable_value': row[variable],  
            'percent_evacuated': percent_evacuated
        })

# Create DataFrame from results
result_df = pd.DataFrame(results)

# Group by sigma and calculate mean and 95% CI
grouped = result_df.groupby('variable_value')['percent_evacuated']
summary = grouped.agg(['mean', 'count', 'std'])
summary['sem'] = grouped.apply(sem)
summary['ci95'] = 1.96 * summary['sem']  # 95% confidence interval

# Plot mean with error bars
plt.figure(figsize=(6, 4))
plt.errorbar(summary.index, summary['mean'], yerr=summary['ci95'], fmt='o-', capsize=5)
plt.xlabel(x_label, fontsize=16)
plt.ylabel('\n'.join(wrap("Evacuated within 25 mins (% of Population)", 25)), fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()
