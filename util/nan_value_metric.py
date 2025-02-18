import pandas as pd
import matplotlib.pyplot as plt

# Assume df is your original DataFrame
# Filter rows where RMSE is NaN

df = pd.read_pickle("experiment_database.pkl")

print("column name :",df.columns)

df_nan = df[df['RMSE_model'].isna()]

# List of columns starting with "metric"
metric_cols = [col for col in df.columns if col.startswith('metric')]

# Get unique noise values (for columns in subplot grid)
noise_values = sorted(df_nan['noise_level'].unique())

# Get unique (algorithm, input_experiment_folder) pairs (for rows in subplot grid)
pairs = df_nan[['optimization_function', 'input_experiment_folder']].drop_duplicates().sort_values(
    by=['optimization_function', 'input_experiment_folder'])
pairs_list = pairs.values.tolist()  # each element is [algorithm, input_experiment_folder]

print(pairs_list,noise_values)

# Define subplot grid dimensions:
n_rows = len(pairs_list)  # 6 rows if there are 6 unique pairs
n_cols = len(noise_values)  # 3 columns if there are 3 unique noise values

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)

# Ensure axes is 2D (helps when there is only one row or one col)
if n_rows == 1:
    axes = axes.reshape(1, -1)
if n_cols == 1:
    axes = axes.reshape(-1, 1)

# Loop through each unique (algorithm, input_experiment_folder) pair and noise value
for row_idx, (alg, exp_folder) in enumerate(pairs_list):
    for col_idx, noise in enumerate(noise_values):
        # Filter for the current combination
        subset = df_nan[
            (df_nan['optimization_function'] == alg) &
            (df_nan['input_experiment_folder'] == exp_folder) &
            (df_nan['noise_level'] == noise)
        ]
        
        # For each metric column, get the values (dropping any additional NaNs)
        boxplot_data = [subset[metric].dropna() for metric in metric_cols]
        
        ax = axes[row_idx, col_idx]
        ax.boxplot(boxplot_data, labels=metric_cols, patch_artist=True,showfliers=False)
        
        # Set subplot title for the top row and y-label for the first column
        if row_idx == 0:
            ax.set_title(f"Noise: {noise}")
        if col_idx == 0:
            ax.set_ylabel(f"Alg: {alg}\nExp: {exp_folder}")
            
plt.tight_layout()
plt.show()
