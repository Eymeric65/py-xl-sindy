import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Helper function to darken a color by a given factor.
def darken_color(color, factor):
    rgb = mcolors.to_rgb(color)
    return tuple(c * factor for c in rgb)

# Load the DataFrame.
df = pd.read_pickle("experiment_database.pkl")

# Create a 'couple' column combining algoritm and optimization_function.
df['couple'] = df['algoritm'] + " - " + df['optimization_function']

# Get a sorted list of unique couples.
couples = sorted(df['couple'].unique())

# Assign a base color for each couple using the tab10 colormap.
cmap = plt.cm.get_cmap('tab10')
base_colors = {couple: cmap(i % 10) for i, couple in enumerate(couples)}

# Determine the maximum noise level (used to scale darkening).
max_noise = df['noise_level'].max()

# Define the three metrics.
metrics = ['RMSE_validation', 'RMSE_acceleration', 'RMSE_model']

# Set up three subplots that share the same x-axis.
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 12))
plt.subplots_adjust(hspace=0.3)

width = 0.8  # width available for each couple group

# Loop over each metric and its corresponding subplot.
for ax, metric in zip(axs, metrics):
    # For each couple, plot a box for each noise level.
    for i, couple in enumerate(couples):
        df_couple = df[df['couple'] == couple]
        # Get unique noise levels for this couple, sorted in ascending order.
        noise_levels = sorted(df_couple['noise_level'].unique())
        n_levels = len(noise_levels)
        
        # Compute x positions for each noise level within the group.
        if n_levels == 1:
            pos_list = [i]
        else:
            pos_list = np.linspace(i - width/2, i + width/2, n_levels)
        
        for pos, noise in zip(pos_list, noise_levels):
            # Use all non-NaN values for the current metric.
            data = df_couple[df_couple['noise_level'] == noise][metric].dropna().values
            if len(data) == 0:
                continue  # Skip if there are no valid points.
            
            # Determine the color: noise=0 uses the base color; higher noise gets a darker shade.
            base_color = base_colors[couple]
            if noise == 0:
                color = base_color
            else:
                factor = 1 - (noise / max_noise * 0.5)  # noise 0 => factor 1; max_noise => factor ~0.5.
                color = darken_color(base_color, factor)
            
            # Plot the boxplot for this noise level.
            bp = ax.boxplot(data, positions=[pos], widths=width/(n_levels+1),
                            patch_artist=True, showfliers=False)
            for box in bp['boxes']:
                box.set(facecolor=color, alpha=0.7)
            for whisker in bp['whiskers']:
                whisker.set(color=color)
            for cap in bp['caps']:
                cap.set(color=color)
            for median in bp['medians']:
                median.set(color='black')
            
            # Annotate with the count of data points.
            count = len(data)
            y_max = np.max(data)
            ax.text(pos, -0.3, f"n={count}", ha='center', va='top', fontsize=8, color='black')
    
    ax.set_ylabel(metric)
    ax.set_title(metric)

# Set the shared x-axis: one tick per couple.
axs[-1].set_xticks(range(len(couples)))
axs[-1].set_xticklabels(couples, rotation=45, ha='right')
axs[-1].set_xlabel('Couple (algoritm - optimization_function)')

fig.suptitle("Box Plots for RMSE_validation, RMSE_acceleration, and RMSE_model\nby Couple and Noise Level", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
