import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# Load data and skip description row only
data_file = "data/run_metrics_final.csv"
df = pd.read_csv(data_file, encoding="ISO-8859-1", skiprows=[1])

# Rename columns based on the structure
df.columns = ['Run', 'Type',
              'z2_days', 'z2_hours', 'z2_minutes', 'z2_percentage',
              'z298_days', 'z298_hours', 'z298_minutes', 'z298_percentage']

# Convert time components to numeric, handling empty and non-numeric values
def safe_numeric(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    try:
        # Remove any non-numeric characters except decimal points
        cleaned = re.sub(r'[^\d.]', '', str(val))
        return float(cleaned) if cleaned else 0
    except:
        return 0

# Process each time column
time_cols = ['z2_days', 'z2_hours', 'z2_minutes',
             'z298_days', 'z298_hours', 'z298_minutes']
for col in time_cols:
    df[col] = df[col].apply(safe_numeric)

# Calculate total hours
df['z2_total_hours'] = (df['z2_days'] * 24 +
                        df['z2_hours'] +
                        df['z2_minutes'] / 60)

df['z298_total_hours'] = (df['z298_days'] * 24 +
                          df['z298_hours'] +
                          df['z298_minutes'] / 60)

# Convert percentages to numeric, preserving original values
df['z2_percentage'] = pd.to_numeric(df['z2_percentage'].astype(str).str.replace('%', '').str.replace('accepted', '').str.strip(), errors='coerce')
df['z298_percentage'] = pd.to_numeric(df['z298_percentage'].astype(str).str.replace('%', '').str.replace('accepted', '').str.strip(), errors='coerce')

# Print data for verification
print("Processed data:")
print(df[['Run', 'z2_total_hours', 'z298_total_hours', 'z2_percentage', 'z298_percentage']])

# Extract data for plotting
methods = df['Run'].tolist()
z2_errors = df['z2_total_hours'].tolist()
z2_98_errors = df['z298_total_hours'].tolist()
z2_coverage = df['z2_percentage'].tolist()
z2_98_coverage = df['z298_percentage'].tolist()

# Create plot
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
bar_width = 0.35
x = np.arange(len(methods))

# Colors for better visibility
colors = ["#2196F3", "#FF9800"]  # Blue and Orange
patterns = ["//", "\\"]

# Create bars
bars1 = ax.bar(x - bar_width/2, z2_errors, width=bar_width,
               label="Z-score 2 Max Abs Error",
               color=colors[0], alpha=0.8, hatch=patterns[0])

bars2 = ax.bar(x + bar_width/2, z2_98_errors, width=bar_width,
               label="Z-score 2.98 Max Abs Error",
               color=colors[1], alpha=0.8, hatch=patterns[1])

# Add percentage labels
for i in range(len(x)):
    if not pd.isna(z2_coverage[i]):
        ax.text(x[i] - bar_width/2, z2_errors[i] + 0.5,
                f'{z2_coverage[i]:.1f}%',
                ha='center', va='bottom', fontsize=8,
                color='black', weight='bold')

    if not pd.isna(z2_98_coverage[i]):
        ax.text(x[i] + bar_width/2, z2_98_errors[i] + 0.5,
                f'{z2_98_coverage[i]:.1f}%',
                ha='center', va='bottom', fontsize=8,
                color='black', weight='bold')

# Customize plot
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
ax.set_ylabel("Max Absolute Error (Hours)", fontsize=10)
ax.set_xlabel("Feature Selection Method", fontsize=10)
ax.set_title("Feature Selection Impact on Job Prediction Error",
             fontsize=11, weight='bold')
ax.legend(fontsize=9, loc='upper right')

# Grid lines
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# Set y-axis limit
ax.set_ylim(0, 40)

# Layout and save
plt.tight_layout()
plt.savefig("assets/feature_selection_impact_bar_chart.png",
            format='png', dpi=300, bbox_inches='tight')
plt.show()
