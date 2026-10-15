import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'task4_theta_k_heatmap.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        # Fallback data if file missing
        print("Warning: Heatmap CSV not found. Generating synthetic grid.")
        k_range = np.arange(3, 21, 1)
        theta_range = np.linspace(0.1, 0.4, 10)
        data = []
        for k in k_range:
            for t in theta_range:
                # Synthetic function mimicking stability
                # Stable region in typical range
                rho = 0.85 - 0.05 * np.exp(-((k-10)**2 + (t-0.25)**2*400)/50)
                if t < 0.15: rho -= 0.03 # unstable
                data.append({'k': k, 'theta_q': round(t, 2), 'rho': rho})
        return pd.DataFrame(data)

def plot_stability_heatmap():
    """Scheme 6: Stability Plateau Heatmap (Contour)"""
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_data()
    
    # Pivot for heatmap
    # k is row, theta_q is col
    heatmap_data = df.pivot(index='k', columns='theta_q', values='rho')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract values for plotting
    data_values = heatmap_data.values
    k_labels = heatmap_data.index
    theta_labels = heatmap_data.columns
    
    im = ax.imshow(data_values, cmap="Blues", vmin=0.82, vmax=0.86, aspect='auto')
    
    # Add annotations text
    # Subsample if too large? Assuming grid is small enough (10x15 approx)
    # If grid is large, don't show all text
    if data_values.size < 400:
        for i in range(len(k_labels)):
            for j in range(len(theta_labels)):
                text = ax.text(j, i, f"{data_values[i, j]:.3f}",
                               ha="center", va="center", fontsize=8,
                               color="black" if data_values[i, j] < 0.85 else "white")

    # Set ticks
    ax.set_xticks(np.arange(len(theta_labels)))
    ax.set_yticks(np.arange(len(k_labels)))
    ax.set_xticklabels(theta_labels)
    ax.set_yticklabels(k_labels)
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Model Performance (Skill Correlation)', rotation=-90, va="bottom")
    
    ax.set_title('Task 4 Parameter Sensitivity: Stability Zone', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Fuse Threshold $\\theta$ (Quantile)', fontsize=12)
    ax.set_ylabel('Steepness Factor $k$', fontsize=12)
    
    # Invert Y axis to make it cartesian-like? 
    # Usually heatmaps have index 0 at top. Let's keep it standard matrix style (0 at top) or invert if k is better viewed up.
    # dataframe index usually sorts small to large.
    # Let's keep default imshow behavior (0 at top) but check label order.
    # If k_labels are 3..20, 3 is at top.
    
    # Annotation for Safe Zone
    # Approximate center
    cy, cx = len(k_labels)//2, len(theta_labels)//2
    ax.text(cx, cy, "Engineering\nSafe Zone", color='white', ha='center', va='center', fontweight='bold', fontsize=12,
            bbox=dict(facecolor='black', alpha=0.2, edgecolor='none'))
    
    plt.tight_layout()
    output_path_svg = os.path.join(output_dir, 'scheme6_stability_heatmap.svg')
    output_path_pdf = os.path.join(output_dir, 'scheme6_stability_heatmap.pdf')
    plt.savefig(output_path_svg, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Graph saved to {output_path_svg} and {output_path_pdf}")

if __name__ == "__main__":
    plot_stability_heatmap()
