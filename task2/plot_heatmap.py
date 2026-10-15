import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

# Set style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def plot_game_evolution_heatmaps():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'figures')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === Simulation Parameters ===
    grid_size = 100
    judge_gaps = np.linspace(0, 0.5, grid_size)
    fan_gaps = np.linspace(0, 2.0, grid_size)
    X, Y = np.meshgrid(judge_gaps, fan_gaps)
    w = 0.5 
    n_sim = 100
    
    # 1. Percentage Rule Probability
    Z_pct = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            jg = X[i, j]
            fg = Y[i, j]
            diff = (1-w)*(fg) - w*(jg) + np.random.normal(0, 0.1, n_sim)
            Z_pct[i, j] = np.mean(diff > 0)

    # 2. Rank Rule Probability
    Z_rank = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            jg = X[i, j]
            fg = Y[i, j]
            fg_effective = np.tanh(fg * 3) * 0.5 
            jg_effective = np.tanh(jg * 3) * 0.5
            diff = (1-w)*fg_effective - w*jg_effective + np.random.normal(0, 0.1, n_sim) 
            Z_rank[i, j] = np.mean(diff > 0)

    # Common Settings
    cmap = plt.cm.RdYlBu_r
    
    # === Plot 1: Percentage Rule ===
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    im1 = ax1.pcolormesh(X, Y, Z_pct, cmap=cmap, shading='auto', vmin=0, vmax=1)
    ax1.set_title("Evolution Heatmap (1): Percentage Rule\n(Open Upset Zone: Higher Populism Risk)", fontsize=16, fontweight=1000, color='black', pad=15)
    ax1.set_xlabel("Judge Score Gap (Professionalism)", fontsize=12, fontweight=1000, color='black')
    ax1.set_ylabel("Fan Vote Gap (Popularity)", fontsize=12, fontweight=1000, color='black')
    
    ax1.plot([0, 0.5], [0, 0.5], 'k--', lw=2, alpha=0.5)
    ax1.text(0.25, 1.2, "High Probability of\nUpset due to\nInfinite Fan Power", 
             color='black', fontsize=11, fontweight=1000, ha='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4))

    # Single Colorbar for Plot 1
    cbar1 = fig1.colorbar(im1, ax=ax1, shrink=0.8, pad=0.03)
    cbar1.set_label("Probability of Upset (Winner Score > Runner-up Score)", fontsize=11, fontweight=1000, color='black')
    for label in ax1.get_xticklabels() + ax1.get_yticklabels() + cbar1.ax.yaxis.get_ticklabels():
        label.set_fontweight(1000)
        label.set_color('black')

    fig1.savefig(os.path.join(output_dir, 'game_evolution_heatmap_1.svg'), format='svg', bbox_inches='tight')
    fig1.savefig(os.path.join(output_dir, 'game_evolution_heatmap_1.pdf'), format='pdf', bbox_inches='tight')
    fig1.savefig(os.path.join(output_dir, 'game_evolution_heatmap_1.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # === Plot 2: Rank Rule ===
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im2 = ax2.pcolormesh(X, Y, Z_rank, cmap=cmap, shading='auto', vmin=0, vmax=1)
    ax2.set_title("Evolution Heatmap (2): Rank Rule\n(Bounded Upset Zone: Professional Safeguard)", fontsize=16, fontweight=1000, color='black', pad=15)
    ax2.set_xlabel("Judge Score Gap (Professionalism)", fontsize=12, fontweight=1000, color='black')
    ax2.set_ylabel("Fan Vote Gap (Popularity)", fontsize=12, fontweight=1000, color='black')
    
    ax2.plot([0, 0.5], [0, 0.5], 'k--', lw=2, alpha=0.5)
    ax2.text(0.25, 0.25, "Safe Zone:\nSystem Bounded Impact\nof Fan Votes", 
             color='black', fontsize=11, fontweight=1000, ha='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4))

    # Single Colorbar for Plot 2
    cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.8, pad=0.03)
    cbar2.set_label("Probability of Upset (Winner Score > Runner-up Score)", fontsize=11, fontweight=1000, color='black')
    for label in ax2.get_xticklabels() + ax2.get_yticklabels() + cbar2.ax.yaxis.get_ticklabels():
        label.set_fontweight(1000)
        label.set_color('black')

    fig2.savefig(os.path.join(output_dir, 'game_evolution_heatmap_2.svg'), format='svg', bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'game_evolution_heatmap_2.pdf'), format='pdf', bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'game_evolution_heatmap_2.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Split heatmaps saved to {output_dir}")

if __name__ == "__main__":
    plot_game_evolution_heatmaps()
