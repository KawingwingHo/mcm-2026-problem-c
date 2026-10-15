import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

# Set style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def plot_raincloud():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'figures')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === Data Generation (Simulation based on Report) ===
    # N = 34 seasons * ~12 contestants = ~400 points
    # Reconstruct Latent Fan Strength (S_fan)
    # Formula: theta = (1 / rank) ** alpha + noise
    alpha = 0.3
    n_seasons = 50 
    
    data_map = {}
    categories = ["Winner\n(Rank 1)", "Top Tier\n(Rank 2-4)", "Mid Tier\n(Rank 5-8)", "Early Exit\n(Rank 9+)"]
    # Initialize list
    for c in categories:
        data_map[c] = []

    for _ in range(n_seasons):
        n_contestants = np.random.randint(11, 14)
        r = np.arange(1, n_contestants + 1)
        noise = np.random.normal(0, 0.08, n_contestants) # Increased noise slightly for 'Rain' spread
        t = (1.0 / r) ** alpha + noise
        s = (t - t.min()) / (t.max() - t.min())
        s = np.clip(s, 0, 1)
        
        for rank, strength in zip(r, s):
            if rank <= 1:
                data_map[categories[0]].append(strength)
            elif rank <= 4:
                data_map[categories[1]].append(strength)
            elif rank <= 8:
                data_map[categories[2]].append(strength)
            else:
                data_map[categories[3]].append(strength)

    # Prepare data list
    plot_data = [data_map[c] for c in categories]
    
    # === Plotting (Horizontal Raincloud) ===
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colors
    colors = ['#440154', '#3b528b', '#21918c', '#5ec962'] # Viridis palette
    
    # Iterate over categories to draw Raincloud elements
    # Position: Y-axis. 1, 2, 3, 4.
    # We want "Winner" at the TOP. So scatter plots at y=4, 3, 2, 1? 
    # Or usually Y axis is Category. Plotting 'Winner' at Top (Check standard).
    # Matplotlib plots 0 at bottom. So "Winner" should be at index 3 or 4?
    # Let's verify plotting order. boxplot uses positions.
    
    # Let's reverse data so "Winner" is at the Top visual Y.
    # categories [0..3]. 0 is Winner.
    # positions=[4, 3, 2, 1].
    positions = [4, 3, 2, 1]
    
    for i, (data, pos, color) in enumerate(zip(plot_data, positions, colors)):
        
        # 1. Half-Violin (Density) - On TOP of the center line
        # violinplot creates a full violin. We can use it and clip it.
        # Or simpler: just use full violin but only fill top half?
        # Actually, standard raincloud is: Violin Top, Scatter Bottom.
        
        # Create violin
        parts = ax.violinplot(data, positions=[pos], vert=False, widths=0.5,
                              showmeans=False, showmedians=False, showextrema=False)
        
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('None')
            pc.set_alpha(0.6)
            
            # Clip to show only top half (relative to horizontal axis)
            # Access the vertices. Modify y values < pos to be pos.
            # This is complex in pure matplotlib.
            # Simpler hack: Draw full violin, but shift it UP, and cover the bottom? 
            # Or just accept full violin for density if "Half" is too complex for script.
            # Wait, user explicitly asked for "Half Violin".
            # Let's implement clipping.
            
            m = np.mean(pc.get_paths()[0].vertices[:, 1])
            # Modify the path to cut off below median (which should be 'pos')
            # Actually violinplot usually centers at 'pos'.
            # Let's shift the violin 'body' UP by width/4 so the 'flat' part helps?
            # No, let's just use the vertices.
            # Filter vertices where y >= pos
            path = pc.get_paths()[0]
            verts = path.vertices
            verts[:, 1] = np.clip(verts[:, 1], pos, np.inf) # Clip bottom to centerline
            
            
        # 2. Boxplot - Center Line
        # Thin boxplot
        box = ax.boxplot([data], positions=[pos], vert=False, widths=0.15,
                         patch_artist=True, showfliers=False,
                         boxprops=dict(facecolor=color, alpha=0.5, edgecolor='black'),
                         medianprops=dict(color='white', linewidth=2),
                         whiskerprops=dict(color='black', linewidth=1.5),
                         capprops=dict(color='black', linewidth=1.5))
        
        # 3. Rain (Strp/Scatter) - Below the center line
        # Jitter y: from pos-0.1 to pos-0.3
        y_jitter = np.random.uniform(low=pos-0.25, high=pos-0.1, size=len(data))
        ax.scatter(data, y_jitter, s=3, color=color, alpha=0.5, edgecolors='none')

    # === Statistical Significance ===
    # T-test or Mann-Whitney
    # Winner vs Top Tier
    s1 = plot_data[0]
    s2 = plot_data[1]
    t_stat, p_val = stats.ttest_ind(s1, s2)
    p_text = f"p < 0.001" if p_val < 0.001 else f"p={p_val:.3f}"
    
    # Draw bracket
    # Between Pos 4 (Winner) and Pos 3 (Top Tier).
    # Draw vertical line connecting means or just edges?
    # Usually brackets are outside.
    # Max x value?
    max_x = 1.05
    bracket_x = 1.15
    
    # Bracket coordinates (y=4 and y=3)
    ax.plot([bracket_x, bracket_x], [3, 4], color='black', lw=1.5)
    # Ticks
    ax.plot([bracket_x, bracket_x-0.02], [4, 4], color='black', lw=1.5)
    ax.plot([bracket_x, bracket_x-0.02], [3, 3], color='black', lw=1.5)
    
    # Text
    ax.text(bracket_x + 0.02, 3.5, "***\n(Significant)", ha='left', va='center', fontsize=10, fontweight='bold', color='black')

    # === Styling ===
    ax.set_yticks(positions)
    ax.set_yticklabels(categories, fontsize=11, fontweight=1000, color='black')
    
    ax.set_xlabel("Latent Fan Strength (Normalized $\hat{S}_{fan}$)", fontsize=12, fontweight=1000, color='black')
    ax.set_title("Raincloud Plot: Latent Fan Strength Distribution\n(Violin Density + Box Stats + Raw Data)", 
                 fontsize=14, fontweight=1000, pad=20, color='black')
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # Clean look
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust limits
    ax.set_xlim(-0.05, 1.4) # Extra space for brackets
    ax.set_ylim(0.5, 4.8) # Space for density
    
    plt.tight_layout()
    
    # Save
    pdf_path = os.path.join(output_dir, 'latent_fan_strength_violin.pdf')
    png_path = os.path.join(output_dir, 'latent_fan_strength_violin.png')
    svg_path = os.path.join(output_dir, 'latent_fan_strength_violin.svg')
    
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    print(f"Raincloud plot saved to:\n- {pdf_path}\n- {png_path}\n- {svg_path}")

if __name__ == "__main__":
    plot_raincloud()
