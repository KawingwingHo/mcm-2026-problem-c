import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def plot_mapping_mechanism():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'figures')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === Data Generation ===
    # X-axis: Contestant A's Fan Votes (Normalized relative to others)
    # Assume 3 other contestants with fixed votes: 10, 20, 30.
    # We vary A's votes from 0 to 100 (Extreme outlier).
    
    others = np.array([10, 20, 30])
    x = np.linspace(0, 100, 500)
    
    # 1. Percentage Rule Score: A / (A + Sum(Others))
    # Normalized to 0-1 scale for comparison or just use raw percentage
    y_pct = x / (x + others.sum())
    
    # 2. Rank Rule Score: Points based on Rank
    # Rank 1=4pts, Rank 2=3pts, Rank 3=2pts, Rank 4=1pts (Example)
    # We need to calculate Rank of A given x
    y_rank = []
    
    # Define points for ranks 1 to 4 (4 contestants)
    points_map = {1: 1.0, 2: 0.66, 3: 0.33, 4: 0.0} # Normalized 0-1 for plotting overlay
    
    for val in x:
        # Current votes of all: [val, 10, 20, 30]
        # Rank of val: sum(1 for o in others if o > val) + 1
        rank = sum(1 for o in others if o > val) + 1
        y_rank.append(points_map[rank])
        
    y_rank = np.array(y_rank)
    
    # === Plotting ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Percentage Curve (Linear-like / Diminishing return but continuous)
    # Actually Pct rule is x/(x+C). It's concave but maintains high sensitivity initially and grows monotonically.
    # Linear approximation for illustration: "Linear Sensitivity"
    # User requested "Linear Curve for Percentage" -> "Sensitive to Magnitude"
    
    ax.plot(x, y_pct, color='#D81B60', linewidth=3, label='Percentage Rule (Continuous/Linear Sensitivity)')
    
    # Plot Rank Curve (Step Function)
    ax.step(x, y_rank, where='post', color='#1E88E5', linewidth=3, label='Rank Rule (Step Function/Magnitude Compression)')
    
    # Styling
    ax.set_title("Mechanism Comparison: Populism vs. Professionalism\n(Why Rank Rule Filters Extreme Fan Noise)", 
                 fontsize=14, fontweight=1000, pad=20)
    ax.set_xlabel("Raw Fan Votes (Extreme Values $\\rightarrow$)", fontsize=12, fontweight=1000)
    ax.set_ylabel("Impact on Final Score (Normalized)", fontsize=12, fontweight=1000)
    
    # === Annotations ===
    
    # 1. High Variance Zone (Extreme Populism)
    # Where x > 30 (Winner), Pct keeps growing, Rank stays flat.
    ax.axvspan(30, 100, facecolor='#FFCDD2', alpha=0.4, edgecolor='#E57373', linestyle='--', linewidth=1.5)
    ax.text(65, 0.35, "Populist Zone (Extreme Outliers)\nPercentage Rule rewarded excess votes\n(High Risk of Vote Brigading)", 
            color='#C2185B', fontsize=10, ha='center', fontweight=1000)
    
    # 2. Magnitude Compression
    ax.annotate("Magnitude Compression\n(Capped Impact)", 
                xy=(40, 1.0), xytext=(60, 0.9),
                arrowprops=dict(arrowstyle="->", color='#0D47A1', lw=2),
                color='#0D47A1', fontweight=1000)

    # 3. Transitions
    # Mark the steps - Adjust horizontal (move left) and vertical (decrease slightly) offsets
    ax.text(4, 0.00 + 0.07, "Rank 4", color='black', fontweight=1000, fontsize=10, va='bottom')
    ax.text(14, 0.33 + 0.07, "Rank 3", color='black', fontweight=1000, fontsize=10, va='bottom')
    ax.text(24, 0.66 + 0.07, "Rank 2", color='black', fontweight=1000, fontsize=10, va='bottom')
    ax.text(34, 1.00 + 0.07, "Rank 1 (Winner)", color='black', fontweight=1000, fontsize=11, va='bottom')
    
    ax.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    pdf_path = os.path.join(output_dir, 'mechanism_comparison.pdf')
    png_path = os.path.join(output_dir, 'mechanism_comparison.png')
    svg_path = os.path.join(output_dir, 'mechanism_comparison.svg')
    
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    print(f"Mechanism plot saved to:\n- {pdf_path}\n- {png_path}\n- {svg_path}")

if __name__ == "__main__":
    plot_mapping_mechanism()
