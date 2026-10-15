import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def plot_task3_methodology():
    output_dir = '/Users/hokawing/Downloads/美赛/task3'
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # helper for drawing boxes
    def draw_box(x, y, w, h, text, color, boxstyle="round,pad=0.3"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=boxstyle, 
                                      linewidth=2, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=10, fontweight='bold', wrap=True)

    # --------------------------
    # Layers
    # --------------------------
    # Layer 1: Data Integration
    draw_box(5, 85, 20, 10, "Observational Data\n(Judge Scores)", "#e1f5fe")
    draw_box(30, 85, 25, 10, "Latent Variable Inversion\n(Reconstructed Fan Votes)", "#e1f5fe")
    draw_box(60, 85, 35, 10, "Metadata Integration\n(Age, Industry, Partner, State)", "#e1f5fe")
    
    # Layer 2: Core Model
    draw_box(15, 62, 70, 15, "Dual-Track Regression Model Framework\nModel J: $Y_{Judge} = \\beta^{(J)} X + \\gamma^{(J)} Z + \\epsilon_J$\nModel F: $Y_{Fan} = \\beta^{(F)} X + \\gamma^{(F)} Z + \\epsilon_F$", "#fff9c4")
    
    # Layer 3: Statistical Testing Engine
    draw_box(5, 35, 28, 15, "Welch's T-Test\nCoefficient Structural\nDivergence Testing", "#c8e6c9")
    draw_box(36, 35, 28, 15, "Intraclass Correlation (ICC)\nPartner Effect\nQuantification", "#c8e6c9")
    draw_box(67, 35, 28, 15, "Effect Size Indices\n($\eta^2$, Cohen's d, $R^2$)\nIntensity Mapping", "#c8e6c9")
    
    # Layer 4: Discovery & Results (Aligns with Abstract Findings 1, 2, 3)
    draw_box(5, 10, 28, 12, "Structural Split\n(Age Sensitivity Gap)\nJudge vs Fan Divergence", "#f8bbd0")
    draw_box(36, 10, 28, 12, "Partner Effect Paradox\n(ICC Gap: 16% vs 5%)\nTech vs Fame Driver", "#f8bbd0")
    draw_box(67, 10, 28, 12, "Geographic Identity\n(Regional Bias)\n$\eta^2 = 9.5\%$", "#f8bbd0")

    # --------------------------
    # Arrows
    # --------------------------
    def draw_arrow(x1, y1, x2, y2):
        ax.add_patch(patches.FancyArrowPatch((x1, y1), (x2, y2), 
                                             arrowstyle='->', mutation_scale=20, 
                                             linewidth=1.5, color='black'))

    # Layer 1 -> 2
    draw_arrow(15, 85, 20, 77)
    draw_arrow(42, 85, 45, 77)
    draw_arrow(75, 85, 70, 77)
    
    # Layer 2 -> 3
    draw_arrow(30, 62, 20, 50)
    draw_arrow(50, 62, 50, 50)
    draw_arrow(70, 62, 80, 50)
    
    # Layer 3 -> 4
    draw_arrow(20, 35, 20, 22)
    draw_arrow(50, 35, 50, 22)
    draw_arrow(80, 35, 80, 22)

    # --------------------------
    # Labels
    # --------------------------
    ax.text(50, 98, "Task 3 Methodology: Dual-Track Factor Attribution Analysis", 
            ha='center', fontsize=16, fontweight='bold')
    
    ax.text(2, 90, "INPUT", rotation=90, va='center', fontsize=12, fontweight='bold', color='gray')
    ax.text(2, 69, "MODEL", rotation=90, va='center', fontsize=12, fontweight='bold', color='gray')
    ax.text(2, 42, "STATS", rotation=90, va='center', fontsize=12, fontweight='bold', color='gray')
    ax.text(2, 16, "FINDINGS", rotation=90, va='center', fontsize=12, fontweight='bold', color='gray')

    plt.tight_layout()
    
    svg_path = os.path.join(output_dir, 'task3_methodology_framework.svg')
    pdf_path = os.path.join(output_dir, 'task3_methodology_framework.pdf')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved Task 3 Methodology Diagram to {svg_path} and {pdf_path}")

if __name__ == "__main__":
    plot_task3_methodology()
