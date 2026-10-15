import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def plot_task1_methodology():
    output_dir = '/Users/hokawing/Downloads/美赛/task1'
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
                fontsize=11, fontweight='bold', wrap=True)

    # --------------------------
    # Layers
    # --------------------------
    # Layer 1: Data Input (Top)
    draw_box(10, 85, 20, 10, "Raw Judge Scores\n(S1-S34)", "#e1f5fe")
    draw_box(40, 85, 20, 10, "Elimination History\n(Sequential Ranks)", "#e1f5fe")
    draw_box(70, 85, 20, 10, "Season Metadata\n(Judge Counts/Cap)", "#e1f5fe")
    
    # Layer 2: Preprocessing
    draw_box(10, 65, 35, 12, "Adaptive Normalization Operator ($\Phi_w$)\nMapping to [0, 1] Elastic Niche", "#fff9c4")
    draw_box(55, 65, 35, 12, "Pareto Truncated Inversion\n$\theta_i \propto (1/rank_i)^\\alpha$", "#fff9c4")
    
    # Layer 3: Model Core
    draw_box(25, 40, 50, 15, "Dynamic Performance Evaluation Model\n$S_{total} = w_j \cdot s_{judge} + (1-w_j) \cdot s_{fan}$\nMLE Optimization for $\\alpha^*$ and $w_j^*$", "#c8e6c9")
    
    # Layer 4: Measurement & Analysis
    draw_box(5, 15, 25, 12, "JPADI Index\nJudge-Public Alignment\nDrift Measurement", "#f8bbd0")
    draw_box(37.5, 15, 25, 12, "Accuracy Verification\n(Predicted vs Actual)\n94.7% Success Rate", "#f8bbd0")
    draw_box(70, 15, 25, 12, "Uncertainty Quant.\nBootstrap Resampling\nMonte Carlo Sim.", "#f8bbd0")

    # --------------------------
    # Arrows
    # --------------------------
    def draw_arrow(x1, y1, x2, y2):
        ax.add_patch(patches.FancyArrowPatch((x1, y1), (x2, y2), 
                                             arrowstyle='->', mutation_scale=20, 
                                             linewidth=1.5, color='black'))

    # Layer 1 -> 2
    draw_arrow(20, 85, 25, 77)
    draw_arrow(50, 85, 50, 77)
    draw_arrow(80, 85, 75, 77)
    
    # Layer 2 -> 3
    draw_arrow(27, 65, 40, 55)
    draw_arrow(73, 65, 60, 55)
    
    # Layer 3 -> 4
    draw_arrow(40, 40, 25, 27)
    draw_arrow(50, 40, 50, 27)
    draw_arrow(60, 40, 75, 27)

    # --------------------------
    # Labels
    # --------------------------
    ax.text(50, 98, "Methodology Framework: Dynamic Performance Estimation Model", 
            ha='center', fontsize=16, fontweight='bold')
    
    # Add small annotations
    ax.text(2, 90, "INPUT", rotation=90, va='center', fontsize=12, fontweight='bold', color='gray')
    ax.text(2, 71, "PROCESS", rotation=90, va='center', fontsize=12, fontweight='bold', color='gray')
    ax.text(2, 47, "MODEL", rotation=90, va='center', fontsize=12, fontweight='bold', color='gray')
    ax.text(2, 21, "EVAL", rotation=90, va='center', fontsize=12, fontweight='bold', color='gray')

    plt.tight_layout()
    
    svg_path = os.path.join(output_dir, 'task1_methodology_framework.svg')
    pdf_path = os.path.join(output_dir, 'task1_methodology_framework.pdf')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved Task 1 Methodology Diagram to {svg_path} and {pdf_path}")

if __name__ == "__main__":
    plot_task1_methodology()
