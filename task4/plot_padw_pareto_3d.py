import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def plot_padw_pareto_3d():
    output_dir = '/Users/hokawing/Downloads/美赛/task4/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # --------------------------
    # Data Preparation
    # --------------------------
    # Axes: X: Professionalism (Skill), Y: Fairness (Controversy Control), Z: Engagement
    
    systems = ['PADW (Proposed)', 'Hybrid System', 'Rank System', 'Percentage System']
    
    # Data points (Standardized [0,1])
    x = [0.95, 1.00, 1.00, 0.86] # Skill Correlation (Professionalism)
    y = [0.97, 0.93, 0.85, 0.78] # Fairness (Controversy Control)
    z = [0.85, 0.80, 0.45, 0.90] # Audience Engagement
    
    colors = ['#d0021b', '#4a90e2', '#333333', '#8b4513'] # Red, Blue, Dark Gray, Dark Brown
    markers = ['o', 's', '^', 'D'] # Circle, Square, Triangle, Diamond
    
    # --------------------------
    # Plotting
    # --------------------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Points and Stems
    for i in range(len(systems)):
        is_padw = (systems[i] == 'PADW (Proposed)')
        size = 400 if is_padw else 150
        alpha = 1.0 if is_padw else 0.8
        
        # Plot the main point
        ax.scatter([x[i]], [y[i]], [z[i]], 
                   color=colors[i], marker=markers[i], s=size, 
                   label=systems[i], depthshade=True, alpha=alpha,
                   edgecolor='black' if is_padw else 'none', linewidth=2 if is_padw else 0)
        
        if is_padw:
            # Highlight PADW with a specific text label
            ax.text(x[i], y[i], z[i] + 0.03, "PADW: Optimal Choice", 
                    color=colors[i], fontsize=12, fontweight='bold', ha='center')
            
            # Triple Projections to emphasize 3D position
            # To Floor
            ax.plot([x[i], x[i]], [y[i], y[i]], [0.3, z[i]], color=colors[i], linestyle='-', alpha=0.8, lw=2.5)
            # To X-Wall (Left)
            ax.plot([0.8, x[i]], [y[i], y[i]], [z[i], z[i]], color=colors[i], linestyle='--', alpha=0.6, lw=1.5)
            # To Y-Wall (Back)
            ax.plot([x[i], x[i]], [1.02, y[i]], [z[i], z[i]], color=colors[i], linestyle='--', alpha=0.6, lw=1.5)
            
        else:
            # Simple stem for others
            ax.plot([x[i], x[i]], [y[i], y[i]], [0.3, z[i]], color=colors[i], linestyle=':', alpha=0.4)
        
        # Plot projected point on floor
        ax.scatter([x[i]], [y[i]], [0.3], color=colors[i], marker=markers[i], s=40, alpha=0.15)

    # --------------------------
    # Axis Styling
    # --------------------------
    ax.set_xlabel('Professionalism (Skill Corr.)', fontsize=11, labelpad=10)
    ax.set_ylabel('Fairness (Controversy Control)', fontsize=11, labelpad=10)
    ax.set_zlabel('Audience Engagement', fontsize=11, labelpad=10)
    
    ax.set_xlim(0.8, 1.02)
    ax.set_ylim(0.7, 1.02)
    ax.set_zlim(0.3, 1.02)
    
    # Grid and View
    ax.view_init(elev=25, azim=135) # Good angle to see the front-right-top corner
    
    # --------------------------
    # Titles and Legend
    # --------------------------
    title = "3D Performance Pareto Frontier"
    subtitle = "PADW achieves the optimal balance (Top-Right-Front corner) across all metrics"
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    plt.title(subtitle, fontsize=12, fontstyle='italic', pad=20)
    
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=10)
    
    # --------------------------
    # Save SVG
    # --------------------------
    svg_path = os.path.join(output_dir, 'padw_pareto_3d.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved 3D Pareto Chart to {svg_path}")

if __name__ == "__main__":
    plot_padw_pareto_3d()
