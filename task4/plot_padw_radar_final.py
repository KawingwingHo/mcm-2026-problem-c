import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def plot_padw_radar_final():
    output_dir = '/Users/hokawing/Downloads/美赛/task4/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # --------------------------
    # Data Preparation
    # --------------------------
    
    # Dimensions (5 Axes)
    labels = ['Skill Correlation', 'Fairness', 'Fan Engagement', 'Robustness', 'Low Controversy\n(Inverted)']
    
    # Data Series (Values 0-1)
    
    # 1. PADW (The "Balanced Leader")
    # Using standardized values from report Table 1 (Standardized [0,1])
    # [Skill, Fairness, Engagement, Robustness, Low Controversy]
    padw_data = [0.95, 1.00, 0.85, 1.00, 0.97]
    
    # 2. Current Hybrid (High Fan/Skill, but 0 transparency)
    hybrid_data = [1.00, 0.05, 0.80, 0.37, 0.93]
    
    # 3. Rank System (Objective but boring/no punishment)
    rank_data = [1.00, 1.00, 0.45, 0.00, 0.85]
    
    # 4. Percentage System (Traditional)
    # Low Skill, Low Robustness, High Engagement
    pct_rule_data = [0.86, 1.00, 0.90, 0.00, 0.78]
    
    data = [padw_data, hybrid_data, rank_data, pct_rule_data]
    names = ['PADW (Proposed)', 'Current Hybrid', 'Rank System', 'Percentage System']
    colors = ['#d0021b', '#4a90e2', '#333333', '#8b4513'] # Red, Blue, Dark Gray, Dark Brown
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
    
    # --------------------------
    # Plotting
    # --------------------------
    
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, polar=True)
    
    # Angles
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Close loop
    
    # Rotate to put Skill at top (0 degrees is usually East, pi/2 is North)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1) # Clockwise
    
    # Draw Background
    plt.xticks(angles[:-1], labels, fontsize=11, fontweight='bold')
    
    # Y-grids
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["", "", "", "", ""], color="grey", size=8) # Hide labels for cleaner look or keep minimal
    plt.ylim(0, 1.1)
    
    # Plot Data
    
    # 1. Percentage System (Far Back)
    values = pct_rule_data + pct_rule_data[:1]
    ax.plot(angles, values, linewidth=1.5, linestyle='dotted', color='#8b4513', label='Percentage System', alpha=0.6)
    
    # 2. Rank System (Back)
    values = rank_data + rank_data[:1]
    ax.plot(angles, values, linewidth=1.5, linestyle='dashdot', color='#333333', label='Rank System', alpha=0.8)
    ax.fill(angles, values, color='#333333', alpha=0.1)
    
    # 3. Current Hybrid (Middle)
    values = hybrid_data + hybrid_data[:1]
    ax.plot(angles, values, linewidth=2, linestyle='dashed', color='#4a90e2', label='Current Hybrid', alpha=0.9)
    ax.fill(angles, values, color='#4a90e2', alpha=0.1)
    
    # 4. PADW (Front - The "Crusher")
    values = padw_data + padw_data[:1]
    ax.plot(angles, values, linewidth=3, linestyle='solid', color='#d0021b', label='PADW (Proposed)')
    ax.fill(angles, values, color='#d0021b', alpha=0.4) # Increased alpha for saturation
    
    # Legend
    plt.legend(loc='lower right', bbox_to_anchor=(1.15, 0.0))
    
    # Title
    title_text = "Comprehensive Performance Evaluation\n"
    subtitle_text = "PADW demonstrates superior balance between fairness and engagement"
    
    # ax.set_title(title_text + subtitle_text, fontsize=14, fontweight='bold', pad=40)
    plt.figtext(0.5, 0.95, title_text, ha='center', fontsize=16, fontweight='bold')
    plt.figtext(0.5, 0.92, subtitle_text, ha='center', fontsize=12, fontstyle='italic')
    
    # Save SVG
    svg_path = os.path.join(output_dir, 'padw_comprehensive_eval.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved Final Radar Chart to {svg_path}")

if __name__ == "__main__":
    plot_padw_radar_final()
