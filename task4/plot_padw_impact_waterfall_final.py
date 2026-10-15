import matplotlib.pyplot as plt
import numpy as np
import os

def plot_padw_impact_waterfall_final():
    output_dir = '/Users/hokawing/Downloads/美赛/task4/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    metrics = [
        'Skill Correlation',
        'Fairness', 
        'Controversy Rate',
        'Low-Score Survival'
    ]
    
    # Percentage Changes (PADW vs Current)
    # Positive = Increase, Negative = Decrease
    values = [6.3, 12.0, -15.2, -20.0]
    
    # Colors
    # Skill/Fairness: Increase is GOOD -> Green
    # Controversy/Survival: Decrease is GOOD -> Red (as per user "Left Red")
    # Interpretation: 
    #   Right (Positive X) = Increase in metric value
    #   Left (Negative X) = Decrease in metric value
    #   User said: "Right Green (PADW Improvement)", "Left Red (Lower Bad Metric)"
    colors = ['#2ecc71', '#2ecc71', '#e74c3c', '#e74c3c'] # Green, Green, Red, Red
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(metrics))
    
    # Horizontal Bar Chart
    bars = ax.barh(y_pos, values, align='center', color=colors, alpha=0.9)
    
    # X-Axis Adjustment
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=12, fontweight='bold')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Percentage Change vs Current System (%)', fontsize=12)
    
    # Baseline line
    ax.axvline(0, color='black', linewidth=1, linestyle='-')
    
    # Add Value Labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        
        # Position label outside the bar (or inside if large enough? Outside is safer)
        if width > 0:
            label_x = width + 1
            align = 'left'
            text = f"+{width:.1f}%"
            # Annotate "Improvement" ?
        else:
            label_x = width - 1
            align = 'right'
            text = f"{width:.1f}%"
            
        ax.text(label_x, bar.get_y() + bar.get_height()/2, text, 
                va='center', ha=align, fontsize=12, fontweight='bold', color='black')
        
    # Title
    plt.title("Quantifiable Improvements of PADW over the Current Hybrid System", fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Set X lims to make room for labels
    ax.set_xlim(-25, 25)
    
    # Style tweaks
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False) # Remove left spine so y labels float? Or keep for connection? Keep.
    
    plt.tight_layout()
    
    svg_path = os.path.join(output_dir, 'padw_impact_waterfall.svg')
    plt.savefig(svg_path, format='svg')
    print(f"Saved Impact Waterfall to {svg_path}")

if __name__ == "__main__":
    plot_padw_impact_waterfall_final()
