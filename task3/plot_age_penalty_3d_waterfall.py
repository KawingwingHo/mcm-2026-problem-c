import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

def plot_age_penalty_3d_waterfall():
    output_dir = '/Users/hokawing/Downloads/美赛/task3/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    # Categories: Judge, Fan
    # Coefficients (Height)
    # Errors (CI)
    
    categories = ['Judge Model', 'Fan Model']
    coeffs = [-0.0588, -0.1405]
    errors = [0.0076, 0.0195]
    
    # Colors
    colors = ['#4a90e2', '#d0021b'] # Blue, Red
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Grid setup
    # X: Category Index (although we can just use 0 for both if they are separated by Y)
    # Y: Depth (Judge front, Fan back)
    # Z: Value
    
    # Let's put them side-by-side in X, AND separated in Y?
    # Waterfall usually means staggered.
    # Let's do:
    # Judge: X=0, Y=0
    # Fan: X=1, Y=1 (or Y=0.5)
    
    # Bar Dimensions
    dx = 0.4
    dy = 0.4
    
    # Positions
    x_pos = [0, 0.5] # Staggered X
    y_pos = [0, 1.0] # Staggered Y (Depth)
    z_pos = np.zeros(2)
    
    # Heights
    dz = np.abs(coeffs) # Height is magnitude. Since they are negative, we plot them downwards? 
                        # Or upwards as "Magnitude of Penalty"?
                        # Original chart was usually "Age Penalty" (implied negative).
                        # Let's plot as positive magnitude but label as negative, OR plot downwards.
                        # Plotting downwards in 3D is sometimes hard to read if floor is opaque.
                        # Let's plot upwards as "Magnitude of Bias" and label with signage.
    
    dz_plot = [abs(c) for c in coeffs]
    
    # Create Bars
    for i in range(2):
        ax.bar3d(x_pos[i], y_pos[i], 0, dx, dy, dz_plot[i], color=colors[i], alpha=0.8, shade=True)
        
        # Add Error Bar (Line on top)
        # Center of top face:
        cx = x_pos[i] + dx/2
        cy = y_pos[i] + dy/2
        cz = dz_plot[i]
        
        # Error len
        err = errors[i] * 1.96 # 95% CI roughly
        
        # Draw line
        # Vertical error bar? Or just a cap?
        # Vertical line from cz-err to cz+err?
        # If we plot distribution, it's +/- err. 
        # Here bar represents Mean. Error is around Mean.
        
        # 3D Line for Error
        ax.plot([cx, cx], [cy, cy], [cz - err, cz + err], color='black', linewidth=1.5, zorder=10)
        # Caps
        cap_len = 0.1
        # Horizontal caps are tricky in 3D (which axis?). Let's do small X-lines.
        ax.plot([cx-cap_len/2, cx+cap_len/2], [cy, cy], [cz+err, cz+err], color='black', linewidth=1.5)
        ax.plot([cx-cap_len/2, cx+cap_len/2], [cy, cy], [cz-err, cz-err], color='black', linewidth=1.5)
        
        # Label
        label = f"{coeffs[i]:.4f}\n(±{err:.4f})"
        ax.text(cx, cy, cz + err + 0.01, label, ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=20)
        
    # Set Limits
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 2.0)
    ax.set_zlim(0, 0.20)
    
    # Labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('Penalty Magnitude (|β|)', fontsize=11)
    
    # TIcks
    ax.set_xticks([])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11, fontweight='bold', va='center', ha='left')
    
    # View
    ax.view_init(elev=20, azim=-50)
    
    ax.set_title("Age Penalty Comparison (3D Waterfall)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    svg_path = os.path.join(output_dir, 'age_penalty_3d_waterfall.svg')
    plt.savefig(svg_path, format='svg')
    print(f"Saved 3D Waterfall to {svg_path}")

if __name__ == "__main__":
    plot_age_penalty_3d_waterfall()
