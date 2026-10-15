import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
import os

# Set style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'sans-serif']
except:
    pass

def draw_sankey_link(ax, x_start, x_end, y_start_top, y_start_bot, y_end_top, y_end_bot, color1, color2, alpha=0.5, n_segments=30):
    """Draws a gradient sigmoid link between two vertical segments."""
    from matplotlib.colors import to_rgba
    
    xs = np.linspace(x_start, x_end, n_segments + 1)
    
    # Sigmoid interpolation for y positions
    def sigmoid(x):
        return 1 / (1 + np.exp(-10 * (x - 0.5)))
    
    # Normalize x for sigmoid
    x_norm = (xs - x_start) / (x_end - x_start)
    y_interp = sigmoid(x_norm)
    
    y_tops = y_start_top + (y_end_top - y_start_top) * y_interp
    y_bots = y_start_bot + (y_end_bot - y_start_bot) * y_interp
    
    rgba1 = to_rgba(color1, alpha)
    rgba2 = to_rgba(color2, alpha)
    
    for i in range(n_segments):
        # Linear interpolation for color
        t = i / n_segments
        curr_color = [rgba1[j] * (1 - t) + rgba2[j] * t for j in range(4)]
        
        poly_xs = [xs[i], xs[i+1], xs[i+1], xs[i]]
        poly_ys = [y_tops[i], y_tops[i+1], y_bots[i+1], y_bots[i]]
        
        ax.fill(poly_xs, poly_ys, color=curr_color, edgecolor='none')

def plot_controversy_sankey():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'figures')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 26) 
    
    # === Data ===
    v_total = 22.5
    v_rank_filtered = 7.3
    v_rank_remain = 15.2
    v_judge_filtered = 7.9
    v_resid = 7.3
    
    # Coordinates
    x_stage1 = 1.0
    x_stage2 = 5.0
    x_stage3 = 9.0
    
    # Y-positions
    y_start_top = 22.0
    y_start_bot = y_start_top - v_total
    
    # Stage 2
    gap = 2.4
    y_s2_filt_top = 24.0
    y_s2_filt_bot = y_s2_filt_top - v_rank_filtered
    
    y_s2_rem_top = y_s2_filt_bot - gap
    y_s2_rem_bot = y_s2_rem_top - v_rank_remain
    
    # Stage 3
    y_s3_filt_top = y_s2_rem_top + 4.0 
    y_s3_filt_bot = y_s3_filt_top - v_judge_filtered
    
    y_s3_resid_top = y_s3_filt_bot - gap
    y_s3_resid_bot = y_s3_resid_top - v_resid
    
    # Colors
    c_red = '#E53935'
    c_red_light = '#EF9A9A'
    c_green = '#43A047'
    c_green_light = '#A5D6A7'
    c_grey = '#546E7A'
    c_grey_light = '#B0BEC5'
    
    # === Drawing Flows ===
    
    # Flow 1: Total -> Rank Intercept (Gradient Red to Green)
    draw_sankey_link(ax, x_stage1+0.2, x_stage2-0.2, 
                     y_start_top, y_start_top - v_rank_filtered,
                     y_s2_filt_top, y_s2_filt_bot, 
                     c_red, c_green, alpha=0.4)
    
    # Flow 2: Total -> Rank Remain (Gradient Red to Red Light)
    draw_sankey_link(ax, x_stage1+0.2, x_stage2-0.2, 
                     y_start_top - v_rank_filtered, y_start_bot,
                     y_s2_rem_top, y_s2_rem_bot, 
                     c_red, c_red_light, alpha=0.4)
    
    # Flow 3: Rank Remain -> Judge Save Intercept (Gradient Red Light to Green)
    draw_sankey_link(ax, x_stage2+0.2, x_stage3-0.2,
                     y_s2_rem_top, y_s2_rem_top - v_judge_filtered,
                     y_s3_filt_top, y_s3_filt_bot,
                     c_red_light, c_green, alpha=0.4)
    
    # Flow 4: Rank Remain -> Final Resid (Gradient Red Light to Grey)
    draw_sankey_link(ax, x_stage2+0.2, x_stage3-0.2,
                     y_s2_rem_top - v_judge_filtered, y_s2_rem_bot,
                     y_s3_resid_top, y_s3_resid_bot,
                     c_red_light, c_grey, alpha=0.4)

    # === Drawing Blocks ===
    ax.add_patch(patches.Rectangle((x_stage1-0.2, y_start_bot), 0.4, v_total, color=c_red, zorder=3))
    ax.add_patch(patches.Rectangle((x_stage2-0.2, y_s2_filt_bot), 0.4, v_rank_filtered, color=c_green, zorder=3))
    ax.add_patch(patches.Rectangle((x_stage2-0.2, y_s2_rem_bot), 0.4, v_rank_remain, color=c_red_light, zorder=3))
    ax.add_patch(patches.Rectangle((x_stage3-0.2, y_s3_filt_bot), 0.4, v_judge_filtered, color=c_green, zorder=3))
    ax.add_patch(patches.Rectangle((x_stage3-0.2, y_s3_resid_bot), 0.4, v_resid, color=c_grey, zorder=3))

    # === Annotations (Outside bars/flows) ===
    
    # Stage 1
    ax.text(x_stage1, y_start_top + 1.5, "Initial Populism Risk\n(Percentage Rule)", ha='center', va='bottom', fontweight='heavy', fontsize=14, color='black')
    ax.text(x_stage1 + 0.4, y_start_bot + v_total/2, "22.5%", ha='left', va='center', fontweight='heavy', fontsize=18, color='black')
    
    # Stage 2
    ax.text(x_stage2, y_s2_filt_top + 1.0, "Intercepted by\nRank Rule", ha='center', va='bottom', fontweight='heavy', fontsize=13, color='black')
    ax.text(x_stage2 + 0.4, y_s2_filt_bot + v_rank_filtered/2, "7.3%", ha='left', va='center', fontweight='heavy', fontsize=18, color='black')
    
    ax.text(x_stage2, y_s2_rem_bot - 1.5, "Remaining Concern\n(Rank System)", ha='center', va='top', fontweight='heavy', fontsize=13, color='black')
    ax.text(x_stage2 + 0.4, y_s2_rem_bot + v_rank_remain/2, "15.2%", ha='left', va='center', fontweight='heavy', fontsize=18, color='black')
    
    # Stage 3
    ax.text(x_stage3, y_s3_filt_top + 1.0, "Intercepted by\nJudge Save", ha='center', va='bottom', fontweight='heavy', fontsize=13, color='black')
    ax.text(x_stage3 + 0.4, y_s3_filt_bot + v_judge_filtered/2, "7.9%", ha='left', va='center', fontweight='heavy', fontsize=18, color='black')
    
    ax.text(x_stage3, y_s3_resid_bot - 1.5, "Residual Controversy\n(Unavoidable)", ha='center', va='top', fontweight='heavy', fontsize=13, color='black')
    ax.text(x_stage3 + 0.4, y_s3_resid_bot + v_resid/2, "7.3%", ha='left', va='center', fontweight='heavy', fontsize=18, color='black')

    # Title
    ax.text(5, 27.5, "Controversy Filtration: From Populism to Professionalism", ha='center', va='center', fontweight='heavy', fontsize=20)

    # Save
    pdf_path = os.path.join(output_dir, 'controversy_sankey.pdf')
    png_path = os.path.join(output_dir, 'controversy_sankey.png')
    svg_path = os.path.join(output_dir, 'controversy_sankey.svg')
    
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    print(f"Diagram saved to:\n- {pdf_path}\n- {png_path}\n- {svg_path}")

if __name__ == "__main__":
    plot_controversy_sankey()
