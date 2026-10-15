import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def plot_padw_architecture_final():
    output_dir = '/Users/hokawing/Downloads/美赛/task4/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Coordinates
    x_input = 0.1
    x_process_A = 0.4
    x_process_B = 0.65
    x_output = 0.9
    
    y_judge = 0.7
    y_fan = 0.3
    y_mid = 0.5
    
    # --------------------------
    # 1. Inputs
    # --------------------------
    
    # Judge Input (Green)
    ax.add_patch(patches.FancyBboxPatch((0.02, y_judge-0.08), 0.15, 0.16, facecolor='#2ecc71', edgecolor='black', alpha=0.8, boxstyle="Round,pad=0.02"))
    ax.text(x_input, y_judge, "Judge Scores\n(Professionalism)", ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Fan Input (Orange)
    ax.add_patch(patches.FancyBboxPatch((0.02, y_fan-0.08), 0.15, 0.16, facecolor='#e67e22', edgecolor='black', alpha=0.8, boxstyle="Round,pad=0.02"))
    ax.text(x_input, y_fan, "Fan Votes\n(Popularity)", ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # --------------------------
    # 2. Processing Modules (Black Box)
    # --------------------------
    
    # Large Container for PADW System
    ax.add_patch(patches.Rectangle((x_process_A - 0.1, 0.1), (x_process_B - x_process_A) + 0.3, 0.8, fill=False, edgecolor='#34495e', linewidth=2, linestyle='--'))
    ax.text((x_process_A + x_process_B + 0.1)/2, 0.93, "PADW System Logic", ha='center', fontsize=14, fontstyle='italic', color='#34495e')
    
    # Module A: Skill Anchor (Lambda) - Funnel Shape
    # Polygon vertices for a funnel
    funnel_x = [x_process_A - 0.05, x_process_A + 0.05, x_process_A + 0.02, x_process_A - 0.02]
    # Actually funnel usually goes wide top/left to narrow bottom/right.
    # Let's make it vertical funnel or horizontal?
    # Fan path flows horizontal. So horizontal funnel.
    
    # Fan Stream Arrow to Funnel
    ax.annotate("", xy=(x_process_A - 0.06, y_fan), xytext=(0.18, y_fan),
                arrowprops=dict(arrowstyle="simple", facecolor='#e67e22', alpha=0.5))
    
    # Funnel Body
    # Wide at left, narrow at right
    funnel_poly = [
        [x_process_A - 0.06, y_fan + 0.1], # Top Left
        [x_process_A + 0.06, y_fan + 0.03], # Top Right (output)
        [x_process_A + 0.06, y_fan - 0.03], # Bottom Right (output)
        [x_process_A - 0.06, y_fan - 0.1], # Bottom Left
    ]
    ax.add_patch(patches.Polygon(funnel_poly, closed=True, facecolor='#9b59b6', edgecolor='black', alpha=0.8))
    ax.text(x_process_A, y_fan, "Module A\nSkill Anchor\nFilter (λ)", ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Bobby Bones Case Annotation
    ax.annotate('Bobby Bones Case:\nLow Skill → High Filters\n(Fan Vote Blocked)', 
                xy=(x_process_A - 0.06, y_fan - 0.05), xytext=(x_process_A - 0.15, y_fan - 0.25),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color='red', lw=2),
                bbox=dict(boxstyle="round,pad=0.5", fc="#ffdddd", ec="red", alpha=0.9), fontsize=10, color='red')
    
    # Module B: Time Decay (Alpha) - Timeline
    # Rectangle around Judge Path
    # Judge Stream Arrow (Passes through, but weighted)
    ax.annotate("", xy=(x_process_B - 0.08, y_judge), xytext=(0.18, y_judge),
                arrowprops=dict(arrowstyle="simple", facecolor='#2ecc71', alpha=0.5))
    
    # Timeline Box
    ax.add_patch(patches.FancyBboxPatch((x_process_B - 0.08, y_mid - 0.15), 0.16, 0.4, facecolor='#3498db', edgecolor='black', alpha=0.8, boxstyle="Round,pad=0.02"))
    ax.text(x_process_B, y_mid + 0.05, "Module B\nTime-Dynamic\nWeighting α(t)", ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Draw mini timeline scale inside
    ax.plot([x_process_B - 0.06, x_process_B + 0.06], [y_mid - 0.05, y_mid - 0.05], color='white', lw=2)
    ax.text(x_process_B, y_mid - 0.08, "W1 → W11", color='white', ha='center', fontsize=9)
    
    # Internal Logic Connections
    # Filtered Fan Vote goes to Module B? Or Mixer?
    # Usually Mixer is the end.
    # Let's say Module B regulates the MIX.
    # So both streams go into Module B (The Mixer).
    
    # Route Fan from Funnel to Mixer
    ax.annotate("", xy=(x_process_B - 0.08, y_mid - 0.05), xytext=(x_process_A + 0.06, y_fan),
                arrowprops=dict(arrowstyle="simple", facecolor='#e67e22', alpha=0.5))
    
    # Route Judge to Mixer (already drawn)
    # Actually Judge connects to Mixer too.
    ax.annotate("", xy=(x_process_B - 0.08, y_mid + 0.05), xytext=(0.4, y_judge), # From hypothetical middle point
                arrowprops=dict(arrowstyle="simple", facecolor='#2ecc71', alpha=0.5))
    
    # --------------------------
    # 3. Output
    # --------------------------
    
    # Output Arrow
    ax.annotate("", xy=(0.95, y_mid), xytext=(x_process_B + 0.08, y_mid),
                arrowprops=dict(arrowstyle="simple", facecolor='#34495e', alpha=0.8))
                
    ax.text(0.9, y_mid, "Final\nScore", ha='center', va='center', fontsize=14, fontweight='bold')
    
    # --------------------------
    # Title & Styling
    # --------------------------
    
    title = "Architecture of the Performance-Anchored Dynamic Weighting (PADW) System"
    note = "Note: The system integrates a skill-anchored filter ($\lambda$) and time-dynamic weighting ($\\alpha(t)$) to mitigate popularity bias."
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
    plt.figtext(0.5, 0.02, note, ha='center', fontsize=11, fontstyle='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Save
    svg_path = os.path.join(output_dir, 'padw_architecture.svg')
    pdf_path = os.path.join(output_dir, 'padw_architecture.pdf')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved Architecture Diagram to {svg_path} and {pdf_path}")

if __name__ == "__main__":
    plot_padw_architecture_final()
