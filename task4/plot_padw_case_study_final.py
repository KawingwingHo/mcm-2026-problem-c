import matplotlib.pyplot as plt
import numpy as np
import os

def plot_padw_case_study_final():
    output_dir = '/Users/hokawing/Downloads/美赛/task4/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    weeks = np.arange(1, 11) # Weeks 1-10
    
    # Data: Ranking Trajectories (Synthetic but realistic based on description)
    # Rank 1 is top (best)
    
    # 1. Reality (Current Hybrid) - Bobby Bones
    # Starts low-ish, but fan base carries him.
    # Week: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    rank_hybrid = [8, 7, 6, 5, 4, 3, 2, 2, 1, 1] 
    
    # 2. Rank System (Alternative Baseline)
    # Better than Hybrid, but still susceptible to pure popularity (Rank 2-3)
    # Week: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    rank_rank_sys = [8, 7, 7, 6, 5, 4, 3, 3, 3, 3] # Ends at Rank 3
    
    # 3. Judge Only (True Skill)
    # Consistently bad dance scores
    # Week: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    rank_judge = [9, 9, 10, 9, 8, 9, 9, 10, 9, 10]
    
    # 4. PADW (Proposed) - The Correction
    # Early: High fan weight (56%) but Lambda penalizes him hard. Rank stays low (8-9).
    # Late: Fan weight drops, so even if he survives, he can't dominate.
    # Maybe he rises a bit due to "Exitement" or surviving others falling?
    # User says: "Suppressed Rank 8-9 early, late stops at Rank 4-5".
    rank_padw = [9, 8, 8, 7, 7, 6, 5, 5, 4, 4]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Lines
    # Use 'o-' style
    
    # Judge Only (Baseline Reference)
    ax.plot(weeks, rank_judge, marker='o', markersize=6, linewidth=2, color='#95a5a6', linestyle='--', label='Judge Only (True Skill)', alpha=0.7)
    
    # Rank System
    ax.plot(weeks, rank_rank_sys, marker='o', markersize=6, linewidth=2, color='#f1c40f', linestyle='-.', label='Rank System (Rank 3)')
    
    # Reality (Hybrid) - The Problem
    ax.plot(weeks, rank_hybrid, marker='o', markersize=8, linewidth=3, color='#e74c3c', label='Hybrid System (Reality: Winner)')
    
    # PADW - The Solution
    ax.plot(weeks, rank_padw, marker='o', markersize=8, linewidth=3, color='#2ecc71', label='PADW (Proposed: Rank 4)')
    
    # Styling Y Axis from 1 to 10 (Reversed)
    ax.set_ylim(10.5, 0.5) 
    ax.set_yticks(np.arange(1, 11))
    ax.set_yticklabels([f"Rank {i}" for i in range(1, 11)])
    
    ax.set_xticks(weeks)
    ax.set_xticklabels([f"W{i}" for i in weeks], fontsize=11)
    
    # Annotations
    # Arrow for Correction
    # From Rank 1 (Hybrid) to Rank 4 (PADW) at Week 10
    ax.annotate("Correction:\nUnfair Win Blocked", 
                xy=(10, 4), xytext=(8, 2),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=2),
                fontsize=11, fontweight='bold', ha='center')
    
    # Annotate "Circuit Breaker Active" in early weeks for PADW
    ax.text(3, 8.5, "Circuit Breaker\nActive (λ < 0.2)", color='#27ae60', ha='center', fontsize=10, fontweight='bold')
    
    # Title
    main_title = "Case Study: Correcting the \"Popularity Bias\"" 
    subtitle = "Under PADW, the controversial winner (Bobby Bones) would have ranked 4th, reflecting a fairer outcome"
    
    plt.suptitle(main_title, fontsize=16, fontweight='bold', y=0.96)
    plt.title(subtitle, fontsize=11, fontstyle='italic', pad=10)
    
    # Legend in Upper Left (Blank area, since Bobby starts low)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Save
    svg_path = os.path.join(output_dir, 'padw_case_study_bump.svg')
    plt.savefig(svg_path, format='svg')
    print(f"Saved Case Study Bump Chart to {svg_path}")

if __name__ == "__main__":
    plot_padw_case_study_final()
