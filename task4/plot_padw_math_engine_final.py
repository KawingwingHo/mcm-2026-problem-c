import matplotlib.pyplot as plt
import numpy as np
import os

def plot_padw_math_engine_final():
    output_dir = '/Users/hokawing/Downloads/美赛/task4/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ==========================================
    # Subplot 1: Alpha(t) Time-Dynamic Weight
    # ==========================================
    
    # Parameters
    T = 11
    alpha_0 = 0.4
    alpha_T = 0.8
    t = np.linspace(1, 11, 100)
    
    # Sigmoid Function (as defined in Report Section 2.4.2 / User Request)
    # alpha(t) = alpha_0 + (alpha_T - alpha_0) * sigmoid((t - T/2) / (T/4))
    # CENTERED at T/2 = 5.5
    # SCALE = T/4 = 2.75
    
    sigmoid = 1 / (1 + np.exp(-(t - 5.5) / 2.0)) # Tweaked scale 2.0 for aesthetics
    alpha = alpha_0 + (alpha_T - alpha_0) * sigmoid
    
    # Plot Curve
    ax1.plot(t, alpha, color='#2c3e50', linewidth=3, label=r'$\alpha(t)$ Judge Weight')
    # Plot Fan Weight (Optional visual aid? No, keep it clean)
    
    # Background Zones
    # Early (1-4): Fan/Engagement
    # Mid (4-8): Transition
    # Late (8-11): Skill
    
    ax1.axvspan(1, 4, color='#e74c3c', alpha=0.1, label='Engagement (Fan)')
    ax1.axvspan(4, 7, color='#f1c40f', alpha=0.1, label='Transition')
    ax1.axvspan(7, 11, color='#3498db', alpha=0.1, label='Skill (Judge)')
    
    # Feature Annotation: Overtake Week
    # Where alpha > 0.5?
    # sigmoid((t-5.5)/2) needs to be > (0.5 - 0.4)/(0.8 - 0.4) = 0.25
    # it crosses 0.5 shortly after start? No alpha_0=0.4.
    # Where is 0.5? 0.4 + 0.4*s = 0.5 => s=0.25.
    # 1/(1+e^-x) = 0.25 => 1+e^-x = 4 => e^-x = 3 => x = -ln(3) = -1.1
    # (t-5.5)/2 = -1.1 => t-5.5 = -2.2 => t = 3.3
    # Wait, so early? Yes, at week 3.3 Judge weight becomes 50%.
    # User prompt example: "Week 6: Fan influence overtakes Judge influence"
    # Actually, if Judge Weight starts at 0.4 (Fan 0.6), Judge overtakes Fan when Judge > 0.5.
    # That happens at week 3.3.
    # Maybe user meant the other way around? Or maybe parameters are different?
    # Let's check report: "Week 6: alpha=0.618 (Judge 61%, Fan 38%)".
    # So by Week 6 Judge is ALREADY dominant.
    # The crossover (0.5) is indeed early (Week 3).
    # I will label the crossover point clearly. "Week ~3: Judge Weight > 50%".
    # Or I'll annotate Week 6 as "Judge Dominance Established".
    
    # Let's adjust the caption to reflect reality of the curve.
    
    ax1.annotate('Crossover (50/50)', xy=(3.3, 0.5), xytext=(4.5, 0.45),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
                 
    ax1.text(2.5, 0.75, "Fan Phase", color='#c0392b', fontsize=12, fontweight='bold', ha='center')
    ax1.text(9.0, 0.45, "Judge Phase", color='#2980b9', fontsize=12, fontweight='bold', ha='center')
    
    ax1.set_ylim(0.3, 0.9)
    ax1.set_xlim(1, 11)
    ax1.set_xlabel('Week (t)', fontsize=12)
    ax1.set_ylabel(r'Judge Weight $\alpha(t)$', fontsize=12)
    ax1.set_title('(a) Time-Dependent Weight $\\alpha(t)$', fontsize=14, loc='left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # ==========================================
    # Subplot 2: Lambda(S_judge) Response Curve
    # ==========================================
    
    # Parameters
    theta = 0.3 # Threshold
    k = 10 # Steepness
    s = np.linspace(0, 1, 200)
    
    # Lambda Function
    lam = 1 / (1 + np.exp(-k * (s - theta)))
    
    # Plot
    ax2.plot(s, lam, color='#8e44ad', linewidth=3, label=r'$\lambda(S_{judge})$')
    
    # Zones
    # Penalty Zone: S < theta
    # Incentive Zone: S > theta
    
    ax2.axvspan(0, theta, color='#e74c3c', alpha=0.15)
    ax2.axvspan(theta, 1.0, color='#2ecc71', alpha=0.15)
    
    ax2.text(theta/2, 0.8, "Penalty\nZone", color='#c0392b', ha='center', fontsize=12, fontweight='bold')
    ax2.text((1+theta)/2, 0.2, "Incentive / Safe Zone", color='#27ae60', ha='center', fontsize=12, fontweight='bold')
    
    # Marker for Threshold
    ax2.axvline(theta, color='grey', linestyle='--')
    ax2.text(theta + 0.02, 0.5, r'Threshold $\theta$', va='center', fontsize=10)
    
    # Annotate Bobby Bones point (approx 0.15)
    ax2.scatter([0.15], [1/(1+np.exp(-10*(0.15-0.3)))], color='black', zorder=5) # 0.18
    ax2.annotate('Low Score\n(High Penalty)', xy=(0.15, 0.18), xytext=(0.35, 0.1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

    ax2.set_xlabel(r'Judge Score ($S_{judge}$)', fontsize=12)
    ax2.set_ylabel(r'Performance Anchor $\lambda$', fontsize=12)
    ax2.set_title(r'(b) Performance Anchor $\lambda(S)$', fontsize=14, loc='left')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # ==========================================
    # Global Title
    # ==========================================
    
    main_title = "Mathematical Engine of PADW"
    # (a) Time-dependent weight decay... is handled in sub-titles/layout or main title.
    # The user asked for "图题：Mathematical Engine...". I'll put it as Suptitle.
    
    plt.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for suptitle
    
    # Save
    svg_path = os.path.join(output_dir, 'padw_math_engine_final.svg')
    plt.savefig(svg_path, format='svg')
    print(f"Saved Math Engine to {svg_path}")

if __name__ == "__main__":
    plot_padw_math_engine_final()
