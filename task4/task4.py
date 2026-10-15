"""
Task 4: 新机制设计——动态表现锚定系统 (鲁棒增强版 v2.0)
======================================================
Performance-Anchored Dynamic Weighting with Time-Dynamic (PADW-TD v2.0)

核心升级:
---------
1. 集成 λ: 多阈值平均，降低阈值敏感度
2. 自适应 k: 根据选手数量调整陡峭因子
3. Sigmoid α(t): 非线性时间动态权重

最终得分公式 (PADW-TD v2.0):
Score_final = α(t) * S_judge + (1 - α(t)) * λ_ensemble * S_fan
"""

import pandas as pd
import numpy as np
import warnings
import os
from scipy import stats

warnings.filterwarnings('ignore')

# =============================================================================
# 参数设置 (鲁棒增强版)
# =============================================================================
ALPHA = 0.3  # Pareto 分布参数

# 多阈值集成参数 (鲁棒性增强)
THRESHOLD_QUANTILES = [0.15, 0.20, 0.25]  # 多阈值集成

# 自适应陡峭因子参数
K_BASE = 10  # 基准陡峭因子
K_EARLY = 8   # 早期更温和
K_LATE = 15   # 后期更陡峭

# 时间动态权重参数 (Sigmoid 版)
ALPHA_EARLY = 0.45  # 早期评委权重
ALPHA_LATE = 0.90   # 后期评委权重
TOTAL_WEEKS = 11

# =============================================================================
# 鲁棒性增强函数
# =============================================================================
def load_data():
    """加载并准备数据"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_dir, 'task1', 'cleaned_data_final.csv'))
    
    df['week_out'] = df['results'].str.extract(r'Eliminated Week (\d+)').astype(float)
    max_week = df['week_out'].max()
    if pd.isna(max_week):
        max_week = 11
    df['week_out'] = df['week_out'].fillna(max_week + 1)
    
    for season in df['season'].unique():
        mask = df['season'] == season
        sdata = df[mask].copy()
        sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
        sdata['s_fan'] = (1.0 / sdata['rank']) ** ALPHA
        s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
        if s_max > s_min:
            sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)
        else:
            sdata['s_fan'] = 0.5
        df.loc[mask, 's_fan'] = sdata['s_fan']
    
    df['s_judge'] = df['avg_normalized_score']
    return df

def compute_performance_coefficient(s_judge, threshold, k):
    """计算表现系数 λ (Sigmoid 软熔断)"""
    return 1.0 / (1.0 + np.exp(-k * (s_judge - threshold)))

def compute_ensemble_lambda(s_judge, k, quantiles=THRESHOLD_QUANTILES):
    """
    计算集成 λ (多阈值平均) - 鲁棒性增强
    
    通过多个阈值的 λ 取平均，降低单一阈值敏感度
    """
    lambdas = []
    for q in quantiles:
        threshold = s_judge.quantile(q)
        lam = compute_performance_coefficient(s_judge, threshold, k)
        lambdas.append(lam)
    return np.mean(lambdas, axis=0)

def adaptive_k(n_contestants):
    """
    自适应陡峭因子 - 鲁棒性增强
    
    早期选手多: 更温和 (k=8)
    后期选手少: 更陡峭 (k=15)
    """
    if n_contestants >= 10:
        return K_EARLY
    elif n_contestants >= 5:
        return K_BASE
    return K_LATE

def compute_time_dynamic_weight_sigmoid(t, T=TOTAL_WEEKS, alpha_0=ALPHA_EARLY, alpha_T=ALPHA_LATE):
    """
    Sigmoid 时间动态权重 - 鲁棒性增强
    
    使用 Sigmoid 替代线性插值，中期变化更快，两端更稳定
    α(t) = α_0 + (α_T - α_0) * σ((t - T/2) / (T/4))
    """
    x = (t - T/2) / (T/4)
    sigmoid = 1 / (1 + np.exp(-x))
    return alpha_0 + (alpha_T - alpha_0) * sigmoid

def compute_time_dynamic_weight_linear(t, T=TOTAL_WEEKS, alpha_0=ALPHA_EARLY, alpha_T=ALPHA_LATE):
    """线性时间动态权重 (原版)"""
    return alpha_0 + (alpha_T - alpha_0) * (t / T)


def simulate_padw_system(df, k=K_BASE, q=0.20):
    """
    模拟 PADW v2.0 (鲁棒增强版)
    """
    results = []
    
    for season in df['season'].unique():
        sdata = df[df['season'] == season].copy()
        if len(sdata) < 3:
            continue
        
        # 自适应 k
        k = adaptive_k(len(sdata))
        
        # 计算集成 λ
        sdata['lambda'] = compute_ensemble_lambda(sdata['s_judge'], k)
        
        # PADW 得分
        sdata['score_padw'] = sdata['s_judge'] + sdata['lambda'] * sdata['s_fan']
        
        # 淘汰者 (得分最低)
        eliminated_idx = sdata['score_padw'].idxmin()
        eliminated_name = sdata.loc[eliminated_idx, 'celebrity_name']
        eliminated_judge = sdata.loc[eliminated_idx, 's_judge']
        eliminated_lambda = sdata.loc[eliminated_idx, 'lambda']
        
        results.append({
            'season': season,
            'k': k,
            'eliminated': eliminated_name,
            'eliminated_judge_score': eliminated_judge,
            'eliminated_lambda': eliminated_lambda,
            'padw_scores': sdata['score_padw'].tolist()
        })
    
    return results, df

def compare_five_systems(df):
    """
    对比五种赛制系统：
    1. 排名制 (Rank Rule)
    2. 百分比制 (Percentage Rule)
    3. 混合制 (Rank + Judge Save)
    4. PADW (新提案 - 静态版)
    5. PADW-TD (新提案 - 时间动态版)
    """
    print("=" * 70)
    print("Task 4: 五种赛制系统对比分析")
    print("=" * 70)
    
    metrics = {
        'Rank': {'skill_corr': [], 'controversy': 0, 'low_score_survive': 0},
        'Pct': {'skill_corr': [], 'controversy': 0, 'low_score_survive': 0},
        'Hybrid': {'skill_corr': [], 'controversy': 0, 'low_score_survive': 0},
        'PADW': {'skill_corr': [], 'controversy': 0, 'low_score_survive': 0},
        'PADW-TD': {'skill_corr': [], 'controversy': 0, 'low_score_survive': 0}
    }
    
    total_seasons = 0
    
    for season in df['season'].unique():
        sdata = df[df['season'] == season].copy()
        if len(sdata) < 3:
            continue
        
        total_seasons += 1
        
        # 1. 排名制
        sdata['rank_judge'] = sdata['s_judge'].rank(ascending=False)
        sdata['rank_fan'] = sdata['s_fan'].rank(ascending=False)
        sdata['score_rank'] = sdata['rank_judge'] + sdata['rank_fan']
        elim_rank = sdata.loc[sdata['score_rank'].idxmax(), 'celebrity_name']
        elim_rank_judge = sdata.loc[sdata['score_rank'].idxmax(), 's_judge']
        
        # 2. 百分比制
        sdata['score_pct'] = 0.5 * sdata['s_judge'] + 0.5 * sdata['s_fan']
        elim_pct = sdata.loc[sdata['score_pct'].idxmin(), 'celebrity_name']
        elim_pct_judge = sdata.loc[sdata['score_pct'].idxmin(), 's_judge']
        
        # 3. 混合制 (Rank + Judge Save)
        bottom2_rank = sdata.nlargest(2, 'score_rank')
        if len(bottom2_rank) >= 2:
            # 评委选择淘汰评委分低的
            elim_hybrid = bottom2_rank.loc[bottom2_rank['s_judge'].idxmin(), 'celebrity_name']
            elim_hybrid_judge = bottom2_rank['s_judge'].min()
        else:
            elim_hybrid = elim_rank
            elim_hybrid_judge = elim_rank_judge
        
        # 4. PADW v2.0 (鲁棒版 - 集成λ + 自适应k)
        k = adaptive_k(len(sdata))
        sdata['lambda'] = compute_ensemble_lambda(sdata['s_judge'], k)
        sdata['score_padw'] = sdata['s_judge'] + sdata['lambda'] * sdata['s_fan']
        elim_padw = sdata.loc[sdata['score_padw'].idxmin(), 'celebrity_name']
        elim_padw_judge = sdata.loc[sdata['score_padw'].idxmin(), 's_judge']
        
        # 5. PADW-TD v2.0 (鲁棒版 - Sigmoid α(t))
        mid_week = TOTAL_WEEKS // 2
        alpha_t = compute_time_dynamic_weight_sigmoid(mid_week)
        sdata['score_padw_td'] = alpha_t * sdata['s_judge'] + (1 - alpha_t) * sdata['lambda'] * sdata['s_fan']
        elim_padw_td = sdata.loc[sdata['score_padw_td'].idxmin(), 'celebrity_name']
        elim_padw_td_judge = sdata.loc[sdata['score_padw_td'].idxmin(), 's_judge']
        
        # 统计低分选手存活 (评委分后20%但没被淘汰)
        low_threshold = sdata['s_judge'].quantile(0.2)
        low_scorers = sdata[sdata['s_judge'] <= low_threshold]['celebrity_name'].tolist()
        
        if elim_rank not in low_scorers and len(low_scorers) > 0:
            metrics['Rank']['low_score_survive'] += 1
        if elim_pct not in low_scorers and len(low_scorers) > 0:
            metrics['Pct']['low_score_survive'] += 1
        if elim_hybrid not in low_scorers and len(low_scorers) > 0:
            metrics['Hybrid']['low_score_survive'] += 1
        if elim_padw not in low_scorers and len(low_scorers) > 0:
            metrics['PADW']['low_score_survive'] += 1
        if elim_padw_td not in low_scorers and len(low_scorers) > 0:
            metrics['PADW-TD']['low_score_survive'] += 1
        
        # 统计争议 (低评委分但高粉丝分的选手夺冠)
        # 简化: 淘汰的选手评委分是否较高
        median_judge = sdata['s_judge'].median()
        if elim_rank_judge > median_judge:
            metrics['Rank']['controversy'] += 1
        if elim_pct_judge > median_judge:
            metrics['Pct']['controversy'] += 1
        if elim_hybrid_judge > median_judge:
            metrics['Hybrid']['controversy'] += 1
        if elim_padw_judge > median_judge:
            metrics['PADW']['controversy'] += 1
        if elim_padw_td_judge > median_judge:
            metrics['PADW-TD']['controversy'] += 1
        
        # 计算 Skill Correlation
        for system, score_col in [('Rank', 'score_rank'), ('Pct', 'score_pct'), 
                                   ('Hybrid', 'score_rank'), ('PADW', 'score_padw'),
                                   ('PADW-TD', 'score_padw_td')]:
            if score_col == 'score_rank':
                # 排名制分数越高越差，需要反向
                corr, _ = stats.spearmanr(-sdata[score_col], sdata['s_judge'])
            else:
                corr, _ = stats.spearmanr(sdata[score_col], sdata['s_judge'])
            metrics[system]['skill_corr'].append(corr)
    
    # 汇总结果
    print(f"\n样本量: {total_seasons} 个赛季")
    print("\n" + "-" * 70)
    print("【五种赛制综合对比】")
    print("-" * 70)
    
    summary = []
    for system in ['Rank', 'Pct', 'Hybrid', 'PADW', 'PADW-TD']:
        avg_corr = np.mean(metrics[system]['skill_corr'])
        controversy_rate = metrics[system]['controversy'] / total_seasons * 100
        low_survive_rate = metrics[system]['low_score_survive'] / total_seasons * 100
        
        summary.append({
            'System': system,
            'Skill_Corr': avg_corr,
            'Controversy_Rate': controversy_rate,
            'Low_Score_Survive': low_survive_rate
        })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    # 打印时间动态权重信息
    print("\n" + "-" * 70)
    print("【时间动态权重 α(t) 曲线】")
    print("-" * 70)
    print(f"参数: α_0 = {ALPHA_EARLY}, α_T = {ALPHA_LATE}, T = {TOTAL_WEEKS}")
    for t in [1, 3, 6, 9, 11]:
        alpha_t = compute_time_dynamic_weight_sigmoid(t)
        print(f"  Week {t:2d}: α(t) = {alpha_t:.3f} (评委:{alpha_t*100:.1f}%, 粉丝:{(1-alpha_t)*100:.1f}%)")
    
    return summary_df, metrics

def analyze_padw_mechanism(df):
    """
    深度分析 PADW 机制的效果
    """
    print("\n" + "=" * 70)
    print("【PADW 机制效果分析】")
    print("=" * 70)
    
    # 1. λ 分布分析 (使用集成 λ)
    all_lambdas = []
    all_judges = []
    
    for season in df['season'].unique():
        sdata = df[df['season'] == season].copy()
        if len(sdata) < 3:
            continue
        
        k = adaptive_k(len(sdata))
        sdata['lambda'] = compute_ensemble_lambda(sdata['s_judge'], k)
        all_lambdas.extend(sdata['lambda'].tolist())
        all_judges.extend(sdata['s_judge'].tolist())
    
    # λ 分布统计
    lambdas = np.array(all_lambdas)
    print(f"\n表现系数 λ 分布统计:")
    print(f"  均值: {np.mean(lambdas):.4f}")
    print(f"  中位数: {np.median(lambdas):.4f}")
    print(f"  标准差: {np.std(lambdas):.4f}")
    print(f"  λ < 0.5 比例: {(lambdas < 0.5).mean()*100:.1f}% (粉丝票被显著削弱)")
    print(f"  λ > 0.9 比例: {(lambdas > 0.9).mean()*100:.1f}% (粉丝票几乎不受影响)")
    
    # 2. 案例分析
    print("\n" + "-" * 70)
    print("【争议案例模拟】")
    print("-" * 70)
    
    # 模拟 Bobby Bones 场景
    bobby_judge = 0.15  # 极低评委分
    bobby_fan = 0.95    # 极高粉丝票
    threshold_sim = 0.3  # 假设熔断阈值
    k_sim = K_BASE
    
    lambda_bobby = compute_performance_coefficient(bobby_judge, threshold_sim, k_sim)
    score_pct_bobby = 0.5 * bobby_judge + 0.5 * bobby_fan
    score_padw_bobby = bobby_judge + lambda_bobby * bobby_fan
    
    print(f"\n场景: Bobby Bones 类型选手")
    print(f"  评委分: {bobby_judge:.2f} (极低)")
    print(f"  粉丝票: {bobby_fan:.2f} (极高)")
    print(f"  熔断阈值 θ: {threshold_sim:.2f}")
    print(f"  表现系数 λ: {lambda_bobby:.4f}")
    print(f"  百分比制得分: {score_pct_bobby:.4f}")
    print(f"  PADW 得分: {score_padw_bobby:.4f}")
    print(f"  → PADW 使其得分下降 {(score_pct_bobby - score_padw_bobby)/score_pct_bobby*100:.1f}%")
    
    # 模拟技术型选手场景
    tech_judge = 0.75   # 高评委分
    tech_fan = 0.35     # 低粉丝票
    
    lambda_tech = compute_performance_coefficient(tech_judge, threshold_sim, k_sim)
    score_pct_tech = 0.5 * tech_judge + 0.5 * tech_fan
    score_padw_tech = tech_judge + lambda_tech * tech_fan
    
    print(f"\n场景: 技术型选手")
    print(f"  评委分: {tech_judge:.2f} (高)")
    print(f"  粉丝票: {tech_fan:.2f} (低)")
    print(f"  表现系数 λ: {lambda_tech:.4f}")
    print(f"  百分比制得分: {score_pct_tech:.4f}")
    print(f"  PADW 得分: {score_padw_tech:.4f}")
    print(f"  → PADW 使其得分提升 {(score_padw_tech - score_pct_tech)/score_pct_tech*100:.1f}%")

def generate_producer_memo(summary_df):
    """
    生成给制作人的备忘录
    """
    print("\n" + "=" * 70)
    print("【致节目制作人备忘录】")
    print("=" * 70)
    
    padw_row = summary_df[summary_df['System'] == 'PADW'].iloc[0]
    pct_row = summary_df[summary_df['System'] == 'Pct'].iloc[0]
    
    print(f"""
尊敬的制作人：

基于对34个赛季数据的定量分析，我们提议采用"动态表现锚定系统"(PADW)
取代现有的百分比制。以下是我们的建议及理由：

【为什么现有百分比制存在问题？】
1. 专业相关度仅为 {pct_row['Skill_Corr']:.4f}，冠军含金量受质疑
2. 争议案例发生率 {pct_row['Controversy_Rate']:.1f}%，损害节目公信力
3. Bobby Bones 式争议持续发酵，社交媒体负面舆论

【为什么选择 PADW？】
1. 专业相关度提升至 {padw_row['Skill_Corr']:.4f} (+{(padw_row['Skill_Corr']-pct_row['Skill_Corr'])*100:.1f}%)
2. 争议案例发生率降至 {padw_row['Controversy_Rate']:.1f}% (降幅 {pct_row['Controversy_Rate']-padw_row['Controversy_Rate']:.1f}%)
3. 低分选手存活率控制在 {padw_row['Low_Score_Survive']:.1f}%

【PADW 的商业价值】
1. 刺激投票量：粉丝为了抵消 λ 惩罚需投更多票
2. 制造悬念：技术型选手不再被"民粹碾压"淘汰
3. 维护口碑：减少"黑幕"质疑，提升节目专业形象

【实施建议】
1. 熔断阈值 θ 设为当周评委分的后20%分位
2. 陡峭因子 k=10，实现"软着陆"而非"一刀切"
3. 对观众公开规则，强调"技术决定票值"

综上，PADW 是一个兼顾公平性与商业性的创新机制。

此致
敬礼
""")

def main():
    print("=" * 70)
    print("Task 4: 新机制设计分析报告")
    print("Performance-Anchored Dynamic Weighting with Time-Dynamic (PADW-TD)")
    print("=" * 70)
    
    # 加载数据
    df = load_data()
    print(f"\n数据加载完成: {len(df)} 条记录, {df['season'].nunique()} 个赛季")
    
    # 五种系统对比
    summary_df, metrics = compare_five_systems(df)
    
    
    # PADW 机制分析
    analyze_padw_mechanism(df)
    
    # 生成备忘录
    generate_producer_memo(summary_df)
    
    # 保存结果
    output_path = os.path.dirname(os.path.abspath(__file__))
    summary_df.to_csv(os.path.join(output_path, 'task4_system_comparison.csv'), index=False)
    print(f"\n结果已保存至: task4_system_comparison.csv")

if __name__ == "__main__":
    main()
