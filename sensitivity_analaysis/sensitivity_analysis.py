"""
灵敏度分析模块：Task 1, 3, 4 综合分析
==========================================
Sensitivity Analysis for DWTS Competition Models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 数据加载
# =============================================================================
def load_data():
    """加载数据集"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_dir, 'task1', 'cleaned_data_final.csv'))
    
    df['week_out'] = df['results'].str.extract(r'Eliminated Week (\d+)').astype(float)
    max_week = df['week_out'].max()
    if pd.isna(max_week):
        max_week = 11
    df['week_out'] = df['week_out'].fillna(max_week + 1)
    df['s_judge'] = df['avg_normalized_score']
    
    return df

# =============================================================================
# Task 1 灵敏度分析
# =============================================================================
def task1_sensitivity_alpha(df, alpha_range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
    """Task 1: 帕累托参数 α 灵敏度分析"""
    print("=" * 70)
    print("Task 1 灵敏度分析: 帕累托参数 α")
    print("=" * 70)
    
    results = []
    
    for alpha in alpha_range:
        correct = 0
        total = 0
        
        for season in df['season'].unique():
            sdata = df[df['season'] == season].copy()
            if len(sdata) < 3:
                continue
            
            sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
            sdata['s_fan'] = (1.0 / sdata['rank']) ** alpha
            s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
            if s_max > s_min:
                sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)
            else:
                sdata['s_fan'] = 0.5
            
            w_j = 0.03
            sdata['score'] = w_j * sdata['s_judge'] + (1 - w_j) * sdata['s_fan']
            
            predicted = sdata.loc[sdata['score'].idxmin(), 'celebrity_name']
            actual = sdata.loc[sdata['week_out'].idxmin(), 'celebrity_name']
            
            if predicted == actual:
                correct += 1
            total += 1
        
        accuracy = correct / total * 100 if total > 0 else 0
        results.append({'alpha': alpha, 'accuracy': accuracy, 'correct': correct, 'total': total})
        print(f"α = {alpha:.2f}: 准确率 = {accuracy:.2f}%")
    
    return pd.DataFrame(results)

def task1_rank_correlation(df, alpha_range=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.99]):
    """Task 1: 秩相关系数分析 (Kendall's τ 和 Spearman's ρ)"""
    print("\n" + "=" * 70)
    print("Task 1 灵敏度分析: 秩相关系数 (避免虚假鲁棒性)")
    print("=" * 70)
    
    results = []
    baseline_ranks = None
    
    for alpha in alpha_range:
        all_ranks = []
        
        for season in df['season'].unique():
            sdata = df[df['season'] == season].copy()
            if len(sdata) < 3:
                continue
            
            sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
            sdata['s_fan'] = (1.0 / sdata['rank']) ** alpha
            s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
            if s_max > s_min:
                sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)
            else:
                sdata['s_fan'] = 0.5
            
            w_j = 0.03
            sdata['score'] = w_j * sdata['s_judge'] + (1 - w_j) * sdata['s_fan']
            sdata['pred_rank'] = sdata['score'].rank(ascending=True)
            
            all_ranks.extend(list(zip(sdata['pred_rank'].values, sdata['week_out'].values)))
        
        all_ranks_df = pd.DataFrame(all_ranks, columns=['pred', 'actual'])
        
        # 计算秩相关系数
        tau, tau_p = stats.kendalltau(all_ranks_df['pred'], all_ranks_df['actual'])
        rho, rho_p = stats.spearmanr(all_ranks_df['pred'], all_ranks_df['actual'])
        
        # 与基准排名的相关性
        if baseline_ranks is None and alpha == 0.3:
            baseline_ranks = all_ranks_df['pred'].values
        
        if baseline_ranks is not None and len(baseline_ranks) == len(all_ranks_df):
            vs_base_tau, _ = stats.kendalltau(all_ranks_df['pred'], baseline_ranks)
        else:
            vs_base_tau = 1.0
        
        results.append({
            'alpha': alpha,
            'kendall_tau': tau,
            'spearman_rho': rho,
            'vs_baseline_tau': vs_base_tau
        })
        print(f"α = {alpha:.2f}: Kendall τ = {tau:.4f}, Spearman ρ = {rho:.4f}, vs baseline τ = {vs_base_tau:.4f}")
    
    return pd.DataFrame(results)

def task1_sensitivity_weight(df, weight_range=[0.01, 0.03, 0.05, 0.10, 0.15, 0.20]):
    """Task 1: 评委权重 wⱼ 灵敏度分析"""
    print("\n" + "=" * 70)
    print("Task 1 灵敏度分析: 评委权重 wⱼ")
    print("=" * 70)
    
    results = []
    alpha = 0.3
    
    for w_j in weight_range:
        correct = 0
        total = 0
        
        for season in df['season'].unique():
            sdata = df[df['season'] == season].copy()
            if len(sdata) < 3:
                continue
            
            sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
            sdata['s_fan'] = (1.0 / sdata['rank']) ** alpha
            s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
            if s_max > s_min:
                sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)
            else:
                sdata['s_fan'] = 0.5
            
            sdata['score'] = w_j * sdata['s_judge'] + (1 - w_j) * sdata['s_fan']
            
            predicted = sdata.loc[sdata['score'].idxmin(), 'celebrity_name']
            actual = sdata.loc[sdata['week_out'].idxmin(), 'celebrity_name']
            
            if predicted == actual:
                correct += 1
            total += 1
        
        accuracy = correct / total * 100 if total > 0 else 0
        results.append({'weight': w_j, 'accuracy': accuracy})
        print(f"wⱼ = {w_j:.2f}: 准确率 = {accuracy:.2f}%")
    
    return pd.DataFrame(results)



# =============================================================================
# Task 3 灵敏度分析
# =============================================================================
def task3_sensitivity_pvalue(df, p_thresholds=[0.01, 0.05, 0.10, 0.15]):
    """Task 3: 显著性阈值 p 灵敏度分析"""
    print("\n" + "=" * 70)
    print("Task 3 灵敏度分析: 显著性阈值 p")
    print("=" * 70)
    
    df_analysis = df.copy()
    df_analysis['age'] = pd.to_numeric(df_analysis['celebrity_age_during_season'], errors='coerce')
    df_analysis['s_judge'] = df_analysis['avg_normalized_score'].astype(float)
    
    results = []
    
    for p_threshold in p_thresholds:
        significant_factors = []
        
        # 年龄效应
        age_data = df_analysis[df_analysis['age'].notna()]
        if len(age_data) > 10:
            corr_age, p_age = stats.pearsonr(age_data['age'], age_data['week_out'])
            if p_age < p_threshold:
                significant_factors.append('Age')
        
        # 评委分效应
        judge_data = df_analysis[df_analysis['s_judge'].notna()]
        if len(judge_data) > 10:
            corr_judge, p_judge = stats.pearsonr(judge_data['s_judge'], judge_data['week_out'])
            if p_judge < p_threshold:
                significant_factors.append('Judge Score')
        
        results.append({
            'p_threshold': p_threshold,
            'n_significant': len(significant_factors),
            'significant_factors': ', '.join(significant_factors) if significant_factors else 'None'
        })
        print(f"p < {p_threshold}: {len(significant_factors)} 个显著因素 - {significant_factors}")
    
    return pd.DataFrame(results)

def task3_sensitivity_age_cutoff(df, age_cutoffs=[30, 35, 40, 45, 50]):
    """Task 3: 年龄分组边界灵敏度分析"""
    print("\n" + "=" * 70)
    print("Task 3 灵敏度分析: 年龄分组边界")
    print("=" * 70)
    
    df_analysis = df.copy()
    df_analysis['age'] = pd.to_numeric(df_analysis['celebrity_age_during_season'], errors='coerce')
    df_analysis = df_analysis[df_analysis['age'].notna()]
    
    results = []
    
    for cutoff in age_cutoffs:
        young_data = df_analysis[df_analysis['age'] < cutoff]['week_out']
        old_data = df_analysis[df_analysis['age'] >= cutoff]['week_out']
        
        if len(young_data) > 5 and len(old_data) > 5:
            young = young_data.mean()
            old = old_data.mean()
            diff = old - young
            t_stat, p_val = stats.ttest_ind(young_data.dropna(), old_data.dropna())
        else:
            young, old, diff, p_val = np.nan, np.nan, np.nan, np.nan
        
        results.append({
            'cutoff': cutoff,
            'young_mean': young,
            'old_mean': old,
            'diff': diff,
            'p_value': p_val
        })
        if not np.isnan(young):
            print(f"年龄边界 {cutoff}: 年轻={young:.2f}周, 年长={old:.2f}周, Δ={diff:.2f}, p={p_val:.4f}")
        else:
            print(f"年龄边界 {cutoff}: 数据不足")
    
    return pd.DataFrame(results)

def task3_bootstrap_analysis(df, n_bootstrap=500):
    """Task 3: Bootstrap 重采样分析 (检验系数稳定性)"""
    print("\n" + "=" * 70)
    print(f"Task 3 灵敏度分析: Bootstrap 重采样 ({n_bootstrap} 次)")
    print("=" * 70)
    
    df_analysis = df.copy()
    df_analysis['age'] = pd.to_numeric(df_analysis['celebrity_age_during_season'], errors='coerce')
    df_analysis['s_judge'] = df_analysis['avg_normalized_score'].astype(float)
    df_analysis = df_analysis[df_analysis['age'].notna() & df_analysis['s_judge'].notna()]
    
    age_coeffs = []
    
    for i in range(n_bootstrap):
        # 有放回抽样
        sample = df_analysis.sample(n=len(df_analysis), replace=True)
        
        # 简单线性回归: week_out ~ age
        if len(sample) > 10:
            slope, intercept, r, p, se = stats.linregress(sample['age'], sample['week_out'])
            age_coeffs.append(slope)
    
    age_coeffs = np.array(age_coeffs)
    
    results = {
        'mean': np.mean(age_coeffs),
        'std': np.std(age_coeffs),
        'ci_lower': np.percentile(age_coeffs, 2.5),
        'ci_upper': np.percentile(age_coeffs, 97.5),
        'skewness': stats.skew(age_coeffs),
        'kurtosis': stats.kurtosis(age_coeffs)
    }
    
    print(f"年龄系数 Bootstrap 分布:")
    print(f"  均值: {results['mean']:.4f}")
    print(f"  标准差: {results['std']:.4f}")
    print(f"  95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    print(f"  偏度: {results['skewness']:.4f}")
    print(f"  峰度: {results['kurtosis']:.4f}")
    
    # 判断分布形态
    if abs(results['skewness']) < 0.5 and abs(results['kurtosis']) < 1:
        print("  结论: 系数分布接近正态，因子归因具有统计稳定性 ✓")
    else:
        print("  结论: 系数分布偏离正态，需关注特定样本点影响")
    
    return pd.DataFrame([results])

def task3_leave_one_out(df):
    """Task 3: Leave-One-Out 特征重要性分析"""
    print("\n" + "=" * 70)
    print("Task 3 灵敏度分析: Leave-One-Out 特征重要性")
    print("=" * 70)
    
    df_analysis = df.copy()
    df_analysis['age'] = pd.to_numeric(df_analysis['celebrity_age_during_season'], errors='coerce')
    df_analysis['s_judge'] = df_analysis['avg_normalized_score'].astype(float)
    
    # 使用实际可用的列名构建特征
    # celebrity_industry 包含行业信息
    industry_col = df_analysis['celebrity_industry'].fillna('').str.lower()
    
    features = {
        'Age': df_analysis['age'],
        'Judge_Score': df_analysis['s_judge'],
        'Singer': industry_col.str.contains('music|singer|band', na=False).astype(int),
        'Actor': industry_col.str.contains('actor|actress|film|tv', na=False).astype(int),
        'Athlete': industry_col.str.contains('athlete|sport|nfl|nba|olympic', na=False).astype(int),
        'Reality_TV': industry_col.str.contains('reality|tv personality', na=False).astype(int)
    }
    
    feature_df = pd.DataFrame(features)
    feature_df['week_out'] = df_analysis['week_out']
    feature_df = feature_df.dropna()
    
    y = feature_df['week_out'].values
    
    def compute_r2(X, y):
        """使用 numpy 计算 OLS R²"""
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            y_pred = X_with_intercept @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot
        except:
            return 0
    
    # 全特征模型 R²
    X_full = feature_df.drop('week_out', axis=1).values
    r2_full = compute_r2(X_full, y)
    print(f"全特征模型 R² = {r2_full:.4f}")
    
    results = []
    feature_names = list(features.keys())
    
    for i, feature_name in enumerate(feature_names):
        # 移除该特征 (删除第i列)
        X_reduced = np.delete(X_full, i, axis=1)
        r2_reduced = compute_r2(X_reduced, y)
        
        delta_r2 = (r2_reduced - r2_full) * 100
        
        results.append({
            'feature': feature_name,
            'r2_without': r2_reduced,
            'delta_r2_pct': delta_r2,
            'importance_rank': 0
        })
        print(f"移除 {feature_name}: R² = {r2_reduced:.4f}, ΔR² = {delta_r2:+.2f}%")
    
    # 按重要性排序
    results_df = pd.DataFrame(results)
    results_df['importance_rank'] = results_df['delta_r2_pct'].abs().rank(ascending=False).astype(int)
    results_df = results_df.sort_values('importance_rank')
    
    print("\n特征重要性排序:")
    for _, row in results_df.iterrows():
        level = "关键" if abs(row['delta_r2_pct']) > 5 else "重要" if abs(row['delta_r2_pct']) > 2 else "次要" if abs(row['delta_r2_pct']) > 0.5 else "可忽略"
        print(f"  #{int(row['importance_rank'])} {row['feature']}: ΔR² = {row['delta_r2_pct']:+.2f}% ({level})")
    
    return results_df

def task3_interaction_effects(df):
    """Task 3: 特征交互效应分析 (Age × Feature)"""
    print("\n" + "=" * 70)
    print("Task 3 灵敏度分析: 特征交互效应 (Age × Feature)")
    print("=" * 70)
    
    df_analysis = df.copy()
    df_analysis['age'] = pd.to_numeric(df_analysis['celebrity_age_during_season'], errors='coerce')
    df_analysis['s_judge'] = df_analysis['avg_normalized_score'].astype(float)
    
    # 使用 celebrity_industry 构建特征
    industry_col = df_analysis['celebrity_industry'].fillna('').str.lower()
    
    feature_df = pd.DataFrame({
        'Age': df_analysis['age'],
        'Judge_Score': df_analysis['s_judge'],
        'Singer': industry_col.str.contains('music|singer|band', na=False).astype(int),
        'Actor': industry_col.str.contains('actor|actress|film|tv', na=False).astype(int),
        'Athlete': industry_col.str.contains('athlete|sport|nfl|nba|olympic', na=False).astype(int),
    })
    feature_df['week_out'] = df_analysis['week_out']
    feature_df = feature_df.dropna()
    
    y = feature_df['week_out'].values
    X_base = feature_df[['Age', 'Judge_Score', 'Singer', 'Actor', 'Athlete']].values
    
    def compute_r2_and_coef(X, y):
        """使用 numpy 计算 OLS R² 和系数"""
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            y_pred = X_with_intercept @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot
            return r2, beta[-1]  # 返回最后一个系数 (交互项)
        except:
            return 0, 0
    
    # 基准模型 (无交互项)
    r2_base, _ = compute_r2_and_coef(X_base, y)
    print(f"基准模型 (无交互项) R² = {r2_base:.4f}")
    
    results = []
    interaction_features = ['Singer', 'Actor', 'Athlete', 'Judge_Score']
    feature_indices = {'Singer': 2, 'Actor': 3, 'Athlete': 4, 'Judge_Score': 1}
    
    for interact_feat in interaction_features:
        # 添加 Age × Feature 交互项
        age_col = feature_df['Age'].values
        interact_col = feature_df[interact_feat].values
        interaction_term = age_col * interact_col
        X_interact = np.column_stack([X_base, interaction_term])
        
        r2_interact, interact_coef = compute_r2_and_coef(X_interact, y)
        delta_r2 = (r2_interact - r2_base) * 100
        
        results.append({
            'interaction': f'Age × {interact_feat}',
            'r2_with_interaction': r2_interact,
            'delta_r2_pct': delta_r2,
            'interaction_coef': interact_coef,
            'significant': abs(delta_r2) > 0.5
        })
        
        sig_mark = "**" if abs(delta_r2) > 0.5 else ""
        print(f"Age × {interact_feat}: R² = {r2_interact:.4f}, ΔR² = {delta_r2:+.2f}%, β = {interact_coef:.4f} {sig_mark}")
    
    results_df = pd.DataFrame(results)
    
    # 总结
    significant_interactions = results_df[results_df['significant']]
    if len(significant_interactions) > 0:
        print(f"\n发现 {len(significant_interactions)} 个显著交互效应:")
        for _, row in significant_interactions.iterrows():
            direction = "正向" if row['interaction_coef'] > 0 else "负向"
            print(f"  - {row['interaction']}: {direction}交互 (β={row['interaction_coef']:.4f})")
    else:
        print("\n未发现显著交互效应 (ΔR² < 0.5%)")
    
    return results_df

def task3_partner_interaction(df):
    """Task 3: 舞伴交互效应 (Age × Partner_Quality)"""
    print("\n" + "=" * 70)
    print("Task 3 灵敏度分析: 舞伴交互效应 (Age × Partner)")
    print("=" * 70)
    
    df_analysis = df.copy()
    df_analysis['age'] = pd.to_numeric(df_analysis['celebrity_age_during_season'], errors='coerce')
    
    partner_stats = df_analysis.groupby('ballroom_partner')['avg_normalized_score'].mean().to_dict()
    df_analysis['partner_quality'] = df_analysis['ballroom_partner'].map(partner_stats)
    
    df_analysis = df_analysis.dropna(subset=['age', 'partner_quality', 'week_out'])
    
    y = df_analysis['week_out'].values
    X = np.column_stack([df_analysis['age'].values, df_analysis['partner_quality'].values])
    interaction = df_analysis['age'].values * df_analysis['partner_quality'].values
    X_interact = np.column_stack([np.ones(len(X)), X, interaction])
    
    try:
        beta, _, _, _ = np.linalg.lstsq(X_interact, y, rcond=None)
        print("回归方程: Y = β0 + β1*Age + β2*Partner + β3*(Age*Partner)")
        print(f"  β1 (Age)     : {beta[1]:.4f}")
        print(f"  β2 (Partner) : {beta[2]:.4f}")
        print(f"  β3 (Interact): {beta[3]:.4f}")
        
        if abs(beta[3]) > 0.05:
            direction = "正向" if beta[3] > 0 else "负向"
            print(f"\n结论: 存在显著{direction}交互效应 (β3={beta[3]:.4f})。")
        else:
            print("\n结论: 交互效应不显著。")
        return pd.DataFrame({'term': ['Intercept', 'Age', 'Partner', 'Interaction'], 'coefficient': beta})
    except Exception as e:
        print(f"Regression failed: {e}")
        return pd.DataFrame()

def task4_theta_k_heatmap(df, theta_range=[0.10, 0.15, 0.20, 0.25, 0.30], k_range=[3, 5, 8, 10, 15, 20]):
    """Task 4: θ×k 双参数热力图 (参数耦合分析)"""
    print("\n" + "=" * 70)
    print("Task 4 灵敏度分析: θ×k 双参数热力图")
    print("=" * 70)
    
    results = []
    heatmap_matrix = np.zeros((len(k_range), len(theta_range)))
    
    for i, k in enumerate(k_range):
        for j, theta_q in enumerate(theta_range):
            skill_corrs = []
            penalty_triggered = 0
            total = 0
            
            for season in df['season'].unique():
                sdata = df[df['season'] == season].copy()
                if len(sdata) < 3:
                    continue
                
                sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
                sdata['s_fan'] = (1.0 / sdata['rank']) ** 0.3
                s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
                if s_max > s_min:
                    sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)
                
                threshold = sdata['s_judge'].quantile(theta_q)
                sdata['lambda'] = 1.0 / (1.0 + np.exp(-k * (sdata['s_judge'] - threshold)))
                
                alpha_t = 0.6  # 固定中期权重
                sdata['score'] = alpha_t * sdata['s_judge'] + (1 - alpha_t) * sdata['lambda'] * sdata['s_fan']
                
                corr, _ = stats.spearmanr(sdata['score'], sdata['s_judge'])
                skill_corrs.append(corr)
                
                penalty_triggered += (sdata['lambda'] < 0.5).sum()
                total += len(sdata)
            
            rho = np.mean(skill_corrs)
            penalty_rate = penalty_triggered / total * 100 if total > 0 else 0
            heatmap_matrix[i, j] = rho
            
            results.append({
                'theta_q': theta_q,
                'k': k,
                'rho': rho,
                'penalty_rate': penalty_rate
            })
            print(f"θ=Q{int(theta_q*100)}%, k={k}: ρ={rho:.4f}, 惩罚率={penalty_rate:.1f}%")
    
    print("\n热力图矩阵 (行=k, 列=θ):")
    print("       ", "  ".join([f"Q{int(t*100)}%" for t in theta_range]))
    for i, k in enumerate(k_range):
        row = "  ".join([f"{heatmap_matrix[i,j]:.3f}" for j in range(len(theta_range))])
        print(f"k={k:2d}:  {row}")
    
    return pd.DataFrame(results), heatmap_matrix

# =============================================================================
# Task 4 灵敏度分析
# =============================================================================
def compute_lambda(s_judge, threshold, k):
    """计算表现系数 λ"""
    return 1.0 / (1.0 + np.exp(-k * (s_judge - threshold)))

def compute_adaptive_threshold(s_judge_series, k_std=1.0):
    """计算自适应阈值 θ_t = μ_t - k * σ_t"""
    mu = s_judge_series.mean()
    sigma = s_judge_series.std()
    return mu - k_std * sigma

def compute_alpha_t(t, T=11, alpha_0=0.4, alpha_T=0.8):
    """计算时间动态权重 α(t) - 线性版"""
    return alpha_0 + (alpha_T - alpha_0) * (t / T)

# =============================================================================
# 鲁棒性增强函数
# =============================================================================
def compute_ensemble_lambda(s_judge, k, quantiles=[0.15, 0.20, 0.25]):
    """集成表现系数 λ̄ (多阈值平均)"""
    lambdas_sum = np.zeros(len(s_judge))
    for q in quantiles:
        threshold = s_judge.quantile(q)
        lambdas_sum += 1.0 / (1.0 + np.exp(-k * (s_judge - threshold)))
    return lambdas_sum / len(quantiles)

def adaptive_k(n_contestants):
    """自适应陡峭因子 k(n)"""
    if n_contestants >= 10:
        return 8
    elif n_contestants >= 5:
        return 10
    else:
        return 15

def compute_alpha_sigmoid(t, T=11, alpha_0=0.45, alpha_T=0.90):
    """Sigmoid 时间动态权重 α(t)"""
    x = (t - T/2) / (T/4)
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    return alpha_0 + (alpha_T - alpha_0) * sigmoid

def task4_robustness_comparison(df):
    """Task 4: 基础版 vs 鲁棒增强版对比"""
    print("\n" + "=" * 70)
    print("Task 4 鲁棒性对比: 基础版 vs 增强版")
    print("=" * 70)
    
    results = {'basic': [], 'enhanced': []}
    
    for version in ['basic', 'enhanced']:
        skill_corrs = []
        controversy = 0
        high_score_eliminated = 0
        total = 0
        
        for season in df['season'].unique():
            sdata = df[df['season'] == season].copy()
            if len(sdata) < 3:
                continue
            
            # 计算粉丝得分
            sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
            sdata['s_fan'] = (1.0 / sdata['rank']) ** 0.3
            s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
            if s_max > s_min:
                sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)
            
            if version == 'basic':
                # 基础版: 单一阈值, 固定k, 线性α(t)
                k = 10
                threshold = sdata['s_judge'].quantile(0.20)
                sdata['lambda'] = compute_lambda(sdata['s_judge'], threshold, k)
                alpha_t = compute_alpha_t(6)
            else:
                # 增强版: 集成λ, 自适应k, Sigmoid α(t)
                k = adaptive_k(len(sdata))
                sdata['lambda'] = compute_ensemble_lambda(sdata['s_judge'], k)
                alpha_t = compute_alpha_sigmoid(6)
            
            sdata['score'] = alpha_t * sdata['s_judge'] + (1 - alpha_t) * sdata['lambda'] * sdata['s_fan']
            
            # 统计指标
            elim_idx = sdata['score'].idxmin()
            elim_judge = sdata.loc[elim_idx, 's_judge']
            
            if elim_judge > sdata['s_judge'].median():
                controversy += 1
            if elim_judge > sdata['s_judge'].quantile(0.5):
                high_score_eliminated += 1
            
            total += 1
            corr, _ = stats.spearmanr(sdata['score'], sdata['s_judge'])
            skill_corrs.append(corr)
        
        results[version] = {
            'skill_corr': np.mean(skill_corrs),
            'controversy_rate': controversy / total * 100 if total > 0 else 0,
            'misfire_rate': high_score_eliminated / total * 100 if total > 0 else 0
        }
    
    # 输出对比
    print(f"\n{'指标':<20} {'基础版':<15} {'增强版':<15} {'改进':<10}")
    print("-" * 60)
    
    basic = results['basic']
    enhanced = results['enhanced']
    
    print(f"{'Skill Correlation':<20} {basic['skill_corr']:.4f}       {enhanced['skill_corr']:.4f}       +{(enhanced['skill_corr']-basic['skill_corr'])*100:.2f}%")
    print(f"{'争议率':<20} {basic['controversy_rate']:.1f}%          {enhanced['controversy_rate']:.1f}%          {enhanced['controversy_rate']-basic['controversy_rate']:.1f}%")
    print(f"{'误杀率':<20} {basic['misfire_rate']:.1f}%          {enhanced['misfire_rate']:.1f}%          {enhanced['misfire_rate']-basic['misfire_rate']:.1f}%")
    
    # α(t) 曲线对比
    print("\n时间动态权重曲线对比:")
    print(f"{'Week':<8} {'线性α(t)':<12} {'Sigmoid α(t)':<12}")
    for t in [1, 3, 6, 9, 11]:
        linear = compute_alpha_t(t)
        sigmoid = compute_alpha_sigmoid(t)
        print(f"{t:<8} {linear:.3f}        {sigmoid:.3f}")
    
    return pd.DataFrame([
        {'version': 'basic', **basic},
        {'version': 'enhanced', **enhanced}
    ])

def task4_sensitivity_threshold(df, q_range=[0.10, 0.15, 0.20, 0.25, 0.30]):
    """Task 4: 熔断阈值 θ 灵敏度分析"""
    print("\n" + "=" * 70)
    print("Task 4 灵敏度分析: 熔断阈值 θ (分位数)")
    print("=" * 70)
    
    results = []
    k = 10
    
    for q in q_range:
        controversy = 0
        high_score_eliminated = 0
        total = 0
        skill_corrs = []
        
        for season in df['season'].unique():
            sdata = df[df['season'] == season].copy()
            if len(sdata) < 3:
                continue
            
            sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
            sdata['s_fan'] = (1.0 / sdata['rank']) ** 0.3
            s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
            if s_max > s_min:
                sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)
            
            threshold = sdata['s_judge'].quantile(q)
            sdata['lambda'] = compute_lambda(sdata['s_judge'], threshold, k)
            
            alpha_t = compute_alpha_t(6)
            sdata['score_padw'] = alpha_t * sdata['s_judge'] + (1 - alpha_t) * sdata['lambda'] * sdata['s_fan']
            
            elim_idx = sdata['score_padw'].idxmin()
            elim_judge = sdata.loc[elim_idx, 's_judge']
            
            median_judge = sdata['s_judge'].median()
            if elim_judge > median_judge:
                controversy += 1
            
            top_50_threshold = sdata['s_judge'].quantile(0.5)
            if elim_judge > top_50_threshold:
                high_score_eliminated += 1
            
            total += 1
            
            corr, _ = stats.spearmanr(sdata['score_padw'], sdata['s_judge'])
            skill_corrs.append(corr)
        
        controversy_rate = controversy / total * 100 if total > 0 else 0
        misfire_rate = high_score_eliminated / total * 100 if total > 0 else 0
        avg_skill_corr = np.mean(skill_corrs)
        
        results.append({
            'threshold_q': q,
            'skill_corr': avg_skill_corr,
            'controversy_rate': controversy_rate,
            'misfire_rate': misfire_rate
        })
        print(f"θ = Q{int(q*100)}%: ρ={avg_skill_corr:.4f}, 争议率={controversy_rate:.1f}%, 误杀率={misfire_rate:.1f}%")
    
    return pd.DataFrame(results)

def task4_sensitivity_k(df, k_range=[3, 5, 10, 15, 20, 30]):
    """Task 4: 陡峭因子 k 灵敏度分析"""
    print("\n" + "=" * 70)
    print("Task 4 灵敏度分析: 陡峭因子 k")
    print("=" * 70)
    
    results = []
    q = 0.20
    
    for k in k_range:
        all_lambdas = []
        
        for season in df['season'].unique():
            sdata = df[df['season'] == season].copy()
            if len(sdata) < 3:
                continue
            
            threshold = sdata['s_judge'].quantile(q)
            lambdas = compute_lambda(sdata['s_judge'], threshold, k)
            all_lambdas.extend(lambdas.tolist())
        
        lambdas_arr = np.array(all_lambdas)
        
        results.append({
            'k': k,
            'lambda_mean': np.mean(lambdas_arr),
            'lambda_std': np.std(lambdas_arr),
            'pct_below_0.5': (lambdas_arr < 0.5).mean() * 100,
            'pct_above_0.9': (lambdas_arr > 0.9).mean() * 100
        })
        print(f"k = {k:2d}: λ均值={np.mean(lambdas_arr):.3f}, λ<0.5占比={((lambdas_arr < 0.5).mean()*100):.1f}%")
    
    return pd.DataFrame(results)

def task4_sensitivity_alpha_grid(df, alpha_0_range=[0.30, 0.40, 0.50], alpha_T_range=[0.70, 0.80, 0.90]):
    """Task 4: 时间动态权重 (α₀, αT) 网格搜索"""
    print("\n" + "=" * 70)
    print("Task 4 灵敏度分析: 时间动态权重网格搜索")
    print("=" * 70)
    
    results = []
    k = 10
    q = 0.20
    
    for alpha_0 in alpha_0_range:
        for alpha_T in alpha_T_range:
            skill_corrs = []
            
            for season in df['season'].unique():
                sdata = df[df['season'] == season].copy()
                if len(sdata) < 3:
                    continue
                
                sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
                sdata['s_fan'] = (1.0 / sdata['rank']) ** 0.3
                s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
                if s_max > s_min:
                    sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)
                
                threshold = sdata['s_judge'].quantile(q)
                sdata['lambda'] = compute_lambda(sdata['s_judge'], threshold, k)
                
                alpha_t = alpha_0 + (alpha_T - alpha_0) * (6 / 11)
                sdata['score_padw'] = alpha_t * sdata['s_judge'] + (1 - alpha_t) * sdata['lambda'] * sdata['s_fan']
                
                corr, _ = stats.spearmanr(sdata['score_padw'], sdata['s_judge'])
                skill_corrs.append(corr)
            
            avg_corr = np.mean(skill_corrs)
            results.append({
                'alpha_0': alpha_0,
                'alpha_T': alpha_T,
                'skill_corr': avg_corr
            })
            print(f"α₀={alpha_0:.2f}, αT={alpha_T:.2f}: ρ={avg_corr:.4f}")
    
    return pd.DataFrame(results)

# =============================================================================
# Task 1 灵敏度分析
# =============================================================================
def task1_ks_test(df):
    """Task 1: Kolmogorov-Smirnov Test for Score Distribution Robustness"""
    print("\n" + "=" * 70)
    print("Task 1 灵敏度分析: K-S 检验 (得分分布鲁棒性)")
    print("=" * 70)

    results = []
    for season in df['season'].unique():
        sdata = df[df['season'] == season].copy()
        if len(sdata) < 3:
            continue

        sdata['s_judge'] = sdata['avg_normalized_score'].astype(float)
        sdata['s_fan'] = (1.0 / sdata['week_out'].rank(ascending=False, method='min')) ** 0.3
        s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
        if s_max > s_min:
            sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)

        # Base model (alpha=0.6, no lambda)
        sdata['score_base'] = 0.6 * sdata['s_judge'] + 0.4 * sdata['s_fan']

        # Enhanced model (example parameters)
        k = adaptive_k(len(sdata))
        sdata['lambda'] = compute_ensemble_lambda(sdata['s_judge'], k)
        alpha_t = compute_alpha_sigmoid(6) # Mid-season alpha
        sdata['score_enhanced'] = alpha_t * sdata['s_judge'] + (1 - alpha_t) * sdata['lambda'] * sdata['s_fan']

        # K-S test between base and enhanced scores
        if len(sdata['score_base']) > 1 and len(sdata['score_enhanced']) > 1:
            ks_stat, p_value = stats.ks_2samp(sdata['score_base'], sdata['score_enhanced'])
            results.append({
                'season': season,
                'ks_stat': ks_stat,
                'p_value': p_value,
                'significant_diff': p_value < 0.05
            })
            print(f"Season {season}: K-S Stat={ks_stat:.4f}, P-value={p_value:.4f} (Significant: {p_value < 0.05})")
        else:
            print(f"Season {season}: Not enough data for K-S test.")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        print(f"\nOverall K-S Test Results: {results_df['significant_diff'].sum()} out of {len(results_df)} seasons show significant score distribution difference (p < 0.05).")
    return results_df

def task1_variance_dilution(df):
    """Task 1: Variance Dilution Analysis (得分方差稀释分析)"""
    print("\n" + "=" * 70)
    print("Task 1 灵敏度分析: 得分方差稀释分析")
    print("=" * 70)

    results = []
    for season in df['season'].unique():
        sdata = df[df['season'] == season].copy()
        if len(sdata) < 3:
            continue

        sdata['s_judge'] = sdata['avg_normalized_score'].astype(float)
        sdata['s_fan'] = (1.0 / sdata['week_out'].rank(ascending=False, method='min')) ** 0.3
        s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
        if s_max > s_min:
            sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)

        # Base model (alpha=0.6, no lambda)
        sdata['score_base'] = 0.6 * sdata['s_judge'] + 0.4 * sdata['s_fan']

        # Enhanced model (example parameters)
        k = adaptive_k(len(sdata))
        sdata['lambda'] = compute_ensemble_lambda(sdata['s_judge'], k)
        alpha_t = compute_alpha_sigmoid(6) # Mid-season alpha
        sdata['score_enhanced'] = alpha_t * sdata['s_judge'] + (1 - alpha_t) * sdata['lambda'] * sdata['s_fan']

        var_judge = sdata['s_judge'].var()
        var_fan = sdata['s_fan'].var()
        var_base = sdata['score_base'].var()
        var_enhanced = sdata['score_enhanced'].var()

        if var_judge > 0 and var_fan > 0:
            results.append({
                'season': season,
                'var_judge': var_judge,
                'var_fan': var_fan,
                'var_base_model': var_base,
                'var_enhanced_model': var_enhanced,
                'dilution_base_pct': (1 - var_base / var_judge) * 100 if var_judge > 0 else np.nan,
                'dilution_enhanced_pct': (1 - var_enhanced / var_judge) * 100 if var_judge > 0 else np.nan
            })
            print(f"Season {season}: Var(Judge)={var_judge:.4f}, Var(Base)={var_base:.4f} ({results[-1]['dilution_base_pct']:.1f}% dilution), Var(Enhanced)={var_enhanced:.4f} ({results[-1]['dilution_enhanced_pct']:.1f}% dilution)")
        else:
            print(f"Season {season}: Variance is zero, skipping dilution analysis.")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        print(f"\nAverage Variance Dilution (Base Model): {results_df['dilution_base_pct'].mean():.1f}%")
        print(f"Average Variance Dilution (Enhanced Model): {results_df['dilution_enhanced_pct'].mean():.1f}%")
    return results_df

# =============================================================================
# Task 2 灵敏度分析
# =============================================================================
def task2_stress_testing(df, noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20]):
    """Task 2: 压力测试 (Judge Score 噪声敏感性)"""
    print("\n" + "=" * 70)
    print("Task 2 灵敏度分析: 压力测试 (Judge Score 噪声敏感性)")
    print("=" * 70)

    results = []
    for noise_level in noise_levels:
        avg_corrs = []
        avg_controversy = []
        avg_misfire = []
        total_seasons = 0

        for season in df['season'].unique():
            sdata = df[df['season'] == season].copy()
            if len(sdata) < 3:
                continue

            sdata['s_judge_original'] = sdata['avg_normalized_score'].astype(float)
            
            # Add Gaussian noise to s_judge
            if noise_level > 0:
                noise = np.random.normal(0, noise_level * sdata['s_judge_original'].std(), len(sdata))
                sdata['s_judge'] = sdata['s_judge_original'] + noise
                # Clip scores to be within reasonable bounds (e.g., 0-1 for normalized scores)
                sdata['s_judge'] = sdata['s_judge'].clip(0, 1)
            else:
                sdata['s_judge'] = sdata['s_judge_original']

            sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
            sdata['s_fan'] = (1.0 / sdata['rank']) ** 0.3
            s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
            if s_max > s_min:
                sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)

            # Use enhanced model parameters for robustness
            k = adaptive_k(len(sdata))
            sdata['lambda'] = compute_ensemble_lambda(sdata['s_judge'], k)
            alpha_t = compute_alpha_sigmoid(6) # Mid-season alpha
            sdata['score'] = alpha_t * sdata['s_judge'] + (1 - alpha_t) * sdata['lambda'] * sdata['s_fan']

            # Calculate metrics
            corr, _ = stats.spearmanr(sdata['score'], sdata['s_judge_original']) # Compare with original judge score
            avg_corrs.append(corr)

            elim_idx = sdata['score'].idxmin()
            elim_judge = sdata.loc[elim_idx, 's_judge_original'] # Use original judge score for controversy
            
            controversy = 1 if elim_judge > sdata['s_judge_original'].median() else 0
            misfire = 1 if elim_judge > sdata['s_judge_original'].quantile(0.5) else 0

            avg_controversy.append(controversy)
            avg_misfire.append(misfire)
            total_seasons += 1
        
        if total_seasons > 0:
            results.append({
                'noise_level': noise_level,
                'avg_skill_corr': np.mean(avg_corrs),
                'avg_controversy_rate': np.mean(avg_controversy) * 100,
                'avg_misfire_rate': np.mean(avg_misfire) * 100
            })
            print(f"Noise Level={noise_level:.2f}: Avg ρ={results[-1]['avg_skill_corr']:.4f}, Avg Controversy={results[-1]['avg_controversy_rate']:.1f}%, Avg Misfire={results[-1]['avg_misfire_rate']:.1f}%")
        else:
            print(f"Noise Level={noise_level:.2f}: Not enough data for analysis.")

    return pd.DataFrame(results)

# =============================================================================
# Task 4 灵敏度分析
# =============================================================================
def task4_decision_consistency(df, k=10, q=0.20, alpha_t_val=0.6):
    """Task 4: 决策一致性分析 (基础模型 vs 增强模型)"""
    print("\n" + "=" * 70)
    print("Task 4 灵敏度分析: 决策一致性分析")
    print("=" * 70)

    results = []
    total_seasons = 0
    consistent_eliminations = 0
    
    for season in df['season'].unique():
        sdata = df[df['season'] == season].copy()
        if len(sdata) < 3:
            continue
        
        sdata['s_judge'] = sdata['avg_normalized_score'].astype(float)
        sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
        sdata['s_fan'] = (1.0 / sdata['rank']) ** 0.3
        s_min, s_max = sdata['s_fan'].min(), sdata['s_fan'].max()
        if s_max > s_min:
            sdata['s_fan'] = (sdata['s_fan'] - s_min) / (s_max - s_min)
        
        # --- 基础模型 ---
        # 固定参数 k, q, alpha_t
        threshold_base = sdata['s_judge'].quantile(q)
        sdata['lambda_base'] = compute_lambda(sdata['s_judge'], threshold_base, k)
        sdata['score_base'] = alpha_t_val * sdata['s_judge'] + (1 - alpha_t_val) * sdata['lambda_base'] * sdata['s_fan']
        
        elim_base = sdata.loc[sdata['score_base'].idxmin()]
        
        # --- 增强模型 ---
        k_enhanced = adaptive_k(len(sdata))
        sdata['lambda_enhanced'] = compute_ensemble_lambda(sdata['s_judge'], k_enhanced)
        alpha_t_enhanced = compute_alpha_sigmoid(6) # Mid-season alpha
        sdata['score_enhanced'] = alpha_t_enhanced * sdata['s_judge'] + (1 - alpha_t_enhanced) * sdata['lambda_enhanced'] * sdata['s_fan']
        
        elim_enhanced = sdata.loc[sdata['score_enhanced'].idxmin()]
        
        is_consistent = (elim_base['celebrity_name'] == elim_enhanced['celebrity_name'])
        if is_consistent:
            consistent_eliminations += 1
        total_seasons += 1

        results.append({
            'season': season,
            'eliminated_base': elim_base['celebrity_name'],
            'eliminated_enhanced': elim_enhanced['celebrity_name'],
            'consistency': is_consistent
        })
        print(f"Season {season}: Base Elim={elim_base['celebrity_name']}, Enhanced Elim={elim_enhanced['celebrity_name']} (Consistent: {is_consistent})")

    consistency_rate = (consistent_eliminations / total_seasons) * 100 if total_seasons > 0 else 0
    print(f"\nOverall Decision Consistency: {consistent_eliminations} out of {total_seasons} seasons ({consistency_rate:.2f}%) had consistent elimination decisions between base and enhanced models.")
    
    return pd.DataFrame(results)

# =============================================================================
# 可视化
# =============================================================================
def plot_sensitivity_results(task1_alpha, task1_weight, task4_threshold, task4_k, output_dir):
    """生成灵敏度分析可视化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Task 1: Alpha
    ax1 = axes[0, 0]
    ax1.plot(task1_alpha['alpha'], task1_alpha['accuracy'], 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=task1_alpha['accuracy'].max(), color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Pareto Parameter α', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Task 1: Pareto Parameter Sensitivity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([80, 90])
    
    # Task 1: Weight
    ax2 = axes[0, 1]
    ax2.plot(task1_weight['weight'], task1_weight['accuracy'], 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=task1_weight['accuracy'].max(), color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Judge Weight wⱼ', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Task 1: Judge Weight Sensitivity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Task 4: Threshold
    ax3 = axes[1, 0]
    ax3.plot(task4_threshold['threshold_q']*100, task4_threshold['skill_corr'], 'ro-', 
             linewidth=2, markersize=8, label='Skill Correlation')
    ax3.set_xlabel('Threshold Quantile (%)', fontsize=12)
    ax3.set_ylabel('Skill Correlation ρ', fontsize=12, color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(task4_threshold['threshold_q']*100, task4_threshold['controversy_rate'], 'b^--', 
                  linewidth=2, markersize=8, label='Controversy Rate')
    ax3_twin.set_ylabel('Controversy Rate (%)', fontsize=12, color='b')
    ax3_twin.tick_params(axis='y', labelcolor='b')
    
    ax3.set_title('Task 4: Threshold Sensitivity', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Task 4: k
    ax4 = axes[1, 1]
    ax4.plot(task4_k['k'], task4_k['pct_below_0.5'], 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Steepness Factor k', fontsize=12)
    ax4.set_ylabel('% with λ < 0.5 (Penalized)', fontsize=12)
    ax4.set_title('Task 4: Steepness Factor Sensitivity', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_analysis_plots.png'), dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {os.path.join(output_dir, 'sensitivity_analysis_plots.png')}")

def generate_tornado_chart(results_dict, output_dir):
    """生成龙卷风图"""
    
    tornado_data = []
    
    if 'task1_alpha' in results_dict:
        df = results_dict['task1_alpha']
        base = df[df['alpha'] == 0.3]['accuracy'].values[0]
        for _, row in df.iterrows():
            if row['alpha'] != 0.3:
                impact = row['accuracy'] - base
                tornado_data.append({
                    'parameter': f"Task1: α={row['alpha']}",
                    'impact': impact
                })
    
    if 'task4_threshold' in results_dict:
        df = results_dict['task4_threshold']
        base = df[df['threshold_q'] == 0.20]['skill_corr'].values[0]
        for _, row in df.iterrows():
            if row['threshold_q'] != 0.20:
                impact = (row['skill_corr'] - base) * 100
                tornado_data.append({
                    'parameter': f"Task4: θ=Q{int(row['threshold_q']*100)}%",
                    'impact': impact
                })
    
    if not tornado_data:
        return
    
    tornado_df = pd.DataFrame(tornado_data)
    tornado_df = tornado_df.sort_values('impact', key=abs, ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in tornado_df['impact']]
    ax.barh(tornado_df['parameter'], tornado_df['impact'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Impact on Model Performance', fontsize=12)
    ax.set_title('Tornado Chart: Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tornado_chart.png'), dpi=150, bbox_inches='tight')
    print(f"龙卷风图已保存: {os.path.join(output_dir, 'tornado_chart.png')}")

# =============================================================================
# 主函数
# =============================================================================
def main():
    print("=" * 70)
    print("综合灵敏度分析: Task 1, 3, 4 (增强版)")
    print("=" * 70)
    
    df = load_data()
    print(f"\n数据加载完成: {len(df)} 条记录, {df['season'].nunique()} 个赛季")
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Task 1 分析
    task1_alpha = task1_sensitivity_alpha(df)
    task1_rank = task1_rank_correlation(df)
    task1_weight = task1_sensitivity_weight(df)
    task1_ks_test_results = task1_ks_test(df)
    task1_variance_dilution_results = task1_variance_dilution(df)
    
    # Task 2 分析
    task2_stress_testing_results = task2_stress_testing(df)
    
    # Task 3 分析
    task3_pvalue = task3_sensitivity_pvalue(df)
    task3_age = task3_sensitivity_age_cutoff(df)
    task3_bootstrap = task3_bootstrap_analysis(df)
    task3_loo = task3_leave_one_out(df)
    task3_interact = task3_interaction_effects(df)
    task3_partner = task3_partner_interaction(df)
    
    # Task 4 分析
    task4_threshold = task4_sensitivity_threshold(df)
    task4_k = task4_sensitivity_k(df)
    task4_alpha_grid = task4_sensitivity_alpha_grid(df)
    task4_heatmap, heatmap_matrix = task4_theta_k_heatmap(df)
    task4_robustness = task4_robustness_comparison(df)
    task4_decision = task4_decision_consistency(df)
    
    # 保存CSV结果
    task1_alpha.to_csv(os.path.join(output_dir, 'task1_alpha_sensitivity.csv'), index=False)
    task1_rank.to_csv(os.path.join(output_dir, 'task1_rank_correlation.csv'), index=False)
    task1_weight.to_csv(os.path.join(output_dir, 'task1_weight_sensitivity.csv'), index=False)
    task1_ks_test_results.to_csv(os.path.join(output_dir, 'task1_ks_test.csv'), index=False)
    task1_variance_dilution_results.to_csv(os.path.join(output_dir, 'task1_variance_dilution.csv'), index=False)
    task2_stress_testing_results.to_csv(os.path.join(output_dir, 'task2_stress_testing.csv'), index=False)
    task3_pvalue.to_csv(os.path.join(output_dir, 'task3_pvalue_sensitivity.csv'), index=False)
    task3_age.to_csv(os.path.join(output_dir, 'task3_age_sensitivity.csv'), index=False)
    task3_bootstrap.to_csv(os.path.join(output_dir, 'task3_bootstrap_analysis.csv'), index=False)
    task3_loo.to_csv(os.path.join(output_dir, 'task3_leave_one_out.csv'), index=False)
    task3_interact.to_csv(os.path.join(output_dir, 'task3_interaction_effects.csv'), index=False)
    task3_partner.to_csv(os.path.join(output_dir, 'task3_partner_interaction.csv'), index=False)
    task4_threshold.to_csv(os.path.join(output_dir, 'task4_threshold_sensitivity.csv'), index=False)
    task4_k.to_csv(os.path.join(output_dir, 'task4_k_sensitivity.csv'), index=False)
    task4_alpha_grid.to_csv(os.path.join(output_dir, 'task4_alpha_grid_sensitivity.csv'), index=False)
    task4_heatmap.to_csv(os.path.join(output_dir, 'task4_theta_k_heatmap.csv'), index=False)
    task4_robustness.to_csv(os.path.join(output_dir, 'task4_robustness_comparison.csv'), index=False)
    task4_decision.to_csv(os.path.join(output_dir, 'task4_decision_consistency.csv'), index=False)
    
    print("\n" + "=" * 70)
    print("所有 CSV 结果已保存")
    print("=" * 70)
    
    # 可视化
    plot_sensitivity_results(task1_alpha, task1_weight, task4_threshold, task4_k, output_dir)
    
    results_dict = {
        'task1_alpha': task1_alpha,
        'task4_threshold': task4_threshold
    }
    generate_tornado_chart(results_dict, output_dir)
    
    print("\n灵敏度分析完成!")

if __name__ == "__main__":
    main()

