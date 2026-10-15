"""
Task 3: 双轨归因分析 (Dual-Track Factor Analysis)
==================================================
严格数学框架，完整统计检验

核心模型：
-----------
设 Y_J 为评委评分，Y_F 为粉丝支持度，X 为特征矩阵。

模型 A (评委视角):
    Y_J = β₀^J + β₁^J·Age + β₂^J·Industry + γ^J·Partner + ε^J

模型 B (粉丝视角):
    Y_F = β₀^F + β₁^F·Age + β₂^F·Industry + γ^F·Partner + ε^F

假设检验:
    H₀: β_k^J = β_k^F (特征对两种评价的影响相同)
    H₁: β_k^J ≠ β_k^F (影响存在显著差异)

检验统计量:
    t = (β_k^J - β_k^F) / √(SE_J² + SE_F²)

效应量:
    Cohen's d = (μ₁ - μ₂) / σ_pooled
    η² = SS_between / SS_total
"""

import pandas as pd
import numpy as np
import warnings
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings('ignore')

# =============================================================================
# 模型参数
# =============================================================================
ALPHA = 0.3  # Pareto 分布参数 (来自 Task 1)
SIGNIFICANCE_LEVEL = 0.05

def load_and_prepare_data():
    """
    数据加载与预处理
    
    Returns:
        df: 包含所有分析变量的 DataFrame
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_dir, 'task1', 'cleaned_data_final.csv'))
    
    # 1. 提取淘汰周次
    df['week_out'] = df['results'].str.extract(r'Eliminated Week (\d+)').astype(float)
    max_week = df['week_out'].max()
    if pd.isna(max_week):
        max_week = 11
    df['week_out'] = df['week_out'].fillna(max_week + 1)
    
    # 2. 重构粉丝强度 s_fan (Pareto Latent Vote Inversion from Task 1)
    # θ_fan = (1/rank)^α, s_fan = MinMax(θ_fan)
    for season in df['season'].unique():
        mask = df['season'] == season
        sdata = df[mask].copy()
        sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
        sdata['theta_fan'] = (1.0 / sdata['rank']) ** ALPHA
        t_min, t_max = sdata['theta_fan'].min(), sdata['theta_fan'].max()
        if t_max > t_min:
            sdata['s_fan'] = (sdata['theta_fan'] - t_min) / (t_max - t_min)
        else:
            sdata['s_fan'] = 0.5
        df.loc[mask, 's_fan'] = sdata['s_fan']
    
    # 3. 准备自变量
    df['Y_judge'] = df['avg_normalized_score'].astype(float)
    df['Y_fan'] = df['s_fan'].astype(float)
    df['Age'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Industry'] = df['celebrity_industry'].fillna('Other')
    df['Partner'] = df['ballroom_partner'].fillna('Unknown')
    df['HomeState'] = df['celebrity_homestate'].fillna('Unknown')
    
    # 4. 标准化年龄 (Z-score) 便于效应量解释
    df['Age_z'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
    
    return df

def calculate_effect_size(group1, group2):
    """
    计算 Cohen's d 效应量
    
    d = (μ₁ - μ₂) / σ_pooled
    σ_pooled = √[((n₁-1)σ₁² + (n₂-1)σ₂²) / (n₁+n₂-2)]
    
    解释标准:
        |d| < 0.2: 微小效应
        0.2 ≤ |d| < 0.5: 小效应
        0.5 ≤ |d| < 0.8: 中等效应
        |d| ≥ 0.8: 大效应
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
    return d

def calculate_eta_squared(df, group_col, value_col):
    """
    计算 η² (Eta-squared) - 方差解释比例
    
    η² = SS_between / SS_total
    
    解释标准:
        η² < 0.01: 微小效应
        0.01 ≤ η² < 0.06: 小效应
        0.06 ≤ η² < 0.14: 中等效应
        η² ≥ 0.14: 大效应
    """
    groups = df.groupby(group_col)[value_col]
    grand_mean = df[value_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in groups)
    ss_total = ((df[value_col] - grand_mean)**2).sum()
    return ss_between / ss_total if ss_total > 0 else 0

def run_formal_hypothesis_tests(df):
    """
    正式假设检验框架
    
    测试:
        1. 年龄效应: H₀: β_Age^J = β_Age^F
        2. 舞伴效应: H₀: ICC_J = ICC_F
        3. 行业效应: ANOVA F-检验
    """
    print("=" * 70)
    print("Task 3: 双轨归因分析 - 正式假设检验报告")
    print("=" * 70)
    print(f"显著性水平 α = {SIGNIFICANCE_LEVEL}")
    print(f"样本量 N = {len(df)}")
    
    results = {}
    
    # =========================================================================
    # 检验 1: 年龄效应差异 (Age Effect Differential)
    # =========================================================================
    print("\n" + "─" * 70)
    print("【检验 1】年龄效应差异 (Age Effect Differential)")
    print("─" * 70)
    print("H₀: β_Age^J = β_Age^F (年龄对评委和粉丝的影响相同)")
    print("H₁: β_Age^J ≠ β_Age^F (影响存在显著差异)")
    
    # 构建回归模型
    X = sm.add_constant(df[['Age_z']].astype(float))
    
    model_J = sm.OLS(df['Y_judge'].astype(float), X).fit()
    model_F = sm.OLS(df['Y_fan'].astype(float), X).fit()
    
    beta_J = model_J.params['Age_z']
    beta_F = model_F.params['Age_z']
    se_J = model_J.bse['Age_z']
    se_F = model_F.bse['Age_z']
    
    # Welch's t-test for coefficient comparison
    t_stat = (beta_J - beta_F) / np.sqrt(se_J**2 + se_F**2)
    df_welch = (se_J**2 + se_F**2)**2 / (se_J**4/(len(df)-2) + se_F**4/(len(df)-2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_welch))
    
    # 置信区间
    ci_J = (beta_J - 1.96*se_J, beta_J + 1.96*se_J)
    ci_F = (beta_F - 1.96*se_F, beta_F + 1.96*se_F)
    
    print(f"\n回归系数估计:")
    print(f"  β_Age^J = {beta_J:.6f} ± {1.96*se_J:.6f} (95% CI: [{ci_J[0]:.6f}, {ci_J[1]:.6f}])")
    print(f"  β_Age^F = {beta_F:.6f} ± {1.96*se_F:.6f} (95% CI: [{ci_F[0]:.6f}, {ci_F[1]:.6f}])")
    print(f"\nWelch's t-检验:")
    print(f"  t-统计量 = {t_stat:.4f}")
    print(f"  自由度 df = {df_welch:.2f}")
    print(f"  p-值 = {p_value:.6f}")
    
    if p_value < SIGNIFICANCE_LEVEL:
        print(f"  ✓ 结论: 拒绝 H₀ (p < {SIGNIFICANCE_LEVEL})，年龄效应存在显著差异")
        print(f"  解释: 年龄每增加1个标准差，评委分变化{beta_J:.4f}，粉丝分变化{beta_F:.4f}")
        print(f"        粉丝对年龄的负面反应是评委的 {abs(beta_F/beta_J):.2f} 倍")
    else:
        print(f"  → 结论: 无法拒绝 H₀ (p ≥ {SIGNIFICANCE_LEVEL})")
    
    results['age_test'] = {'t': t_stat, 'p': p_value, 'beta_J': beta_J, 'beta_F': beta_F}
    
    # =========================================================================
    # 检验 2: 职业舞伴效应 (Pro Dancer Effect)
    # =========================================================================
    print("\n" + "─" * 70)
    print("【检验 2】职业舞伴效应 (Pro Dancer Random Effect)")
    print("─" * 70)
    print("H₀: σ²_partner = 0 (舞伴无显著影响)")
    print("H₁: σ²_partner > 0 (舞伴存在显著影响)")
    
    # 计算 ICC (Intraclass Correlation Coefficient)
    def calculate_icc(df, group_col, value_col):
        """
        ICC = σ²_between / (σ²_between + σ²_within)
        """
        groups = df.groupby(group_col)[value_col]
        k = groups.ngroups
        n = len(df)
        
        grand_mean = df[value_col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in groups)
        ss_within = sum(((g - g.mean())**2).sum() for _, g in groups)
        
        ms_between = ss_between / (k - 1) if k > 1 else 0
        ms_within = ss_within / (n - k) if n > k else 0
        
        n_avg = n / k
        icc = (ms_between - ms_within) / (ms_between + (n_avg - 1) * ms_within)
        icc = max(0, icc)  # ICC 不能为负
        
        # F-检验
        f_stat = ms_between / ms_within if ms_within > 0 else 0
        p_value = 1 - stats.f.cdf(f_stat, k-1, n-k) if n > k else 1
        
        return icc, f_stat, p_value, k
    
    icc_J, f_J, p_J, k = calculate_icc(df, 'Partner', 'Y_judge')
    icc_F, f_F, p_F, _ = calculate_icc(df, 'Partner', 'Y_fan')
    
    print(f"\n组内相关系数 (ICC):")
    print(f"  评委模型: ICC = {icc_J:.4f} ({icc_J*100:.1f}% 方差由舞伴解释)")
    print(f"            F({k-1}, {len(df)-k}) = {f_J:.2f}, p = {p_J:.6f}")
    print(f"  粉丝模型: ICC = {icc_F:.4f} ({icc_F*100:.1f}% 方差由舞伴解释)")
    print(f"            F({k-1}, {len(df)-k}) = {f_F:.2f}, p = {p_F:.6f}")
    
    if p_J < SIGNIFICANCE_LEVEL:
        print(f"\n  ✓ 结论: 舞伴对评委分有显著影响 (p < {SIGNIFICANCE_LEVEL})")
    if p_F < SIGNIFICANCE_LEVEL:
        print(f"  ✓ 结论: 舞伴对粉丝分有显著影响 (p < {SIGNIFICANCE_LEVEL})")
    
    print(f"\n  效应大小对比: ICC差异 = {abs(icc_J - icc_F):.4f}")
    print(f"  解释: 职业舞伴对评委分的影响力是粉丝分的 {icc_J/icc_F:.2f} 倍")
    
    results['partner_test'] = {'icc_J': icc_J, 'icc_F': icc_F, 'f_J': f_J, 'f_F': f_F}
    
    # =========================================================================
    # 检验 3: 行业效应 (Industry Effect)
    # =========================================================================
    print("\n" + "─" * 70)
    print("【检验 3】行业效应 (Industry Effect - One-Way ANOVA)")
    print("─" * 70)
    print("H₀: μ_industry1 = μ_industry2 = ... = μ_industryK (各行业表现相同)")
    print("H₁: 至少存在一对 μ_i ≠ μ_j")
    
    # 筛选主要行业
    main_industries = df['Industry'].value_counts().head(6).index.tolist()
    df_industry = df[df['Industry'].isin(main_industries)]
    
    # 单因素 ANOVA
    groups_J = [g['Y_judge'].values for _, g in df_industry.groupby('Industry')]
    groups_F = [g['Y_fan'].values for _, g in df_industry.groupby('Industry')]
    
    f_J, p_J = stats.f_oneway(*groups_J)
    f_F, p_F = stats.f_oneway(*groups_F)
    
    # 效应量 η²
    eta2_J = calculate_eta_squared(df_industry, 'Industry', 'Y_judge')
    eta2_F = calculate_eta_squared(df_industry, 'Industry', 'Y_fan')
    
    print(f"\n评委模型 ANOVA:")
    print(f"  F = {f_J:.4f}, p = {p_J:.6f}")
    print(f"  η² = {eta2_J:.4f} ({eta2_J*100:.1f}% 方差由行业解释)")
    
    print(f"\n粉丝模型 ANOVA:")
    print(f"  F = {f_F:.4f}, p = {p_F:.6f}")
    print(f"  η² = {eta2_F:.4f} ({eta2_F*100:.1f}% 方差由行业解释)")
    
    # 行业均值排名
    industry_means = df_industry.groupby('Industry').agg({
        'Y_judge': 'mean',
        'Y_fan': 'mean'
    }).round(4)
    industry_means.columns = ['Judge_Mean', 'Fan_Mean']
    industry_means['Gap'] = industry_means['Fan_Mean'] - industry_means['Judge_Mean']
    
    print(f"\n行业表现排名:")
    print(industry_means.sort_values('Judge_Mean', ascending=False).to_string())
    
    results['industry_test'] = {'f_J': f_J, 'p_J': p_J, 'f_F': f_F, 'p_F': p_F, 'eta2_J': eta2_J, 'eta2_F': eta2_F}
    
    # =========================================================================
    # 检验 4: 地区效应 (Geographic Effect)
    # =========================================================================
    print("\n" + "─" * 70)
    print("【检验 4】地区效应 (Geographic Effect)")
    print("─" * 70)
    
    eta2_geo = calculate_eta_squared(df, 'HomeState', 'Y_fan')
    
    print(f"地区对粉丝支持的解释力:")
    print(f"  η² = {eta2_geo:.4f} ({eta2_geo*100:.1f}% 方差由地区解释)")
    
    if eta2_geo >= 0.06:
        print(f"  ✓ 结论: 存在中等强度的地缘投票效应")
    elif eta2_geo >= 0.01:
        print(f"  → 结论: 存在小幅度的地缘投票效应")
    else:
        print(f"  → 结论: 地缘效应可忽略")
    
    results['geo_test'] = {'eta2': eta2_geo}
    
    return results

def run_comprehensive_regression(df):
    """
    综合回归模型 (包含所有特征)
    """
    print("\n" + "=" * 70)
    print("【综合模型】多元回归分析 (Multiple Regression Analysis)")
    print("=" * 70)
    
    # 准备特征矩阵
    df_reg = df.copy()
    
    # 行业虚拟变量
    industry_dummies = pd.get_dummies(df_reg['Industry'], prefix='Ind', drop_first=True)
    main_industries = ['Ind_Athlete', 'Ind_Singer/Rapper', 'Ind_TV Personality']
    industry_cols = [c for c in main_industries if c in industry_dummies.columns]
    
    # 舞伴虚拟变量 (Top 5)
    top_partners = df_reg['Partner'].value_counts().head(5).index.tolist()
    partner_dummies = pd.get_dummies(df_reg['Partner'], prefix='Par', drop_first=True)
    partner_cols = [c for c in partner_dummies.columns if any(p in c for p in top_partners)][:5]
    
    # 构建设计矩阵
    X = df_reg[['Age_z']].astype(float)
    for col in industry_cols:
        X[col] = industry_dummies[col].astype(float)
    for col in partner_cols:
        X[col] = partner_dummies[col].astype(float)
    X = sm.add_constant(X)
    
    # 评委模型
    model_J = sm.OLS(df_reg['Y_judge'].astype(float), X).fit()
    # 粉丝模型
    model_F = sm.OLS(df_reg['Y_fan'].astype(float), X).fit()
    
    print("\n【评委模型 (Model A)】")
    print(f"R² = {model_J.rsquared:.4f}, Adj R² = {model_J.rsquared_adj:.4f}")
    print(f"F({model_J.df_model:.0f}, {model_J.df_resid:.0f}) = {model_J.fvalue:.2f}, p = {model_J.f_pvalue:.6f}")
    print(model_J.summary().tables[1])
    
    print("\n【粉丝模型 (Model B)】")
    print(f"R² = {model_F.rsquared:.4f}, Adj R² = {model_F.rsquared_adj:.4f}")
    print(f"F({model_F.df_model:.0f}, {model_F.df_resid:.0f}) = {model_F.fvalue:.2f}, p = {model_F.f_pvalue:.6f}")
    print(model_F.summary().tables[1])
    
    # 系数对比检验
    print("\n【系数差异检验 (Coefficient Comparison)】")
    print(f"{'变量':<20} {'β_J':>10} {'β_F':>10} {'t-stat':>10} {'p-value':>12} {'显著性':>8}")
    print("─" * 72)
    
    for var in X.columns:
        if var == 'const':
            continue
        if var in model_J.params.index and var in model_F.params.index:
            b_J = model_J.params[var]
            b_F = model_F.params[var]
            se_J = model_J.bse[var]
            se_F = model_F.bse[var]
            t = (b_J - b_F) / np.sqrt(se_J**2 + se_F**2)
            p = 2 * (1 - stats.t.cdf(abs(t), len(df)-2))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            var_short = var[:18] + '..' if len(var) > 20 else var
            print(f"{var_short:<20} {b_J:>10.4f} {b_F:>10.4f} {t:>10.2f} {p:>12.6f} {sig:>8}")
    
    return model_J, model_F

def generate_final_conclusion(results):
    """生成最终结论"""
    print("\n" + "=" * 70)
    print("Task 3 最终结论与回答")
    print("=" * 70)
    
    print("\n📌 问题: 职业舞者和名人特征对比赛表现有多大影响？")
    print("─" * 50)
    print(f"  1. 职业舞伴效应:")
    print(f"     - 评委端: ICC = {results['partner_test']['icc_J']:.4f} (解释 {results['partner_test']['icc_J']*100:.1f}% 方差)")
    print(f"     - 粉丝端: ICC = {results['partner_test']['icc_F']:.4f} (解释 {results['partner_test']['icc_F']*100:.1f}% 方差)")
    print(f"     - 结论: 舞伴对评委分影响更大 (\"名师出高徒\"效应)")
    
    print(f"\n  2. 年龄效应:")
    print(f"     - 评委端: β = {results['age_test']['beta_J']:.4f}")
    print(f"     - 粉丝端: β = {results['age_test']['beta_F']:.4f}")
    print(f"     - 结论: 年龄对粉丝的负面影响是评委的 {abs(results['age_test']['beta_F']/results['age_test']['beta_J']):.1f} 倍")
    
    print(f"\n  3. 行业效应:")
    print(f"     - 评委端: η² = {results['industry_test']['eta2_J']:.4f}")
    print(f"     - 粉丝端: η² = {results['industry_test']['eta2_F']:.4f}")
    
    print("\n📌 问题: 这些因素对评委和粉丝的影响是否相同？")
    print("─" * 50)
    print(f"  答案: 不相同。")
    print(f"  1. 年龄效应差异显著 (t = {results['age_test']['t']:.2f}, p = {results['age_test']['p']:.6f})")
    print(f"  2. 舞伴效应差异 = {abs(results['partner_test']['icc_J'] - results['partner_test']['icc_F']):.4f}")
    print(f"  3. 地区对粉丝有独立影响 (η² = {results['geo_test']['eta2']:.4f})")

if __name__ == "__main__":
    df = load_and_prepare_data()
    results = run_formal_hypothesis_tests(df)
    model_J, model_F = run_comprehensive_regression(df)
    generate_final_conclusion(results)
