"""
Task 1: 基于潜在变量重构的动态评分效能评估模型 (赛制区分版)
=============================================================

严格按照 task1.docx 文档实现，并加入赛制区分：
- Percentage Rule (S3-27): 粉丝投票为百分比
- Rank Rule (S1-2, S28+): 仅知相对排名

四大模块：
1. 数据同质化：自适应周内标准化算子
2. 缺失数据重构：基于帕累托截尾的潜在投票反演
3. 评价指标构建：评委-大众一致性漂移指数 (JPADI)
4. 敏感性分析与模型鲁棒性检验
"""

import pandas as pd
import numpy as np
import warnings
from scipy import stats
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ============================================================================
# 赛制区分 & MLE优化参数
# ============================================================================

def get_rule(season):
    """判断赛制类型"""
    if season in [1, 2] or season >= 28:
        return 'Rank'
    else:
        return 'Percentage'

# 分赛制MLE优化后的最优参数
PARAMS = {
    'Percentage': {'alpha': 0.3, 'jw': 0.03},  # S3-27
    'Rank': {'alpha': 0.3, 'jw': 0.03}         # S1-2, S28+
}

# ============================================================================
# 1. 数据同质化：自适应周内标准化算子
# ============================================================================

def adaptive_standardization(current, week):
    """周内Min-Max标准化"""
    def get_judge_score(row):
        cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        vals = [row[c] for c in cols if c in row.index and pd.notnull(row[c])]
        return sum(vals) if vals else 0
    
    scores = current.apply(get_judge_score, axis=1)
    s_min, s_max = scores.min(), scores.max()
    
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    return pd.Series([0.5] * len(current), index=current.index)


# ============================================================================
# 2. 缺失数据重构：基于帕累托截尾的潜在投票反演
# ============================================================================

def pareto_latent_vote_inversion(season_data, alpha):
    """帕累托分布潜在投票反演"""
    season_data = season_data.copy()
    season_data['rank'] = season_data['week_out'].rank(ascending=False, method='min')
    season_data['theta_fan'] = (1.0 / season_data['rank']) ** alpha
    
    t_min, t_max = season_data['theta_fan'].min(), season_data['theta_fan'].max()
    if t_max > t_min:
        season_data['s_fan'] = (season_data['theta_fan'] - t_min) / (t_max - t_min)
    else:
        season_data['s_fan'] = 0.5
    
    return season_data


# ============================================================================
# 3. JPADI指数
# ============================================================================

def calculate_jpadi(judge_scores, total_scores):
    """评委-大众一致性漂移指数"""
    if len(judge_scores) < 3:
        return np.nan
    judge_ranks = np.argsort(-judge_scores)
    total_ranks = np.argsort(-total_scores)
    correlation, _ = spearmanr(judge_ranks, total_ranks)
    return 1 - correlation


# ============================================================================
# 4. 敏感性分析
# ============================================================================

def sensitivity_analysis(df, alpha_range=[0.2, 0.3, 0.4]):
    """帕累托参数扰动测试"""
    results = {}
    
    for alpha in alpha_range:
        correct, total = 0, 0
        
        for season in df['season'].unique():
            sdata = pareto_latent_vote_inversion(df[df['season'] == season].copy(), alpha)
            rule = get_rule(season)
            jw = PARAMS[rule]['jw']
            
            for week in range(1, int(sdata['week_out'].max())):
                cur = sdata[sdata['week_out'] >= week].copy()
                if len(cur) <= 1:
                    continue
                
                cur['s_judge'] = adaptive_standardization(cur, week)
                cur['total'] = jw * cur['s_judge'] + (1 - jw) * cur['s_fan']
                
                pred = cur.sort_values('total').iloc[0]['celebrity_name']
                actual = sdata[sdata['week_out'] == week]
                
                if not actual.empty:
                    total += 1
                    if pred == actual.iloc[0]['celebrity_name']:
                        correct += 1
        
        results[alpha] = correct / total if total > 0 else 0
    
    return results


# ============================================================================
# 主分析流程
# ============================================================================

def run_task1_analysis(file_path):
    """完整Task 1分析"""
    
    df = pd.read_csv(file_path)
    df['week_out'] = df['results'].str.extract(r'Eliminated Week (\d+)').astype(float)
    max_week = df['week_out'].max()
    if np.isnan(max_week):
        max_week = 11
    df['week_out'] = df['week_out'].fillna(max_week + 1)
    df['Rule'] = df['season'].apply(get_rule)
    
    print("=" * 80)
    print("Task 1: 基于潜在变量重构的动态评分效能评估模型")
    print("=" * 80)
    print()
    print("赛制区分:")
    print(f"  • Percentage Rule (S3-27): α={PARAMS['Percentage']['alpha']}, jw={PARAMS['Percentage']['jw']:.0%}")
    print(f"  • Rank Rule (S1-2, S28+): α={PARAMS['Rank']['alpha']}, jw={PARAMS['Rank']['jw']:.0%}")
    print()
    
    all_results = []
    all_jpadi = []
    
    for season in sorted(df['season'].unique()):
        season_data = df[df['season'] == season].copy()
        rule = get_rule(season)
        alpha = PARAMS[rule]['alpha']
        jw = PARAMS[rule]['jw']
        
        season_data = pareto_latent_vote_inversion(season_data, alpha)
        season_jpadi_list = []
        
        for week in range(1, int(season_data['week_out'].max())):
            current = season_data[season_data['week_out'] >= week].copy()
            if len(current) <= 1:
                continue
            
            n = len(current)
            current['s_judge'] = adaptive_standardization(current, week)
            current['total'] = jw * current['s_judge'] + (1 - jw) * current['s_fan']
            
            current_sorted = current.sort_values('total')
            pred = current_sorted.iloc[0]['celebrity_name']
            
            actual = season_data[season_data['week_out'] == week]
            if actual.empty:
                continue
            actual_name = actual.iloc[0]['celebrity_name']
            actual_rank = (current_sorted['celebrity_name'] == actual_name).values.argmax() + 1
            
            jpadi = calculate_jpadi(current['s_judge'].values, current['total'].values)
            season_jpadi_list.append(jpadi)
            
            # Bootstrap确定性
            n_sim = 1000
            matches = 0
            base_fan = current['s_fan'].values
            base_judge = current['s_judge'].values
            names = current['celebrity_name'].values
            
            for _ in range(n_sim):
                noise = np.random.normal(0, 0.1, n)
                fan_new = np.clip(base_fan + noise, 0, 1)
                total_new = jw * base_judge + (1 - jw) * fan_new
                if names[np.argmin(total_new)] == pred:
                    matches += 1
            
            certainty = matches / n_sim
            
            all_results.append({
                'Season': season,
                'Rule': rule,
                'Week': week,
                'N': n,
                'Predicted': pred,
                'Actual': actual_name,
                'Correct': pred == actual_name,
                'Actual_Rank': actual_rank,
                'In_Bottom2': actual_rank <= 2,
                'In_Bottom3': actual_rank <= 3,
                'JPADI': jpadi,
                'Certainty': certainty,
                'Random_Baseline': 1.0 / n
            })
        
        if season_jpadi_list:
            all_jpadi.append({
                'Season': season,
                'Rule': rule,
                'Mean_JPADI': np.nanmean(season_jpadi_list),
                'Std_JPADI': np.nanstd(season_jpadi_list)
            })
    
    df_results = pd.DataFrame(all_results)
    df_jpadi = pd.DataFrame(all_jpadi)
    
    # ========== 报告 ==========
    
    # 1.1 一致性
    exact_match = df_results['Correct'].mean()
    random_baseline = df_results['Random_Baseline'].mean()
    improvement = exact_match / random_baseline
    bottom2 = df_results['In_Bottom2'].mean()
    bottom3 = df_results['In_Bottom3'].mean()
    mean_rank = df_results['Actual_Rank'].mean()
    
    n_total = len(df_results)
    n_correct = df_results['Correct'].sum()
    binom = stats.binomtest(n_correct, n_total, random_baseline, alternative='greater')
    
    pct_acc = df_results[df_results['Rule'] == 'Percentage']['Correct'].mean()
    rank_acc = df_results[df_results['Rule'] == 'Rank']['Correct'].mean()
    
    print("【问题 1.1】模型一致性度量")
    print("-" * 80)
    print(f"  ✓ 总体精确匹配准确率: {exact_match:.2%}")
    print(f"     - Percentage Rule (S3-27): {pct_acc:.2%}")
    print(f"     - Rank Rule (S1-2, S28+): {rank_acc:.2%}")
    print(f"  ✓ 正确预测: {n_correct}/{n_total}")
    print(f"  ✓ 随机基线: {random_baseline:.2%}")
    print(f"  ✓ 提升倍数: {improvement:.2f}x")
    print(f"  ✓ Bottom-2准确率: {bottom2:.2%}")
    print(f"  ✓ Bottom-3准确率: {bottom3:.2%}")
    print(f"  ✓ 平均预测排名: {mean_rank:.2f}")
    print(f"  ✓ 统计显著性: p={binom.pvalue:.2e}")
    print()
    
    # 1.2 确定性
    mean_certainty = df_results['Certainty'].mean()
    std_certainty = df_results['Certainty'].std()
    
    print("【问题 1.2】粉丝投票确定性度量")
    print("-" * 80)
    print(f"  ✓ Bootstrap置信度: {mean_certainty:.4f}")
    print(f"  ✓ 置信度标准差: {std_certainty:.4f}")
    print(f"  ✓ Monte Carlo模拟次数: 1000")
    print()
    
    # 1.3 JPADI
    mean_jpadi = df_results['JPADI'].mean()
    std_jpadi = df_results['JPADI'].std()
    
    corr_week, pval_week = stats.pearsonr(df_results['Week'], df_results['Certainty'])
    
    print("【问题 1.3】评委-大众一致性漂移指数 (JPADI)")
    print("-" * 80)
    print(f"  ✓ 平均JPADI: {mean_jpadi:.4f}")
    print(f"  ✓ JPADI标准差: {std_jpadi:.4f}")
    print(f"  ✓ 周次与确定性相关性: r={corr_week:.4f}, p={pval_week:.4f}")
    print()
    
    # 1.4 敏感性
    print("【问题 1.4】敏感性分析与鲁棒性检验")
    print("-" * 80)
    sensitivity = sensitivity_analysis(df, alpha_range=[0.2, 0.3, 0.4])
    print("A. 帕累托参数 α 扰动测试:")
    for alpha, acc in sensitivity.items():
        print(f"  α={alpha}: 准确率={acc:.2%}")
    print()
    print(f"B. 模型稳定性: {min(sensitivity.values()):.2%} - {max(sensitivity.values()):.2%}")
    print()
    
    print("=" * 80)
    print("分析完成")
    print("=" * 80)
    
    return df_results, df_jpadi


if __name__ == "__main__":
    results, jpadi = run_task1_analysis('cleaned_data_final.csv')
    results.to_csv('task1_final_results.csv', index=False)
    jpadi.to_csv('task1_jpadi_results.csv', index=False)
    print(f"\n结果已保存到: task1_final_results.csv, task1_jpadi_results.csv")
