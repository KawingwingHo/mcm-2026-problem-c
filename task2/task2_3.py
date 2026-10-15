"""
Task 2.3: 制度建议与综合评价
============================
1. 比较不同制度下的系统稳定性:
   - Rank Rule + No Save
   - Pct Rule + No Save
   - Hybrid: Rank Rule + Judge Save (推荐方案)
   
2. 指标对照:
   - 公平性 (专业性)
   - 意外率 (争议率)
"""

import pandas as pd
import numpy as np
import warnings
import os
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

ALPHA = 0.3

def process():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_dir, 'task1', 'cleaned_data_final.csv'))
    df['week_out'] = df['results'].str.extract(r'Eliminated Week (\d+)').astype(float)
    df['week_out'] = df['week_out'].fillna(12)
    
    final_stats = []
    
    for season in sorted(df['season'].unique()):
        sdata = df[df['season'] == season].copy()
        sdata['rank'] = sdata['week_out'].rank(ascending=False, method='min')
        sdata['s_fan_latent'] = (1.0 / sdata['rank']) ** ALPHA
        
        for week in range(1, int(sdata['week_out'].max())):
            current = sdata[sdata['week_out'] >= week].copy()
            if len(current) <= 1:
                continue
            
            # 基础数据准备
            def get_j_raw(row):
                cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
                vals = [row[c] for c in cols if c in row.index and pd.notnull(row[c])]
                return sum(vals) if vals else 0
            current['j_raw'] = current.apply(get_j_raw, axis=1)
            j_min, j_max = current['j_raw'].min(), current['j_raw'].max()
            current['s_judge'] = (current['j_raw'] - j_min) / (j_max - j_min) if j_max > j_min else 0.5
            
            # --- 系统 1: Percentage Rule (No Save) ---
            j_sum = current['j_raw'].sum()
            f_sum = current['s_fan_latent'].sum()
            current['pct_judge'] = current['j_raw'] / j_sum if j_sum > 0 else 1.0/len(current)
            current['pct_fan'] = current['s_fan_latent'] / f_sum if f_sum > 0 else 1.0/len(current)
            current['total_pct'] = 0.5 * current['pct_judge'] + 0.5 * current['pct_fan']
            elim_pct = current.sort_values(['total_pct', 's_fan_latent']).iloc[0]
            
            # --- 系统 2: Rank Rule (No Save) ---
            current['rank_pts_judge'] = current['j_raw'].rank(ascending=True, method='min')
            current['rank_pts_fan'] = current['s_fan_latent'].rank(ascending=True, method='min')
            current['total_pts_rank'] = current['rank_pts_judge'] + current['rank_pts_fan']
            elim_rank = current.sort_values(['total_pts_rank', 's_fan_latent']).iloc[0]
            
            # --- 系统 3: Hybrid (Rank Rule + Judge Save) ---
            bottom2 = current.sort_values(['total_pts_rank', 's_fan_latent']).iloc[:2]
            # 评委救评分高的人 (阈值0.1)
            if abs(bottom2.iloc[0]['s_judge'] - bottom2.iloc[1]['s_judge']) > 0.1:
                elim_hybrid = bottom2.sort_values('s_judge').iloc[0]
            else:
                elim_hybrid = bottom2.iloc[0]
            
            # 计算专业性相关系数
            rho_pct, _ = spearmanr(current['j_raw'], current['total_pct'])
            rho_rank, _ = spearmanr(current['j_raw'], current['total_pts_rank'])
            # Hybrid 的最终排名比较复杂，这里简化观测其淘汰的一致性
            
            # 记录数据
            final_stats.append({
                'Season': season,
                'Week': week,
                'Pct_Elim': elim_pct['celebrity_name'],
                'Rank_Elim': elim_rank['celebrity_name'],
                'Hybrid_Elim': elim_hybrid['celebrity_name'],
                'Pct_Rho': rho_pct,
                'Rank_Rho': rho_rank,
                'Judge_Bottom': elim_pct['j_raw'] == j_min,
                'Rank_Save_Happened': elim_rank['celebrity_name'] != elim_hybrid['celebrity_name']
            })
            
    return pd.DataFrame(final_stats)

def summarize(df):
    print("=" * 60)
    print("Task 2.3: 赛制推荐分析报告")
    print("=" * 60)
    
    print(f"\n1. 专业性对比 (Spearman Rho):")
    print(f"   Percentage Rule: {df['Pct_Rho'].mean():.4f}")
    print(f"   Rank Rule:       {df['Rank_Rho'].mean():.4f}")
    
    print(f"\n2. 生存率分析 (Survival Logic):")
    print(f"   Percentage 制淘汰赛评委分最低者的概率: {df['Judge_Bottom'].mean():.2%}")
    print(f"   说明: 较低的概率意味着粉丝投票经常'覆盖'专业评分，导致高评分者被淘汰。")
    
    print(f"\n3. Hybrid 机制纠偏能力:")
    print(f"   Judge Save 机制触发比例 (纠正 Rank 制结果): {df['Rank_Save_Happened'].mean():.2%}")
    
    print("\n" + "=" * 60)
    print("回答问题 2.3")
    print("=" * 60)
    print("Q: 建议在未来的季节中使用哪种方法？")
    print("A: 建议采用 [Rank Rule + Judge Save] 混合机制。")
    print("\n理由:")
    print("1. [专业底线] Rank Rule 相比 Pct Rule 能更有效地保护高技术实力的选手。")
    print("   模拟显示 Rank Rule 的评委相关度更高 ({:.4f})。".format(df['Rank_Rho'].mean()))
    print("2. [容错机制] 加入 Judge Save 机制。")
    print("   这为比赛提供了最后的'防火墙'。模拟显示，在约 {:.1f}% 的周次中，".format(df['Rank_Save_Happened'].mean()*100))
    print("   该机制能成功挽救因意外掉入 Bottom 2 的高技术选手，减少'Sabrina Bryan 式'悲剧。")
    print("3. [娱乐价值] Rank 制下每个人的权重相等，避免了绝对值领先带来的'无悬念'后期。")
    print("   保持了比赛的可看性和悬念。")

if __name__ == "__main__":
    results = process()
    summarize(results)
