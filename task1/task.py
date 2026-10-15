import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def solve_task1_corrected(file_path):
    """
    严格基于final.docx修正版：
    1. 模拟次数：1000次 (Strictly 1000 iterations for Bootstrap)
    2. 粉丝分布：帕累托 (Assumption 2)
    """
    # 1. 数据预处理
    df = pd.read_csv(file_path)
    df['week_out'] = df['results'].str.extract(r'Eliminated Week (\d+)').astype(float)
    max_week = df['week_out'].max()
    if np.isnan(max_week): max_week = 11 
    df['week_out'] = df['week_out'].fillna(max_week + 1)
    
    results_log = []

    # 2. 逐赛季反事实推演
    for season in df['season'].unique():
        season_data = df[df['season'] == season].copy()
        
        # 隐变量估计：帕累托分布 (Assumption 2)
        season_data['rank'] = season_data['week_out'].rank(ascending=False, method='min')
        season_data['theta_fan'] = (1.0 / season_data['rank']) ** 2.0  # 帕累托参数 alpha=2.0
        
        # 归一化粉丝分
        f_min, f_max = season_data['theta_fan'].min(), season_data['theta_fan'].max()
        season_data['s_fan'] = (season_data['theta_fan'] - f_min) / (f_max - f_min)
        
        for week in range(1, int(season_data['week_out'].max())):
            current = season_data[season_data['week_out'] >= week].copy()
            if len(current) <= 1: continue
            
            # 评委分标准化 (Assumption 1)
            def get_score(row):
                cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
                vals = [row[c] for c in cols if pd.notnull(row[c])]
                return sum(vals) if vals else 0
            
            current['judge_raw'] = current.apply(get_score, axis=1)
            j_min, j_max = current['judge_raw'].min(), current['judge_raw'].max()
            current['s_judge'] = 0.5 if j_max == j_min else (current['judge_raw'] - j_min) / (j_max - j_min)
            
            # 综合得分
            current['total'] = current['s_judge'] + current['s_fan']
            pred = current.sort_values('total').iloc[0]['celebrity_name']
            
            # 真实结果
            actual = season_data[season_data['week_out'] == week]
            actual_name = actual.iloc[0]['celebrity_name'] if not actual.empty else None
            
            if actual_name:
                # 1.2 确定性估算：1000次蒙特卡洛模拟 (Corrected to 1000)
                n_sim = 1000
                matches = 0
                base_fan = current['s_fan'].values
                base_judge = current['s_judge'].values
                
                for _ in range(n_sim):
                    # 施加噪声模拟不确定性
                    noise = np.random.normal(0, 0.1, len(base_fan))
                    fan_new = np.clip(base_fan + noise, 0, 1)
                    total_new = base_judge + fan_new
                    new_loser = current.iloc[np.argmin(total_new)]['celebrity_name']
                    if new_loser == pred:
                        matches += 1
                
                certainty = matches / n_sim
                results_log.append({
                    'Season': season, 'Week': week, 
                    'Correct': (pred == actual_name), 'Certainty': certainty
                })

    return pd.DataFrame(results_log)

# 运行并输出
df_res = solve_task1_corrected('cleaned_data_final.csv')

print("【Task 1.1 一致性度量】")
print(f"模型一致性 (Consistency): {df_res['Correct'].mean():.2%}")

print("\n【Task 1.2 粉丝投票确定性 (1000次模拟)】")
print(f"平均确定性 (Mean Certainty): {df_res['Certainty'].mean():.4f}")

print("\n【Task 1.3 确定性时序变化】")
print(df_res.groupby('Week')['Certainty'].mean().round(4))