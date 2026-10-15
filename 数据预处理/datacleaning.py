#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块 (修正版)
严格对应 final.docx 第二部分：数据预处理与特征工程

修改记录：
1. 调整 LOCF (Last Observation Carried Forward) 顺序：移至标准化之后执行。
   (对应文档：Source 27 "用其上一周的标准化得分填补")
2. 修正退赛处理：在计算标准化基准(Min/Max)时排除退赛选手。
   (对应文档：Source 24 "将其移除出当周评价集合")
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path
from typing import Union
warnings.filterwarnings('ignore')


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """加载原始数据"""
    df = pd.read_csv(filepath)
    print(f"原始数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")
    return df


def get_judge_score_columns(df: pd.DataFrame) -> list:
    """获取所有评委评分列"""
    pattern = r'week\d+_judge\d+_score'
    score_cols = [col for col in df.columns if re.match(pattern, col)]
    return score_cols


def get_weeks_info(df: pd.DataFrame) -> dict:
    """获取所有周次及其对应的评委评分列"""
    score_cols = get_judge_score_columns(df)
    weeks_info = {}
    for col in score_cols:
        match = re.match(r'week(\d+)_judge(\d+)_score', col)
        if match:
            week = int(match.group(1))
            if week not in weeks_info:
                weeks_info[week] = []
            weeks_info[week].append(col)
    return weeks_info


def handle_result_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    [步骤 1: 分级处理]
    
    类型 I：自然淘汰 (Eliminated) -> 保留为 Ground Truth
    类型 II：非正常退赛 (Withdrew) -> 标记为右截尾数据 (Right-Censored)
    类型 III：结构性缺失 -> 后续 LOCF 处理
    
    对应文档 Source 20-25
    """
    df = df.copy()
    
    # 初始化
    df['result_type'] = 'normal'
    df['is_censored'] = False  # 标记是否为右截尾（退赛）
    
    # 类型 I：自然淘汰
    eliminated_mask = df['results'].str.contains('Eliminated', case=False, na=False)
    df.loc[eliminated_mask, 'result_type'] = 'eliminated'
    
    # 类型 II：非正常退赛 (Withdrew)
    # 文档要求：标记为右截尾数据
    withdrew_mask = df['results'].str.contains('Withdrew', case=False, na=False) | \
                    df['results'].str.contains('Quit', case=False, na=False)
    df.loc[withdrew_mask, 'result_type'] = 'withdrew'
    df.loc[withdrew_mask, 'is_censored'] = True
    
    # 类型 III：正常完赛（决赛选手）
    final_mask = df['results'].str.contains('Place', case=False, na=False)
    df.loc[final_mask, 'result_type'] = 'finalist'
    
    print(f"结果分类统计:")
    print(df['result_type'].value_counts())
    
    return df


def convert_scores_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """将评分列转换为数值类型"""
    df = df.copy()
    score_cols = get_judge_score_columns(df)
    
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def calculate_weekly_total_score(df: pd.DataFrame) -> pd.DataFrame:
    """计算每周的总评委分数 (原始分)"""
    df = df.copy()
    weeks_info = get_weeks_info(df)
    
    for week, cols in weeks_info.items():
        total_col = f'week{week}_total_score'
        # 计算总分，如果全是NaN则结果为NaN
        df[total_col] = df[cols].sum(axis=1, skipna=True)
        all_na = df[cols].isna().all(axis=1)
        df.loc[all_na, total_col] = np.nan
    
    return df


def standardize_scores_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """
    [步骤 2: 评委分数标准化]
    
    文档 Source 29-31:
    - 解决满分标准变化和评委人数变化
    - 使用 Min-Max 标准化映射到 [0,1]
    - 关键修正：计算 Min/Max 时排除退赛选手 (Source 24 "移除出当周评价集合")
    """
    df = df.copy()
    weeks_info = get_weeks_info(df)
    
    for week in weeks_info.keys():
        total_col = f'week{week}_total_score'
        normalized_col = f'week{week}_normalized_score'
        
        if total_col in df.columns:
            # 1. 定义"评价集合" (Evaluation Set)
            # 排除当周分数无效的人，以及明确标记为"退赛"且当周无完整表现的人
            # (注：为了稳健，我们使用 'is_censored' 排除 Min/Max 计算，
            # 但如果退赛者当周有分，我们后续仍可以给他算一个归一化分，只是他不决定上下限)
            
            valid_scores_mask = ~df[total_col].isna() & (df[total_col] > 0)
            
            # 严格对照文档 Source 24：退赛者不贡献"规则模拟"，即不作为基准
            # 因此计算 min/max 时排除 is_censored 为 True 的行
            benchmark_mask = valid_scores_mask & (~df['is_censored'])
            
            if benchmark_mask.sum() > 0:
                week_scores = df.loc[benchmark_mask, total_col]
                min_score = week_scores.min()
                max_score = week_scores.max()
                
                # 对所有有分数的人进行标准化 (包括可能退赛但当周有分的人)
                # 使用基准集合的 Min/Max
                if max_score > min_score:
                    df.loc[valid_scores_mask, normalized_col] = (
                        (df.loc[valid_scores_mask, total_col] - min_score) / 
                        (max_score - min_score)
                    )
                    # 修正边界：如果有人的分超过了基准Max（极其罕见），截断到1
                    df.loc[df[normalized_col] > 1, normalized_col] = 1.0
                    df.loc[df[normalized_col] < 0, normalized_col] = 0.0
                else:
                    # 所有人分数相同
                    df.loc[valid_scores_mask, normalized_col] = 0.5
            else:
                # 极端情况：当周全是退赛者或无人得分
                pass
                
    return df


def extract_elimination_info(df: pd.DataFrame) -> pd.DataFrame:
    """提取淘汰周数和名次，用于辅助 LOCF 判断"""
    df = df.copy()
    df['elimination_week'] = np.nan
    df['final_rank'] = np.nan
    
    for idx in df.index:
        result = str(df.loc[idx, 'results'])
        
        # 提取 Week
        match = re.search(r'Week\s*(\d+)', result)
        if match:
            df.loc[idx, 'elimination_week'] = int(match.group(1))
            
        # 提取 Place
        if 'Place' in result:
            rank_match = re.search(r'(\d+)(?:st|nd|rd|th)\s+Place', result)
            if rank_match:
                df.loc[idx, 'final_rank'] = int(rank_match.group(1))
            # 处理 "Winner" 或 "Runner-up" 等特殊词汇
            elif 'Winner' in result or '1st' in result:
                df.loc[idx, 'final_rank'] = 1
    
    return df


def fill_missing_normalized_scores_locf(df: pd.DataFrame) -> pd.DataFrame:
    """
    [步骤 3: 结构性缺失处理 (LOCF)]
    
    文档 Source 27:
    "若选手晋级但某项分为 N/A... 用其上一周的**标准化得分**填补"
    
    逻辑：
    - 只能在选手"存活"期间填补
    - 填补对象是 normalized_score，不是原始分
    """
    df = df.copy()
    weeks_info = get_weeks_info(df)
    sorted_weeks = sorted(weeks_info.keys())
    
    # 遍历每一行
    for idx in df.index:
        # 确定选手的有效生命周期
        elim_week = df.loc[idx, 'elimination_week']
        is_finalist = pd.notna(df.loc[idx, 'final_rank'])
        
        # 如果是决赛选手，假设全赛季活跃；否则活跃到淘汰周
        max_active_week = sorted_weeks[-1] if is_finalist else (elim_week if pd.notna(elim_week) else 0)
        
        # 遍历每一周进行检查
        for i, week in enumerate(sorted_weeks):
            # 如果超过了活跃周期，停止填补
            if week > max_active_week:
                break
                
            current_col = f'week{week}_normalized_score'
            
            # 检查当周是否缺失
            is_missing = pd.isna(df.loc[idx, current_col])
            
            if is_missing:
                # 尝试用上一周填补
                if i > 0:
                    prev_week = sorted_weeks[i-1]
                    prev_col = f'week{prev_week}_normalized_score'
                    prev_val = df.loc[idx, prev_col]
                    
                    if pd.notna(prev_val):
                        df.loc[idx, current_col] = prev_val
                        # print(f"LOCF Applied: {df.loc[idx, 'celebrity_name']} Week {week} filled with {prev_val:.3f}")
    
    return df


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """创建汇总统计"""
    df = df.copy()
    weeks_info = get_weeks_info(df)
    normalized_cols = [f'week{w}_normalized_score' for w in weeks_info.keys()]
    
    # 仅计算存在的列
    valid_cols = [c for c in normalized_cols if c in df.columns]
    
    if valid_cols:
        df['avg_normalized_score'] = df[valid_cols].mean(axis=1)
        df['weeks_competed'] = df[valid_cols].notna().sum(axis=1)
    
    return df


def clean_demographic_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗人口统计学数据 (保持原逻辑)"""
    df = df.copy()
    if 'celebrity_age_during_season' in df.columns:
        df['celebrity_age_during_season'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce')
    
    if 'celebrity_industry' in df.columns:
        df['industry_category'] = df['celebrity_industry'].str.strip()
    
    return df


def resolve_path(path: Union[str, Path], base_dir: Path) -> Path:
    """将相对路径解析为相对于当前脚本的路径"""
    p = Path(path)
    if p.is_absolute():
        return p
    return base_dir / p


def process_data_pipeline(input_path: Union[str, Path], output_path: Union[str, Path] = None) -> pd.DataFrame:
    """
    修正后的数据处理流水线
    """
    print("=" * 60)
    print("开始数据预处理 (Strict Compliance Mode)...")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent
    input_path = resolve_path(input_path, base_dir)
    if output_path:
        output_path = resolve_path(output_path, base_dir)
    
    # 1. 加载数据
    print("[1/7] 加载原始数据...")
    df = load_data(input_path)
    
    # 2. 结果分类
    print("[2/7] 分类结果 (淘汰/退赛/晋级)...")
    df = handle_result_classification(df)
    
    # 3. 基础清洗与原始分计算
    print("[3/7] 转换数值并计算原始总分...")
    df = convert_scores_to_numeric(df)
    df = calculate_weekly_total_score(df)
    
    # 4. 标准化 (修正点：排除退赛者影响 Min/Max)
    print("[4/7] 执行 Min-Max 标准化 (排除退赛者)...")
    df = standardize_scores_minmax(df)
    
    # 5. 提取生命周期信息 (为 LOCF 做准备)
    print("[5/7] 提取淘汰与名次信息...")
    df = extract_elimination_info(df)
    
    # 6. 结构性缺失填补 (修正点：在标准化后进行 LOCF)
    print("[6/7] 执行 LOCF (基于标准化分数)...")
    df = fill_missing_normalized_scores_locf(df)
    
    # 7. 汇总与输出
    print("[7/7] 生成统计特征...")
    df = create_summary_statistics(df)
    df = clean_demographic_data(df)
    
    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n处理后的数据已保存至: {output_path}")
    
    return df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "2026_MCM_Problem_C_Data.csv"
    output_file = base_dir / "cleaned_data_final.csv"
    
    df_cleaned = process_data_pipeline(input_file, output_file)
    
    print("\n检查数据一致性:")
    print("-" * 30)
    print(df_cleaned[['celebrity_name', 'result_type', 'avg_normalized_score']].head())
