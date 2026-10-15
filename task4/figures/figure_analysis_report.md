# Task 4 可视化图表详细解析报告

本报告针对 `task4/figures/` 目录下的六张核心图表进行深度解析。这些图表共同构成了 **PADW (Performance-Anchored Dynamic Weighting)** 系统的论证体系，旨在向节目制作人和评审证明该机制的先进性、公平性与鲁棒性。

---

## 1. 机制架构图 (The Concept)
- **文件**: [padw_architecture.svg](file:///Users/hokawing/Downloads/美赛/task4/figures/padw_architecture.svg)
- **核心含义**: 这是系统的“逻辑蓝图”。
- **作用**: 
    - **输入端**: 展示了 Judge Scores（专业度）和 Fan Votes（人气度）双轨输入。
    - **Module A (Skill Anchor)**: 以“漏斗”形式展示了 $\lambda$ 过滤器如何拦截不合符专业标准的投票。
    - **Module B (Time-Dynamic)**: 展示了 $\alpha(t)$ 调节器如何随比赛进程动态调整权重。
    - **Bobby Bones 案例**: 特意标注了“高人气/低技术”选手如何在这里被“软熔断”拦截，直观解释了系统对公平性的保障。

## 2. 数学引擎核心 (The Math)
- **文件**: [padw_math_engine_final.svg](file:///Users/hokawing/Downloads/美赛/task4/figures/padw_math_engine_final.svg)
- **核心含义**: 展示模型的数学灵魂与自动调节机制。
- **作用**:
    - **左图 (α曲线)**: 展示了 Judge 权重如何从 0.4 增长到 0.8。背景色块（红色-黄色-蓝色）明确了“早期粉丝主导”到“后期专业决胜”的战略转型。
    - **右图 (λ曲线)**: 展示了表现锚定函数。**Penalty Zone**（惩罚区）证明了系统对极端低分选手的自动识别与干预能力。
    - **关键标注**: 在曲线拐点标注了 50/50 的 Crossover 点，增强了模型的可解释性。

## 3. 多维性能评估 (The Result)
- **文件**: [padw_comprehensive_eval.svg](file:///Users/hokawing/Downloads/美赛/task4/figures/padw_comprehensive_eval.svg)
- **核心含义**: 一锤定音的综合素质对比。
- **作用**: 
    - 通过五角雷达图对比 **PADW (红色饱和区域)**、**Hybrid System (蓝色虚线)**、**Rank System (点划线)** 以及 **Percentage System (深棕色点线)**。
    - **直观结论**: PADW 在专业度 (Skill)、公平性 (Fairness) 和争议控制上形成了全方位包围，同时保留了极高的参与感 (Engagement)，证明了其是目前最平衡的最优解。

## 4. 关键指标对比瀑布图 (The Impact)
- **文件**: [padw_impact_waterfall.svg](file:///Users/hokawing/Downloads/美赛/task4/figures/padw_impact_waterfall.svg)
- **核心含义**: 量化“战功”，展示具体的性能提升百分比。
- **作用**:
    - 以现有系统为 0 基准。
    - **正面提升 (绿色右向)**: 强调了 Skill Correlation (+6.3%) 和 Fairness (+12%) 的硬核增长。
    - **负面降低 (红色左向)**: 强调了争议率 (-15.2%) 和低分幸存率 (-20%) 的显著下降。
    - **评价**: 该图为评委提供了直接的量化论据，是模型实际落地价值的最佳证明。

## 5. 参数敏感性热力图 (The Robustness)
- **文件**: [padw_sensitivity_heatmap.svg](file:///Users/hokawing/Downloads/美赛/task4/figures/padw_sensitivity_heatmap.svg)
- **核心含义**: 证明模型的鲁棒性与参数选择的科学性。
- **作用**:
    - 展示了在不同的初期与后期权重组合下，模型性能的分布。
    - **结论**: 大面积的深色区域（>0.85 的 ρ 值）表明模型在参数扰动下依然极其稳定，并非靠“凑参数”得出的结果。
    - **星号标注**: 核心参数组合 (0.4, 0.8) 恰好位于高亮稳定区的中心位置。

## 6. 案例复盘：排名修正轨迹图 (The Story)
- **文件**: [padw_case_study_bump.svg](file:///Users/hokawing/Downloads/美赛/task4/figures/padw_case_study_bump.svg)
- **核心含义**: 用 Bobby Bones 真实案例讲述“系统纠偏”的故事。
- **作用**:
    - 通过四种曲线（Reality, PADW, Rank, Judge Only）展示 Bobby 的排名演化。
    - **戏剧冲突**: 现实中 Bobby 夺冠 (Red)，但 PADW (Green) 将其修正至第 4 名，体现了“既尊重粉丝热情，又不违背专业底线”的折中艺术。
    - **对比**: 证明了 PADW 比单纯的 Rank  نظام (Yellow) 更加精准，比 Judge Only (Gray) 更加有人情味。

## 7. 3D 性能空间对比图 (The Pareto Frontier)
- **文件**: [padw_pareto_3d.svg](file:///Users/hokawing/Downloads/美赛/task4/figures/padw_pareto_3d.svg)
- **核心含义**: 证明 PADW 是所有模型方案中的“帕累托最优解 (Pareto Optimal)”。
- **作用**:
    - **维度说明**: X 轴 (Professionalism)、Y 轴 (Fairness)、Z 轴 (Engagement)。
    - **帕累托逻辑**: 帕累托前沿代表了在多目标博弈中，不牺牲一个维度就无法提升另一个维度的极限边界。
    - **视觉强化**: 
        - **PADW (红色巨大球体)**：通过三向投影线（地面、左侧墙面、背面墙面）精确锁定其空间坐标。
        - **结论**: 它最靠近坐标系的“理想顶点”（即三个维度均为 100% 的虚拟点），证明其有效地打破了传统机制在“公正”与“娱乐”之间的零和博弈。
    - **对比含义**: 
        - **Percentage System**：虽然 Z 轴（参与度）极高，但在 X/Y 轴（专业/公平）严重坠落。
        - **Rank System**：虽然 X/Y 轴稳健，但在 Z 轴（参与度）表现垫底。
        - **PADW**：通过动态权重调节，占据了空间中唯一的“平衡高点”。

---
*本报告由 Antigravity 整理，基于 Task 4 研究报告中的仿真数据点 (N=421) 生成。*
