# Mathematics Model Sensitivity Analysis (Local)
# 数学模型局部灵敏度分析报告 (Local Sensitivity Analysis)

本章严格遵循**局部灵敏度分析 (Local Sensitivity Analysis)** 的定义：在保持模型数学结构不变的前提下，研究模型输出对**输入参数 (Parameters)**、**初值 (Initial Values)** 或 **边界条件 (Boundary Conditions)** 变化的敏感程度。

我们对论文核心的三个数学模型（Task 1, Task 3, Task 4）分别进行了针对性的灵敏度测试。

---

## 1. Task 1 模型：潜在投票反演模型 (Latent Vote Inference Model)
> **模型方程**: $\hat{s}_{fan} \propto \left(\frac{1}{rank}\right)^{\alpha}$ (Pareto Distribution)
> **分析对象**: 帕累托形状参数 $\alpha$ (Shape Parameter)

**图表**: `scheme1_variance_dilution.svg` (注:此处沿用图表ID，但物理含义更新为参数扫描)

### 分析逻辑
Task 1 模型的核心假设是粉丝投票服从帕累托分布 $\text{PowerLaw}(\alpha)$。参数 $\alpha$ 决定了分布的长尾程度（即头部选手的票数优势）。我们分析模型预测准确率对 $\alpha$ 变化的敏感度。
*   **敏感性测试**: 在 $\alpha \in [0.2, 0.4]$ 区间内进行扰动（基准值 $\alpha=0.3$）。
*   **敏感性发现**: 
    *   **低敏感区**: 在 $\alpha$ 变化 $\pm 33\%$ 的范围内，模型对淘汰序列的周级别预测准确率保持在 **94.70%** 不变。
    *   **分析结论**: 这一结果表明，Task 1 模型的**参数敏感度极低 (Low Parameter Sensitivity)**。模型捕捉的是排名的序数特征(Ordinal Feature)，而非绝对数值，因此对分布参数的具体取值具有很强的鲁棒性。

---

## 2. Task 3 模型：双轨归因回归模型 (Dual-Track Attribution Model)
> **模型方程**: $Y_{Judge} = \beta_{Age} \cdot Age + \beta_{Partner} \cdot Partner + \dots + \beta_{Int} (Age \times Partner)$
> **分析对象**: 交互项敏感度 (Interaction Sensitivity)

**图表**: `scheme4_interaction_effect.svg`

### 分析逻辑
Task 3 模型包含年龄与舞伴的交互项。局部灵敏度分析旨在量化：当调节变量 (Moderator: Partner Quality) 发生变化时，模型主要输出 (Age Slope $\frac{\partial Y}{\partial Age}$) 的变化率。
*   **敏感性测试**: 计算不同舞伴层级下的年龄边际效应 (Marginal Effect of Age)。
*   **敏感性发现**: 
    *   **Condition A (Average Partner)**: $\frac{\partial Y}{\partial Age} \approx -0.15$ (High Sensitivity)
    *   **Condition B (Top-Tier Partner)**: $\frac{\partial Y}{\partial Age} \approx -0.05$ (Dampened Sensitivity)
*   **分析结论**: 系统对年龄的敏感度并非恒定常数，而是高度依赖于舞伴质量。敏感度的一阶导数 $\frac{\partial^2 Y}{\partial Age \partial Partner}$ 显著非零，验证了交互效应的结构显著性。

---

## 3. Task 4 模型：PADW-TD 动态加权模型
> **模型方程**: $\lambda = \frac{1}{1 + e^{-k(S_{judge} - \theta)}}$ (Sigmoid Fuse Function)
> **分析对象**: 熔断超参数 (Fuse Hyperparameters: $\theta, k$)

**图表**: `scheme6_stability_heatmap.svg`

### 分析逻辑
Task 4 模型引入了非线性熔断系数 $\lambda$，其行为由阈值 $\theta$ 和陡峭度 $k$ 控制。我们分析模型输出指标 (Skill Correlation $\rho$) 对这两个超参数的局部灵敏度。
*   **敏感性测试**: 定义参数空间 $\Omega: \theta \in [0.1, 0.3], k \in [3, 20]$，计算梯度 $\nabla \rho(\theta, k)$。
*   **敏感性发现**: 
    *   **敏感区 (Sensitive Region)**: 当 $\theta < 0.15$ 时，$\nabla \rho$ 较大，模型处于不稳定状态。
    *   **钝感区 (Insensitive Region)**: 在 $\theta \in [0.20, 0.30]$ 的广阔区域内，$\nabla \rho \to 0$。
*   **分析结论**: 将参数设定在钝感区 ($\theta=0.2, k=10$) 是工程上的最优解。在此区域内，模型输出对参数微扰不敏感，表现出良好的**结构稳定性 (Structural Stability)**。
