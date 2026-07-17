# Task-5: 机器学习模型在股票预测中的开发应用 — 规格说明书

## 1. 项目概述

### 1.1 目标
以**兆易创新（603986.SH）** 的日线行情数据为基础，构建多种机器学习分类模型，对股票未来涨跌方向进行判断，完整走通一条从 **数据处理 → 特征工程 → 标签构造 → 模型训练 → 评估分析 → 结果保存** 的 ML 工作流。

### 1.2 样本数据
- **股票**: 兆易创新（603986.SH）
- **数据文件**: `Task-3/兆易创新_daily.csv`
- **时间范围**: 2022-01-04 ~ 2026-07-09（约 1092 个交易日）
- **字段**: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount

### 1.3 工作流总览

```
┌──────────┐    ┌──────────┐    ┌────────────┐    ┌───────────┐
│ 数据加载  │ →  │ 特征工程  │ →  │ 标签构造    │ →  │ 时间序列划分│
│ & 清洗    │    │ (技术指标) │    │ (涨/跌方向)  │    │ (时序CV)   │
└──────────┘    └──────────┘    └────────────┘    └───────────┘
                                                         │
                    ┌──────────────────────────────────────┘
                    ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ 特征重要性 │ ←  │ 评估对比  │ ←  │ 预测    │ ←  │ 模型训练  │
    │ 分析      │    │ (4个分类器)│    │ (测试集)  │    │ (4个分类器)│
    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                         │
                    ┌──────────────────────────────────────┘
                    ▼
              ┌──────────┐
              │ 结果保存  │
              │ (CSV/图片)│
              └──────────┘
```

---

## 2. 数据加载与预处理

### 2.1 加载
- 从 `Task-3/兆易创新_daily.csv` 读取
- 解析 `trade_date` 为日期类型，设为索引
- 按日期升序排列

### 2.2 清洗
- 检查缺失值
- 删除全空行
- 若有除权除息导致的价格跳跃，不做额外处理（沿用前复权数据原则）

---

## 3. 特征工程

### 3.1 价格衍生特征
| 特征名称 | 公式 | 说明 |
|---------|------|------|
| `returns_1d` | `close / pre_close - 1` | 日收益率 |
| `returns_5d` | `close / close.shift(5) - 1` | 5日收益率 |
| `returns_10d` | `close / close.shift(10) - 1` | 10日收益率 |
| `returns_20d` | `close / close.shift(20) - 1` | 20日收益率 |
| `high_low_ratio` | `(high - low) / close` | 日内振幅比 |
| `close_open_ratio` | `(close - open) / open` | 日内涨跌幅比 |
| `volume_ratio_5d` | `vol / vol.rolling(5).mean()` | 5日相对成交量 |

### 3.2 技术指标特征

**均线类：**
| 特征名称 | 说明 |
|---------|------|
| `ma5`, `ma10`, `ma20`, `ma60` | 收盘价的 N 日移动平均 |
| `ma5_ratio` | `close / ma5 - 1` — 价格偏离 5 日均线的程度 |
| `ma20_ratio` | `close / ma20 - 1` |
| `ma5_ma20_cross` | `ma5 / ma20 - 1` — 短期均线与长期均线的偏离度 |

**波动类：**
| 特征名称 | 说明 |
|---------|------|
| `std_5`, `std_10`, `std_20` | 收盘价的 N 日滚动标准差 |
| `atr_14` | 14日平均真实波幅 (Average True Range) |
| `bb_upper`, `bb_lower`, `bb_width` | 布林带（20日, ±2σ）上轨 / 下轨 / 带宽 |

**动量类：**
| 特征名称 | 说明 |
|---------|------|
| `rsi_14` | 14日相对强弱指标 |
| `macd`, `macd_signal`, `macd_hist` | MACD 快线(12)、慢线(26)、信号线(9)及柱状图 |
| `momentum_5` | `close - close.shift(5)` — 5日动量 |

**成交量类：**
| 特征名称 | 说明 |
|---------|------|
| `obv` | 能量潮 (On-Balance Volume) |
| `volume_ma5_ratio` | `vol / vol.rolling(5).mean()` |
| `turnover_rate` | `vol / 日均流通股数`（若无流通股数据则跳过） |

### 3.3 时序滞后特征
- 对主要特征（如 `returns_1d`, `rsi_14`, `macd_hist`, `volume_ratio_5d`）创建滞后 1, 2, 3 天的版本
- 例如: `returns_1d_lag1`, `returns_1d_lag2`, `rsi_14_lag1`

### 3.4 特征预处理
- 对数值特征做 **Z-score 标准化**（StandardScaler）或 **MinMax 归一化**（视模型而定）
- 树模型（RF, XGBoost）不需要归一化，但逻辑回归和 SVM 需要
- 处理无穷值和 NaN（后向填充 + 删除残余 NaN 行）
- **关键**: 在时间序列划分后，**只在训练集上拟合 Scaler**，再 transform 验证集/测试集，避免未来信息泄露

### 3.5 初步特征数量预估
- 价格衍生特征: ~8 个
- 技术指标特征: ~20 个
- 滞后特征: ~10 个
- **合计: ~38 个候选特征**

---

## 4. 标签构造

### 4.1 分类标签（涨/跌二分类）

| 标签名称 | 定义 | 类别 |
|---------|------|------|
| `direction_fwd_1` | `returns_1d_fwd > 0 ? 1 : 0` | 1=次日涨, 0=次日跌/平 |
| `direction_fwd_5` | `returns_5d_fwd > 0 ? 1 : 0` | 1=未来5日涨, 0=未来5日跌/平 |
| `direction_fwd_20` | `returns_20d_fwd > 0.05 ? 1 : (returns_20d_fwd < -0.05 ? -1 : 0)` | 1=大涨, -1=大跌, 0=震荡（三分类可选） |

其中收益率的计算公式为：
- `returns_1d_fwd` = `close.shift(-1) / close - 1`
- `returns_5d_fwd` = `close.shift(-5) / close - 1`

**默认选择**: `direction_fwd_5`（未来5日涨跌方向）作为主要分类标签。

### 4.2 标签构造要点
- 构造标签时必须使用 `shift(-n)` 前移，**不能使用未来信息**
- 标签构造完成后，最后 n 行（n=预测窗口）的标签为 NaN，需删除
- 二分类涨/跌作为主任务，三分类（大涨/震荡/大跌）作为可选扩展

---

## 5. 时间序列划分

### 5.1 基本原则
- **不能使用随机 K-Fold 打乱数据** — 时间序列必须保持顺序
- 禁止未来数据泄露到训练集中

### 5.2 划分方案

**方案 A: 固定时间窗口（默认首选）**
```
完整数据: |<-------- 1092 天 -------->|
训练集:   |<--- 70% (765天) --->|
验证集:   |                     |<10%>|
测试集:   |                            |<20%>|
```
- 按日期排序后切分：训练 70%（约 764 天）、验证 10%（约 109 天）、测试 20%（约 218 天）
- 训练集: 2022-01-04 ~ 2025-01 左右
- 测试集: 最后 218 个交易日

**方案 B: 时间序列扩展窗口交叉验证 (Expanding Window CV)**
```
Fold 1: 训练 [0:600],  验证 [600:700]
Fold 2: 训练 [0:700],  验证 [700:800]
Fold 3: 训练 [0:800],  验证 [800:900]
Fold 4: 训练 [0:900],  验证 [900:1000]
```
- 用于模型选择与超参调优
- 最终模型在完整训练集上训练后在固定测试集上评估

### 5.3 实现方式
- 使用 sklearn 的 `TimeSeriesSplit` 或自定义实现
- 所有特征标准化/归一化只从训练集拟合

---

## 6. 机器学习分类模型

### 6.1 模型清单

| # | 模型 | 核心优势 | 是否需要归一化 |
|---|------|---------|-------------|
| 1 | **逻辑回归 (LogisticRegression)** | 线性决策边界、可解释性强、训练快 | ✅ 是 |
| 2 | **随机森林 (RandomForestClassifier)** | 非线性、抗过拟合、自带特征重要性 | ❌ 否 |
| 3 | **XGBoost (XGBClassifier)** | 梯度提升、高精度、正则化防过拟合 | ❌ 否 |
| 4 | **支持向量机 (SVC)** | 核方法处理非线性、小样本鲁棒 | ✅ 是 |

四个模型覆盖了**线性模型（LR）→ 集成树（RF）→ 梯度提升（XGB）→ 核方法（SVM）** 四种不同的分类范式，便于横向对比。

### 6.2 超参数候选

| 模型 | 关键超参数 |
|------|-----------|
| 逻辑回归 (LR) | `C=[0.01, 0.1, 1, 10]`, `penalty=['l2']`, `solver=['lbfgs']`, `class_weight=['balanced', None]` |
| 随机森林 (RF) | `n_estimators=[100, 300]`, `max_depth=[3, 5, 10, None]`, `min_samples_leaf=[5, 10]`, `class_weight=['balanced', None]` |
| XGBoost | `n_estimators=200`, `max_depth=[3, 5, 7]`, `learning_rate=[0.01, 0.05, 0.1]`, `subsample=[0.8, 1.0]`, `scale_pos_weight=[1, 3]` |
| SVM (SVC) | `kernel=['rbf', 'linear']`, `C=[0.1, 1, 10]`, `gamma=['scale', 'auto']`, `class_weight=['balanced', None]` |

注：超参数调优仅在主模型完成后作为可选扩展。对于涨跌不平衡问题，`class_weight='balanced'` 和 `scale_pos_weight` 参数可帮助模型关注少数类。

---

## 7. 训练与预测流程

### 7.1 Pipeline 实现

```
for each 模型 in 模型列表:
    1. 加载数据 → 构造特征 X → 构造标签 y
    2. 时间序列划分为 train/val/test
    3. 标准化（树模型跳过此步）
    4. 在训练集上 fit 模型
    5. 在验证集上 predict → 评估 → 调整超参（可选）
    6. 在测试集上 predict → 评估
    7. 保存预测结果到 DataFrame
```

### 7.2 预测输出
- 分类标签: 0（跌/平）或 1（涨）
- 预测概率: `predict_proba()` 输出的上涨概率（LR/RF/XGBoost 支持，SVC 需设置 `probability=True`）
- 输出整合: 真实标签、预测标签、上涨概率三列并列

---

## 8. 评估指标

### 8.1 分类评估

| 指标 | 含义 | 适用场景 |
|------|------|---------|
| **Accuracy** | 总体准确率 | 整体表现基线 |
| **Precision** | 精确率（预测上涨的准确度） | 减少误买入 |
| **Recall** | 召回率（实际上涨中被捕获的比例） | 减少踏空 |
| **F1-Score** | F1 综合分数 | Precision 与 Recall 的调和平均 |
| **Confusion Matrix** | 混淆矩阵 | 细粒度分析误判类型 |
| **AUC-ROC** | ROC 曲线下面积 | 综合排序能力，不受阈值影响 |

### 8.2 金融专项评估

| 指标 | 含义 | 说明 |
|------|------|------|
| **Direction Accuracy** | 方向准确率 | 预测涨跌方向与实际一致的比例，本质上等同于 Accuracy |
| **Long-only Sharpe Ratio** | 夏普比率 | 仅在模型预测"涨"时持仓、预测"跌"时空仓的模拟策略夏普比 |
| **Annualized Return** | 年化收益率 | 上述模拟策略的年化收益率 |
| **Max Drawdown** | 最大回撤 | 上述模拟策略的最大回撤 |
| **Win Rate** | 胜率 | 持仓期间上涨交易日占比 |

---

## 9. 特征重要性分析

### 9.1 分析方法

| 模型类型 | 方法 | 说明 |
|---------|------|------|
| 逻辑回归 | **系数大小** | `abs(coef_)` 排序，绝对值越大越重要 |
| 树模型 (RF/XGBoost) | **内置重要性** | `feature_importances_`，基于 impurity 减少或 gain |
| SVM | **Permutation Importance** | 打乱单列特征观察对预测误差的影响 |
| 所有模型 | **SHAP Values** (可选扩展) | 博弈论归因，一致性解释 |

### 9.2 输出格式
- 每个模型输出 Top 15 重要特征及其得分
- 横向对比表: 不同模型 Top N 特征的异同
- 可视化: 条形图展示特征重要性排序

---

## 10. 结果保存

### 10.1 文件输出清单

| 文件 | 内容 |
|------|------|
| `predictions/comparison_classification.csv` | 4 个模型在测试集上的真实标签、预测标签、上涨概率 |
| `metrics/classification_metrics.csv` | 各模型的 Accuracy / Precision / Recall / F1 / AUC 汇总表 |
| `metrics/feature_importance.csv` | 各模型 Top 15 特征重要性汇总 |
| `figures/confusion_matrices.png` | 4 个模型的混淆矩阵拼图 |
| `figures/roc_curves.png` | 4 个模型的 ROC 曲线叠图 |
| `figures/feature_importance_top15.png` | 各模型 Top 15 特征重要性横向对比 |
| `figures/model_metrics_comparison.png` | 各模型评估指标柱状对比图 |
| `figures/equity_curve.png` | 各模型预测信号驱动的模拟持仓净值曲线 |

### 10.2 目录结构
```
Task-5/
├── SPEC.md                           ← 本文件
├── ml_stock_classifier.py            ← 主流程脚本（待实现）
├── feature_engineering.py            ← 特征工程模块（待实现）
├── classifier_trainer.py             ← 模型训练与评估模块（待实现）
├── classifier_visualization.py       ← 可视化模块（待实现）
├── predictions/
│   └── comparison_classification.csv
├── metrics/
│   ├── classification_metrics.csv
│   └── feature_importance.csv
├── figures/
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importance_top15.png
│   ├── model_metrics_comparison.png
│   └── equity_curve.png
└── stock_ml_classifier.ipynb         ← Notebook 集成（待实现）
```

---

## 11. 代码架构设计（规划）

### 模块划分

| 模块 | 职责 | 关键类/函数 |
|------|------|------------|
| `feature_engineering.py` | 特征构造 + 标签构造 + 标准化 | `add_technical_features()`, `add_lag_features()`, `build_features_and_labels()` |
| `classifier_trainer.py` | 4 个分类器初始化、训练、评估 | `get_classifiers()`, `train_evaluate()`, `calc_metrics()` |
| `classifier_visualization.py` | 可视化图表 | `plot_confusion_matrices()`, `plot_roc_curves()`, `plot_feature_importance()`, `plot_equity_curve()`, `plot_model_comparison()` |
| `ml_classifier_pipeline.py` | 编排完整流程 | `run_classification_pipeline()` |

### 数据流向
```
load_data()
    → build_features_and_labels()    ← 构造特征 X 与标签 y (direction_fwd_5)
        → fixed_time_split()          ← 70% 训练 / 10% 验证 / 20% 测试
            → for each classifier:
                1. 标准化（LR/SVC 需要，RF/XGB 跳过）
                2. train → predict
                3. calc_metrics()
            → plot_results() + save_results()
```

---

## 12. 实施路线图

| 阶段 | 内容 | 预计代码量 |
|------|------|-----------|
| **Phase 1** | 数据加载 + 特征工程 + 标签构造 | ~150 行 |
| **Phase 2** | 时间序列划分 + 4 个分类器训练与评估 (LR/RF/XGBoost/SVM) | ~180 行 |
| **Phase 3** | 特征重要性分析 + 可视化图表 (5 张图) | ~200 行 |
| **Phase 4** | 结果保存 + Notebook 集成 | ~100 行 |
| **合计** | | **~630 行** |

---

## 13. 注意事项与约束

1. **时间序列意识** — 任何步骤都不能使用未来数据，包括特征构造（不能使用未来的 close 值）、标准化（不能用未来数据的统计量）、标签构造的 NaN 处理
2. **数据量有限（~1092 条）** — 深度学习模型（LSTM/RNN/Transformer）样本不足，暂不纳入本轮计划
3. **金融预测的高噪声特性** — 准确率 > 55% 即有实际交易价值，> 60% 属于优秀模型。结合 Precision 与 Recall 综合判断
4. **树模型 + SVM 互补** — RF/XGBoost 无需归一化，能处理非线性，自带特征重要性；SVM 擅长高维小样本，二者形成互补
5. **代码可复用性** — 模块化设计，后续可替换为其他股票数据或增加新特征/模型
6. **可视化中国红绿惯例** — 涨=红色, 跌=绿色（与 Task-3 保持一致）
