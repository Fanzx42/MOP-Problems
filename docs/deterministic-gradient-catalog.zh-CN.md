# 确定性多目标优化梯度类算法目录

整理日期：2026-10-21。范围：`C:\Users\89526\PycharmProjects` 与 `F:\Algorithms`。

本文件保留静态清点记录；后续统一代码、公式和调用方式见[统一实现说明](unified-algorithms.zh-CN.md)。
外部源码链接使用扫描机器的绝对路径，仅在拥有对应目录的机器上有效；不属于仓库自带文件。

本轮先建立分类目录、源码入口和版本边界。依据本机 README、算法模块、函数入口与 notebook 代码进行静态整理，未运行算法、比较性能或重新验证收敛性。61 个非隐藏一级目录的逐项处理记录见[项目扫描索引](deterministic-gradient-project-index.zh-CN.md)。

## 收录口径

- 给定问题、初点和参数后，迭代使用完整目标值/梯度，且不依赖每轮随机抽样或随机方向选择。随机生成测试问题、随机选初点和随机安排计时顺序，不单独改变算法的这一分类。
- 收录最速下降、共轭梯度、加速、投影、近端、平滑化、条件梯度和前沿梯度下降。ADMM/增广拉格朗日作为扩展单列。
- 一个外层算法的步长、重启、内层求解器和参数选项作为变体登记，避免按文件数量重复计数。目录中“入口已定位”不代表实现正确性、通用可运行性或理论保证已确认。
- `SGD`、`SGDM` 在多个本地项目中指最速下降，不能按名称认定为随机梯度。相反，固定随机种子也不将随机算法变成确定性算法。
- 当前实现、研究原型、文献参考源码和历史对照分别标记。以下分类是本次整理方案；算法名称保留各项目已有约定。

## 总目录

```text
确定性多目标优化梯度类算法
├── 01 基础共同下降与梯度组合
│   ├── MGDA / GD / 多目标最速下降（L2 最小范数）
│   ├── L1 / LP / minimax 共同下降
│   └── 确定性 Entropic LMO-MGDA 与范数几何
├── 02 步长、缩放与正则化
│   ├── 固定、递减、Armijo、非单调与割线搜索
│   ├── AGM / NAGM 自适应与逐目标归一化
│   ├── AdaGD 三分支与 AdGD2 两分支研究版本
│   ├── C-AMG / F-AMG / MO-Adagrad / NSDMO1–4
│   ├── BBDMO 与逐目标 BB 缩放
│   └── 乘子正则化、自动条件调整与 Silver 步长原型
├── 03 共轭梯度与非精确方向
│   ├── 多目标 CG
│   ├── 非精确 GD / CG
│   └── Lazy / 交替方向研究变体
├── 04 加速、动量与历史修正
│   ├── TFY 光滑加速与 NAG 原型
│   ├── AMG-QP / AMG-LP 及重启、衰减变体
│   ├── AMG-LP-minimax / AMG-LP-MM / AMG-LP-vertex
│   ├── LBM 与历史方法
│   └── ExplicitHistoryNAG / new-AUNAG
├── 05 投影、近端与平滑化
│   ├── 多目标投影梯度
│   ├── PGM / ISTA
│   ├── APG / FISTA / zfista
│   ├── IPG 非精确近端梯度
│   └── 平滑化加速近端原型
├── 06 条件梯度 / Frank–Wolfe
│   ├── 基础 FWMOP 与仿射不变、开环、自适应步长
│   ├── AFW / PFW / 分解不变方向
│   ├── BCG / BPCG / FCFW / SSC
│   ├── QC-MNP（二次结构校正扩展）
│   └── 原始—对偶、鞍点与平滑化版本
├── 07 前沿生成与标量化梯度法
│   ├── FD-SD / Front Descent
│   ├── 固定权重与多权重标量化
│   └── FD-BB / FD-LMQN / FD-N 参考源码
└── 08 约束分裂方法（扩展）
    ├── 多目标 Consensus-ADMM
    ├── 单步对偶投影变体
    └── MO-ALM / 近端点及增广拉格朗日原型
```

## 01 基础共同下降与梯度组合

| 方法或分支 | 已定位入口 | 归档说明 |
|---|---|---|
| MGDA / GD | [MOP-Problems：first_order.py](C:/Users/89526/PycharmProjects/MOP-Problems/src/mop_problems/algorithms/first_order.py)，`MGDA`；[MGDA-LP：GDMOO.py](F:/Algorithms/MGDA-LP/Algos/GDMOO.py)，`mgda_minimize` | 统一接口优先查这两处；GD 在问题库中是 MGDA 别名。 |
| 最速下降的其他实现 | [RegularizedGD：Algorithms.py](C:/Users/89526/PycharmProjects/RegularizedGD/Algorithms.py)，`SGDM`；[Scalarization：SGD_MOP.py](C:/Users/89526/PycharmProjects/Scalarization/algorithms/SGD_MOP.py)，`SGD` | 完整梯度版本；与 AdaSGD、AGDM、LNAG 等同族入口交叉登记，不按项目重复算算法。 |
| L1 最小范数 / LP 方向 | [MGDA-LP：GDMOO.py](F:/Algorithms/MGDA-LP/Algos/GDMOO.py)，`solve_mgda_subproblem_l1`、`mgda_minimize(norm="l1")` | 与 L2 子问题及其方向恢复规则区分。 |
| LP/minimax 非加速下降 | [LNAG：LP_minimax.py](C:/Users/89526/PycharmProjects/LNAG/LP_minimax.py)，`GD_LP_minimax`；[nonaccelerated_lp_vertex.py](C:/Users/89526/PycharmProjects/LNAG/numerical_exp/nonaccelerated_lp_vertex.py) | LP 顶点、minimax、历史修正属于不同分支，不能全部简称 MGDA-LP。 |
| 线性方向原型 | [NAGLinear：Algorithms.py](C:/Users/89526/PycharmProjects/NAGLinear/Algorithms.py)，`SGDM_lin`；[AGDM：Alogrithms.py](C:/Users/89526/PycharmProjects/AGDM/Alogrithms.py)，`SGDM_LP` | 保留原型名称，后续按方向子问题对齐。 |
| Entropic LMO-MGDA，确定性分支 | [DROMOO：lmo_mgda.py](F:/Algorithms/DROMOO/literature_reproduction/lmo_mgda.py)，`LMOConfig(algorithm="deterministic")`、`LMOSolver`；[本地说明](F:/Algorithms/DROMOO/literature_reproduction/LMO_MGDA.md) | 已查到此分支以 `batch=None` 使用完整梯度；收录 l2、sign、spectral 几何选项。随机分支另排除。 |

## 02 步长、缩放与正则化

| 方法或分支 | 已定位入口 | 归档说明 |
|---|---|---|
| 固定 / 递减 / Armijo / AGM / NAGM | [adaptive_step_MOP：Algorithms.py](F:/Algorithms/adaptive_step_MOP/deterministic/algorithms/Algorithms.py)，`GD(stp=...)`；[确定性模块指南](F:/Algorithms/adaptive_step_MOP/deterministic/README.md) | `smooth`=固定步长；`open`=递减；`adaptive`=AGM；`normalized`/`nagm`=归一化 NAGM。这里 NAGM 不表示 Nesterov 加速。 |
| 修复的三分支 AdaGD | [icml_afv_benchmark.py](F:/Algorithms/adaptive_step_MOP/deterministic/experiments/icml_afv_benchmark.py)，`run_batch` | 文件明确说明与通用 AGM 的步长不同，独立登记。 |
| 两分支 AdGD2 扩展 | [adagd_speed_diagnosis.py](F:/Algorithms/adaptive_step_MOP/deterministic/experiments/adagd_speed_diagnosis.py)，`run`；[扩展配置](F:/Algorithms/adaptive_step_MOP/deterministic/experiments/icml_adgd2_extension_benchmark.py) | `adgd2_coupled_b2`、`adgd2_coupled_b3`、`adgd2_direct`；理论条件和诊断分支分别看原文档。稿件中的 AdaPG 称谓留待公式级对齐，不先视为同义词。 |
| C-AMG / F-AMG / MO-Adagrad | [literature_baselines.py](F:/Algorithms/adaptive_step_MOP/deterministic/algorithms/literature_baselines.py)，`solve`；[published_adaptive_steps.py](F:/Algorithms/adaptive_step_MOP/deterministic/algorithms/published_adaptive_steps.py) | 按本地文献复现名称登记；此处 AMG 与 LNAG 的加速 AMG 系列不同。 |
| NSDMO1–4 | [icml_adaptive_literature_benchmark.py](F:/Algorithms/adaptive_step_MOP/deterministic/experiments/icml_adaptive_literature_benchmark.py)，`run_published` | 共享 `nsdmo_curvature`、`nsdmo_update`；四个曲率选项作为同族变体。 |
| BBDMO / BB 缩放 | [MOP-Problems：first_order.py](C:/Users/89526/PycharmProjects/MOP-Problems/src/mop_problems/algorithms/first_order.py)，`BBDMO`；[literature_baselines.py](F:/Algorithms/adaptive_step_MOP/deterministic/algorithms/literature_baselines.py)，`BBDMO-Armijo/Max/Avg` 分支 | 逐目标缩放；单调、最大值非单调、平均值非单调线搜索分列为选项。 |
| 正则化与历史乘子 | [RegularizedGD：Algorithms.py](C:/Users/89526/PycharmProjects/RegularizedGD/Algorithms.py)，`RSGDM`；[统一 rho 实验](F:/Algorithms/adaptive_step_MOP/deterministic/experiments/regularization_stepsize_benchmark.py) | 区分外层方法与 `pfw_for_quad_re` 内层正则化子问题；rho、历史权重和步长是交叉配置。 |
| 自适应、自动调整、Silver 早期原型 | [AdaSGD：Algorithms.py](C:/Users/89526/PycharmProjects/AdaSGD/Algorithms.py)，`SGDM_ada/Mada/auto/nauto/ada_silver/ada_new`；[AdaSGDnonconvex](C:/Users/89526/PycharmProjects/AdaSGDnonconvex/Algorithms.py)；[auto condition notebook](<C:/Users/89526/PycharmProjects/auto condition/Untitled.ipynb>) | 按完整函数名保留；与后续项目的同名实现不自动认定等价。 |
| 割线线搜索 | [secant_LS：Algorithms.py](C:/Users/89526/PycharmProjects/secant_LS/Algorithms.py)，`secent_LS`、`SGDM`；[FW 线搜索 notebook](C:/Users/89526/PycharmProjects/secant_LS_FW/LS_FrankWolfe.ipynb) | GD 与 FW 的线搜索分别归档；`secent_LS` 为源码原拼写。 |

固定数据的全梯度应用另见 [full_gradient_mtl.py](F:/Algorithms/adaptive_step_MOP/deterministic/experiments/full_gradient_mtl.py) 与 [molecular_full_gradient.py](F:/Algorithms/adaptive_step_MOP/deterministic/experiments/molecular_full_gradient.py)。这些是实验入口，不额外算作新算法。

## 03 共轭梯度与非精确方向

| 方法或分支 | 已定位入口 | 归档说明 |
|---|---|---|
| 多目标 CG | [Approximate GD：Algorithms.py](<C:/Users/89526/PycharmProjects/Approximate GD/Algorithms.py>)，`CG`；[RegularizedGD：Algorithms.py](C:/Users/89526/PycharmProjects/RegularizedGD/Algorithms.py)，`CG` | 后续按 `beta_cal` 与 Wolfe 条件细分。 |
| 非精确最速下降 | [Approximate GD：Algorithms.py](<C:/Users/89526/PycharmProjects/Approximate GD/Algorithms.py>)，`SGD_inexact`；[RegularizedGD](C:/Users/89526/PycharmProjects/RegularizedGD/Algorithms.py)，`SGDM_inexact` | 非精确性来自方向子问题容差，不能因此归为随机算法。 |
| 非精确共轭梯度 | 同上两处 `CG_inxact` | 保留源码中的 `inxact` 拼写；不要漏检。 |
| Lazy / 交替方向 | [RegularizedGD](C:/Users/89526/PycharmProjects/RegularizedGD/Algorithms.py)，`SGDM_lazy`、`CG_lazy`、`SGDM_alt`；[Approximate GD](<C:/Users/89526/PycharmProjects/Approximate GD/Algorithms.py>)，`altSGD` | 研究变体，后续需核对方向接受和停止规则。 |

## 04 加速、动量与历史修正

| 方法或分支 | 已定位入口 | 归档说明 |
|---|---|---|
| TFY 光滑加速 | [MOP-Problems：first_order.py](C:/Users/89526/PycharmProjects/MOP-Problems/src/mop_problems/algorithms/first_order.py)，`NesterovAcceleratedGradient`；[LNAG：Algorithms.py](C:/Users/89526/PycharmProjects/LNAG/Algorithms.py)，`AMG_Tanabe` | 问题库明确指 TFY 复合加速方法的光滑特例。 |
| NAG / 加速最速下降原型 | [AGDM：Alogrithms.py](C:/Users/89526/PycharmProjects/AGDM/Alogrithms.py)，`Acc_SGDM_*`；[NAGLinear](C:/Users/89526/PycharmProjects/NAGLinear/Algorithms.py)，`NAG*`；[RegularizedGD](C:/Users/89526/PycharmProjects/RegularizedGD/Algorithms.py)，`NAG`、`NAG_inexact` | 保留 Song/Luo、线性化及非精确等本地函数后缀，后续核对对应公式。 |
| AMG-QP / AMG-LP 与变体 | [LNAG：Algorithms.py](C:/Users/89526/PycharmProjects/LNAG/Algorithms.py)，`AccG`、`AccG_wo_Q`、`AMG_QP`、`AMG_LP`、`AMG_LP_Modified` 及 `*_restart` | QP/LP 方向、重启机制分别登记；还含 MaxMin、Softmin、Preference、smoothing 与 safeguard 研究分支。 |
| 衰减时钟 | [AMG_LP_Decay.py](C:/Users/89526/PycharmProjects/LNAG/AMG_LP_Decay.py)、[AMG_LP_Decay_2.py](C:/Users/89526/PycharmProjects/LNAG/AMG_LP_Decay_2.py)、[AMG_LP_Decay_Tight.py](C:/Users/89526/PycharmProjects/LNAG/AMG_LP_Decay_Tight.py) | 公开名使用 `AMG-LP-decay-tight`；旧 `Decay_2_Budget` 属兼容入口。 |
| AMG-LP-minimax | [LP_minimax.py](C:/Users/89526/PycharmProjects/LNAG/LP_minimax.py)，`AMG_LP_minimax` | 与原 `AMG_LP` 分开。 |
| AMG-LP-minimax-modified / AMG-LP-MM | [amg_lp_mm.py](C:/Users/89526/PycharmProjects/LNAG/numerical_exp/amg_lp_mm.py)，`run_amg_lp_mm` | 与 `Algorithms.AMG_LP_Modified` 分开。 |
| AMG-LP-vertex | [history_lp_vertex_mu.py](C:/Users/89526/PycharmProjects/LNAG/numerical_exp/history_lp_vertex_mu.py)，`run` | LP 顶点候选、历史线段修正与正时钟 IMEX；不合并到随机 SAMG-LP-vertex。 |
| LBM | [LBM.py](C:/Users/89526/PycharmProjects/LNAG/LBM.py)，`LBM` | 当前独立入口；历史动量输运方法。 |
| LBM 的其他分支 | [LNAG 算法说明](C:/Users/89526/PycharmProjects/LNAG/README.md)；[lbm_s.py](C:/Users/89526/PycharmProjects/LNAG/numerical_exp/lbm_s.py)；[lbm_forward_history.py](C:/Users/89526/PycharmProjects/LNAG/numerical_exp/lbm_forward_history.py) | Full-history LBM、LBM-S、Shifted Exact-LBM、legacy `AMG_LP_LBM` 保留独立标签。 |
| 显式历史 / 交替更新 | [ExplicitHistoryNAG.py](C:/Users/89526/PycharmProjects/LNAG/ExplicitHistoryNAG.py)，`explicit_history_nag`；[new_AUNAG.py](C:/Users/89526/PycharmProjects/LNAG/new_AUNAG.py)，`new_AUNAG` | 各自独立入口。 |
| 历史失败对照 | [LBM_old.py](C:/Users/89526/PycharmProjects/LNAG/LBM_old.py)，`LBM_old` | 项目已明确标为原始非收敛 Exact Variant I；仅作历史/反例对照。 |

命名以 [LNAG 的 AGENTS.md](C:/Users/89526/PycharmProjects/LNAG/AGENTS.md) 为准。目录收录研究实现不等于将所有分支列为已证明收敛的方法。

## 05 投影、近端与平滑化

| 方法或分支 | 已定位入口 | 归档说明 |
|---|---|---|
| 投影梯度 | [projMOP：Algorithms.py](C:/Users/89526/PycharmProjects/projMOP/Algorithms.py)，`proj_GD`；[FW_D_MOP：Algorithms.py](C:/Users/89526/PycharmProjects/FW_D_MOP/Algorithms.py)，`ProjGMOP`；[Algo_Lp.py](C:/Users/89526/PycharmProjects/FW_D_MOP/Algo_Lp.py)，`ProjGMOP_L2` | 不因项目名含 FW 而漏掉投影梯度。可行域支持需逐实现核对。 |
| Julia 原始/对偶 PG | [pgmoo.jl](C:/Users/89526/PycharmProjects/julia_primal_dual_frank_wolfe/src/pgmoo.jl)，`primal_pgmoo`、`dual_pgmoo` | 同属投影梯度的不同子问题实现路径。 |
| PGM / ISTA | [MOP-Problems](C:/Users/89526/PycharmProjects/MOP-Problems/src/mop_problems/algorithms/first_order.py)，`ProximalGradient`；[zfista](C:/Users/89526/PycharmProjects/zfista/zfista/proximal_gradient.py)，`minimize_proximal_gradient`；[inexact_PG 的 PGM 入口](C:/Users/89526/PycharmProjects/inexact_PG/Algorithms/pgm.py) | 光滑项加非光滑凸项；与欧氏投影特例交叉引用。 |
| APG / FISTA / zfista | [MOP-Problems](C:/Users/89526/PycharmProjects/MOP-Problems/src/mop_problems/algorithms/first_order.py)，`NesterovAcceleratedProximalGradient`；[zfista：proximal_gradient.py](C:/Users/89526/PycharmProjects/zfista/zfista/proximal_gradient.py) | zfista 将普通与加速模式放在同一求解接口。 |
| IPG | [inexactPGMOO.py](C:/Users/89526/PycharmProjects/inexact_PG/Algorithms/inexactPGMOO.py)，`inexact_prox_grad` | 非精确近端梯度外层；内含对偶权重求解和 ADMM 近端计算。 |
| IPG-COPT 实现路径 | [cvxpy_compare.py](C:/Users/89526/PycharmProjects/inexact_PG/Algorithms/cvxpy_compare.py)，`inexact_prox_grad_cvxpy`；[项目说明](C:/Users/89526/PycharmProjects/inexact_PG/README.md) | 复用 IPG 外层的不同子问题后端，登记为实现变体；Toy、Robust、FairGLasso 有问题适配。 |
| 平滑化加速近端原型 | [smoothing：CB3_LQ.py](C:/Users/89526/PycharmProjects/smoothing/CB3_LQ.py)，`optimize_process`；[BK1.py](C:/Users/89526/PycharmProjects/smoothing/BK1.py) | 有完整梯度、动量与近端子问题；按模型内嵌原型收录，尚非统一通用接口。 |

## 06 条件梯度 / Frank–Wolfe

这里收录的是直接更新原问题变量的外层条件梯度。用于 MGDA 权重 QP 的 `FW_solver.py` 放入辅助组件，不重复算外层算法。

| 方法或分支 | 已定位入口 | 归档说明 |
|---|---|---|
| 标准多目标 FW | [FW_D_MOP：Algorithms.py](C:/Users/89526/PycharmProjects/FW_D_MOP/Algorithms.py)，`FWMOP`；[Julia：core.jl](C:/Users/89526/PycharmProjects/julia_primal_dual_frank_wolfe/src/core.jl)，`frank_wolfe_mop` | Python/Julia 并存，保留接口与梯度矩阵方向约定。 |
| 仿射不变 / 开环 / 线搜索 | [affine_invariant_fwmop：mofw.py](C:/Users/89526/PycharmProjects/affine_invariant_fwmop/algorithms/mofw.py)，`multiobjective_fw`；[AdaCG](C:/Users/89526/PycharmProjects/AdaCG/Algorithms.py)，`CondiGrad` | AdaCG 中 CG 是 conditional gradient；与第 03 章 conjugate gradient 分开。 |
| AFW / PFW / BCG / BPCG | [PFWMOO：pfwmoo](C:/Users/89526/PycharmProjects/PFWMOO/pfwmoo)，`AFWMOO`、`PFWMOO`、`BCGMOO`、`BPCGMOO` | 同名独立模块；只取确定性接口。 |
| FCFW / SSC / QC-MNP | [PFWMOO：pfwmoo](C:/Users/89526/PycharmProjects/PFWMOO/pfwmoo)，`FCFWMOO`、`SSCMOO`、`QCMNPMOO` | QC-MNP 利用二次 Hessian，单列结构扩展；其余不因此继承任意单目标版本的理论。 |
| 分解不变 / away 原型 | [FW_D_MOP](C:/Users/89526/PycharmProjects/FW_D_MOP/Algorithms.py)，`DFWMOP*`、`AFWMOP`；[away-step-primal-dual-FW](C:/Users/89526/PycharmProjects/away-step-primal-dual-FW/algorithms/FW_algo.py)，`DIAFWMOP*` | 按具体方向子问题继续细分，保留旧函数名。 |
| 原始—对偶 / 鞍点 FW | [sp_fw_mop](C:/Users/89526/PycharmProjects/sp_fw_mop/Algorithm/algo.py)，`sp_FWMOP`；[Julia 项目](C:/Users/89526/PycharmProjects/julia_primal_dual_frank_wolfe/src)；[早期 Julia/notebook](C:/Users/89526/PycharmProjects/primal_dual_frankwolfe_mop) | Julia 的 `saddle_fw.jl`、`saddle_afw.jl`、`pdhg.jl` 等还含子问题求解器，外层与内层分层登记。 |
| 平滑化与实验原型 | [smoothed_fwmoo.jl](C:/Users/89526/PycharmProjects/julia_primal_dual_frank_wolfe/src/smoothed_fwmoo.jl)，`dual_fwmoo_smoothed`；[saddle_point_FW notebook](C:/Users/89526/PycharmProjects/saddle_point_FW/Untitled.ipynb)；[secant_LS_FW notebook](C:/Users/89526/PycharmProjects/secant_LS_FW/LS_FrankWolfe.ipynb) | notebook 中分别含鞍点/平滑化方向、外层 FWMOP 和不同线搜索。 |

## 07 前沿生成与标量化梯度法

| 方法或分支 | 已定位入口 | 归档说明 |
|---|---|---|
| FD-SD / Front Descent | [独立复现：algorithm.py](C:/Users/89526/PycharmProjects/outer_appro_CVOP/CVOP/front_descent/front_descent/algorithm.py)，`front_descent`；[说明](C:/Users/89526/PycharmProjects/outer_appro_CVOP/CVOP/front_descent/README.md) | 确定性的共同/部分最速方向、细化、探索与非支配过滤；结果为点集。 |
| FD-SD 对比实现和适配器 | [front_descent.py](C:/Users/89526/PycharmProjects/outer_appro_CVOP/benchmarks/sia_front_gradient/front_descent.py)，`run_front_descent`；[mobench 适配器](C:/Users/89526/PycharmProjects/approximate-algorithm-for-convex-vector-optimization/src/mobench/algorithms/fd_sd.py) | 与独立复现版本分开登记，避免把同名方法的初始化、停止规则混用。 |
| 固定权重梯度标量化 | [linear_scralarization.py](C:/Users/89526/PycharmProjects/Scalarization/algorithms/linear_scralarization.py)，`linear_scralar` | 固定权重的确定性梯度下降；与直接共同下降区分。 |
| 多权重前沿原型 | [MOEAD：Algorithms/nsma.py](C:/Users/89526/PycharmProjects/MOEAD/Algorithms/nsma.py)，`nsma`；[scalarization.py](C:/Users/89526/PycharmProjects/MOEAD/Algorithms/scalarization.py)，`linear` | 此局部实现使用权重方向、单纯形 FW 型更新与非支配过滤，迭代中的随机抽样行已注释；按固定 `X0`、`lam` 的确定性原型登记。不是对整个 MOEAD/NSGA-II 项目的纳入。 |
| FD-BB / FD-LMQN / FD-N | [下载的参考源码](C:/Users/89526/PycharmProjects/outer_appro_CVOP/tmp/fd_framework_source/algorithms/gradient_based) | 已有 `fd_bb.py`、`fd_lmqn.py`、`fd_n.py`；属临时目录中的参考源码，未认定为本地完成复现。FD-N 需 Hessian，列为二阶扩展。 |

## 08 约束分裂方法（扩展）

| 方法或分支 | 已定位入口 | 归档说明 |
|---|---|---|
| 多目标 Consensus-ADMM | [ADMMMOO：consensus2026/projected.py](F:/Algorithms/ADMMMOO/consensus2026/projected.py)；[项目说明](F:/Algorithms/ADMMMOO/README.md) | 确定性分裂方法；外层不是普通 GD。COPT、投影梯度、Newton 是内层路径，分层登记。 |
| one_step 对偶投影 | [one_step.md](F:/Algorithms/ADMMMOO/docs/admm/one_step.md) | 同一 ADMM 框架每块一次对偶投影的可选变体。 |
| MO-ALM / 应用扩展 | [ADMMMOO：applications2026](F:/Algorithms/ADMMMOO/applications2026)；[算法文档索引](F:/Algorithms/ADMMMOO/docs/admm/README.md) | 模型与约束分裂扩展，下一轮再逐个对齐外层公式。 |
| 增广拉格朗日早期原型 | [test：Untitled.ipynb](C:/Users/89526/PycharmProjects/test/Untitled.ipynb)、[Untitled1.ipynb](C:/Users/89526/PycharmProjects/test/Untitled1.ipynb) | 多目标约束模型、有限差分和投影梯度内层，作为原型保留索引。 |

## 辅助组件、排除项与待核对项

1. **方向与近端子问题**：各项目的 `FW_solver.py`、`min_norm_solvers*.py`、`dual_solver_ls.py`、`Proximal_operator.py`，以及 `jl_CGSM` 的鞍点/平滑化 notebook。放入“内层求解器”，不按外层算法重复计算。
2. **步长是横向标签**：固定、Armijo、Wolfe、非单调、BB、Silver、割线等，应和外层方法关联；不能把每种组合都计为独立算法家族。
3. **明确排除随机主线**：`SGDMOO/SGD.py`、`Stochastic GD/SGD.py`、`adaptive_step_MOP/stochastic`、`PFWMOO` 的随机批量实现、LNAG 的 SAMG/双样本分支，以及 DROMOO 的随机训练主线。DROMOO 已核对的确定性 LMO 分支是例外。
4. **条件性候选**：CAGrad 等梯度组合模块可能在全梯度输入下构成确定性步骤，但其现有随机训练入口不能直接据此纳入；本轮仅登记为后续配置级核对项。PCGrad 需要进一步区分随机投影顺序与固定顺序实现；不能只看是否用了全批量。
5. **非梯度主类**：OA、IA、IOA、SIA 等像空间逼近，以及 NSGA-II/MOEA-D 等进化方法，不进入主体。含这些方法的混合项目中的 FD 或梯度局部模块已单独提取。
6. **单目标参考**：`adaptive_GD-master`、`adaptiveGD-dynamical-system`、`AffineInvariantOLFW-main`、`boostfw-master`、`SmoothsetFW`、`Sliver` 等留在参考区；不能因名称接近就算作多目标实现。
7. **备份与空目录**：F 盘 `adaptive_step_MOP-source-backup` 作为备份记录，不重复计数；C 盘同名 `adaptive_step_MOP` 未发现相关源码；`bcg_mop` 仅发现 IDE 目录；`MGDA-LP-acl-recovery` 是权限恢复材料。备份与主目录未做内容等价性判断。
8. **文献不等于实现**：发现 PBBMO、SMBBMO 的文献材料，但本轮未定位到对应的独立算法入口，不把 PDF 名称当作已实现方法。

下一轮可沿此目录逐项补齐：准确名称与文献版本、数学子问题、迭代公式、步长/重启规则、问题假设、停止准则、内层精度、代码状态和可复现实验入口。当前目录不据已有报告做性能排名。
