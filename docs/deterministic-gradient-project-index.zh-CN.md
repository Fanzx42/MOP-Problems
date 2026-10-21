# 确定性多目标梯度算法：项目扫描索引

整理日期：2026-10-21。返回[算法分类目录](deterministic-gradient-catalog.zh-CN.md)。

共登记 55 + 6 = 61 个非隐藏一级目录。目录枚举包含被 Git 忽略的研究源码；依赖环境、缓存、数据和生成结果不作为独立算法项目。表中“未定位”仅表示本轮静态筛查结果。隐藏 checkpoint、备份快照和下载源码不重复计数。

## 扫描边界

- 查阅了主要说明、函数/类入口以及相关迭代和采样代码；notebook 读取单元格源码。未执行研究脚本或大型实验。
- 已读取 `LNAG/AGENTS.md` 与 `inexact_PG/AGENTS.md`，沿用算法命名。
- `inexact_PG/.local_archive/cleanup_20261005/audit_work/export-review-wnnrh17f` 返回拒绝访问。现行主目录可读取；此旧归档未纳入完整性承诺。
- `adaptive_step_MOP-source-backup` 只作备份登记，不能据此断言与主目录内容完全相同。
- 此次交付为目录和入口索引；未移动、重命名或合并原算法文件。

## C:\Users\89526\PycharmProjects

| 项目目录 | 本轮归类 | 入口 | 说明 |
|---|---|---|---|
| [AdaCG](<C:/Users/89526/PycharmProjects/AdaCG>) | 纳入 06 | [Algorithms.py](<C:/Users/89526/PycharmProjects/AdaCG/Algorithms.py>) | CondiGrad；这里 CG 指条件梯度。 |
| [adaptive_GD-master](<C:/Users/89526/PycharmProjects/adaptive_GD-master>) | 单目标参考 | [algorithms.py](<C:/Users/89526/PycharmProjects/adaptive_GD-master/algorithms.py>) | 单目标自适应 GD、加速 GD 与训练材料。 |
| [adaptive_step_MOP](<C:/Users/89526/PycharmProjects/adaptive_step_MOP>) | 无相关源码 | — | C 盘同名目录未发现相关源码；有效确定性入口见 F 盘。 |
| [adaptiveGD-dynamical-system](<C:/Users/89526/PycharmProjects/adaptiveGD-dynamical-system>) | 单目标参考 | [README.md](<C:/Users/89526/PycharmProjects/adaptiveGD-dynamical-system/README.md>) | 统一 Lyapunov 步长框架；当前模型为标量 f。 |
| [AdaSGD](<C:/Users/89526/PycharmProjects/AdaSGD>) | 纳入 01、02 | [Algorithms.py](<C:/Users/89526/PycharmProjects/AdaSGD/Algorithms.py>) | SGDM 及自适应、自动条件调整、Silver 原型；使用完整梯度。 |
| [AdaSGDnonconvex](<C:/Users/89526/PycharmProjects/AdaSGDnonconvex>) | 纳入 01、02 | [Algorithms.py](<C:/Users/89526/PycharmProjects/AdaSGDnonconvex/Algorithms.py>) | 非凸自适应、归一化步长与基础 SGDM。 |
| [affine_invariant_fwmop](<C:/Users/89526/PycharmProjects/affine_invariant_fwmop>) | 纳入 06 | [algorithms/mofw.py](<C:/Users/89526/PycharmProjects/affine_invariant_fwmop/algorithms/mofw.py>) | 多目标 FW、开环和线搜索。 |
| [AffineInvariantOLFW-main](<C:/Users/89526/PycharmProjects/AffineInvariantOLFW-main>) | 单目标参考 | [README.md](<C:/Users/89526/PycharmProjects/AffineInvariantOLFW-main/README.md>) | 单目标仿射不变 FW 参考项目。 |
| [AffInvFW](<C:/Users/89526/PycharmProjects/AffInvFW>) | 辅助笔记 | [Untitled.ipynb](<C:/Users/89526/PycharmProjects/AffInvFW/Untitled.ipynb>) | 函数/增长关系作图；未定位独立多目标求解器。 |
| [AGDM](<C:/Users/89526/PycharmProjects/AGDM>) | 纳入 01、04 | [Alogrithms.py](<C:/Users/89526/PycharmProjects/AGDM/Alogrithms.py>) | 最速下降、LP 方向及 Song/Luo 等本地命名加速原型。 |
| [AGP](<C:/Users/89526/PycharmProjects/AGP>) | 内层/鞍点参考 | [algorithm/AGP.py](<C:/Users/89526/PycharmProjects/AGP/algorithm/AGP.py>) | 双线性鞍点的交替梯度投影；不是当前直接 MOP 外层接口。 |
| [Approximate GD](<C:/Users/89526/PycharmProjects/Approximate GD>) | 纳入 01、03 | [Algorithms.py](<C:/Users/89526/PycharmProjects/Approximate GD/Algorithms.py>) | SGD、CG、非精确方向与交替原型。 |
| [approximate-algorithm-for-convex-vector-optimization](<C:/Users/89526/PycharmProjects/approximate-algorithm-for-convex-vector-optimization>) | 部分纳入 07 | [src/mobench/algorithms/fd_sd.py](<C:/Users/89526/PycharmProjects/approximate-algorithm-for-convex-vector-optimization/src/mobench/algorithms/fd_sd.py>) | 保留 Front Descent 适配；OA/IA 等及进化主线排除。 |
| [auto condition](<C:/Users/89526/PycharmProjects/auto condition>) | 纳入 02；notebook 原型 | [Untitled.ipynb](<C:/Users/89526/PycharmProjects/auto condition/Untitled.ipynb>) | SGDM_ada、SGDM_auto、SGDM_nauto。 |
| [away-step-primal-dual-FW](<C:/Users/89526/PycharmProjects/away-step-primal-dual-FW>) | 纳入 06 | [algorithms/FW_algo.py](<C:/Users/89526/PycharmProjects/away-step-primal-dual-FW/algorithms/FW_algo.py>) | FWMOP、away/分解不变与对偶版本。 |
| [bcg_mop](<C:/Users/89526/PycharmProjects/bcg_mop>) | 无相关源码 | — | 仅发现 .idea；BCG 多目标实现位于 PFWMOO。 |
| [boostfw-master](<C:/Users/89526/PycharmProjects/boostfw-master>) | 单目标参考 | [README.md](<C:/Users/89526/PycharmProjects/boostfw-master/README.md>) | Boosting Frank-Wolfe 参考实现。 |
| [complexity-main](<C:/Users/89526/PycharmProjects/complexity-main>) | 辅助参考 | [README.md](<C:/Users/89526/PycharmProjects/complexity-main/README.md>) | 线性最小化与投影计算复杂度材料。 |
| [data](<C:/Users/89526/PycharmProjects/data>) | 数据目录 | — | 不作为独立算法项目。 |
| [FW_D_MOP](<C:/Users/89526/PycharmProjects/FW_D_MOP>) | 纳入 05、06 | [Algorithms.py](<C:/Users/89526/PycharmProjects/FW_D_MOP/Algorithms.py>) | FW、away、分解不变、原始/对偶及投影梯度。 |
| [huawei](<C:/Users/89526/PycharmProjects/huawei>) | 数据/应用辅助 | [dataset/data.py](<C:/Users/89526/PycharmProjects/huawei/dataset/data.py>) | 已定位数据处理文件；未定位独立 MOP 求解器。 |
| [inexact_PG](<C:/Users/89526/PycharmProjects/inexact_PG>) | 纳入 05 | [README.md](<C:/Users/89526/PycharmProjects/inexact_PG/README.md>) | IPG、IPG-COPT、PGM 与内层近端/对偶模块；旧归档有一处访问受限。 |
| [jl_CGSM](<C:/Users/89526/PycharmProjects/jl_CGSM>) | 内层求解器 | [subproblem.ipynb](<C:/Users/89526/PycharmProjects/jl_CGSM/subproblem.ipynb>) | Julia 鞍点与平滑化 FW 子问题。 |
| [julia_primal_dual_frank_wolfe](<C:/Users/89526/PycharmProjects/julia_primal_dual_frank_wolfe>) | 纳入 05、06 | [README.md](<C:/Users/89526/PycharmProjects/julia_primal_dual_frank_wolfe/README.md>) | Julia 多目标 FW、PG、平滑化及原始/对偶子问题。 |
| [julia_test](<C:/Users/89526/PycharmProjects/julia_test>) | 单目标示例 | [Frank-wolfe_example.ipynb](<C:/Users/89526/PycharmProjects/julia_test/Frank-wolfe_example.ipynb>) | Julia FrankWolfe 包调用示例。 |
| [LNAG](<C:/Users/89526/PycharmProjects/LNAG>) | 纳入 01、04 | [README.md](<C:/Users/89526/PycharmProjects/LNAG/README.md>) | AMG、LP/minimax、历史与 LBM 主线；随机分支和历史失败对照分别标记。 |
| [MOEAD](<C:/Users/89526/PycharmProjects/MOEAD>) | 部分纳入 07 | [Algorithms/nsma.py](<C:/Users/89526/PycharmProjects/MOEAD/Algorithms/nsma.py>) | 局部多权重梯度/FW 型前沿原型可收录；进化算法排除。 |
| [MOP-Problems](<C:/Users/89526/PycharmProjects/MOP-Problems>) | 纳入 01、02、04、05 | [docs/algorithms.md](<C:/Users/89526/PycharmProjects/MOP-Problems/docs/algorithms.md>) | 问题库另有 MGDA、BBDMO、PG、NAG/APG 五种基线接口。 |
| [multi-class](<C:/Users/89526/PycharmProjects/multi-class>) | 应用材料 | — | 本轮未定位可独立归档的 MOP 算法入口。 |
| [multiSVM](<C:/Users/89526/PycharmProjects/multiSVM>) | 应用适配 | [SVM/SVM.py](<C:/Users/89526/PycharmProjects/multiSVM/SVM/SVM.py>) | 调用 FWMOP/投影/分解不变算法的 SVM 实验；未作为新算法家族。 |
| [NAGLinear](<C:/Users/89526/PycharmProjects/NAGLinear>) | 纳入 01、04 | [Algorithms.py](<C:/Users/89526/PycharmProjects/NAGLinear/Algorithms.py>) | SGDM_lin、NAG、线性方向、重启与修正原型。 |
| [outer_appro_CVOP](<C:/Users/89526/PycharmProjects/outer_appro_CVOP>) | 部分纳入 07 | [CVOP/front_descent/README.md](<C:/Users/89526/PycharmProjects/outer_appro_CVOP/CVOP/front_descent/README.md>) | FD 独立复现/适配与临时参考源码；IA/IOA/OA 不进梯度主体。 |
| [PCGrad-master](<C:/Users/89526/PycharmProjects/PCGrad-master>) | 随机性待专项核对 | [PCGrad_tf.py](<C:/Users/89526/PycharmProjects/PCGrad-master/PCGrad_tf.py>) | 梯度投影组合实现；存在 shuffle 调用，须核对实际生效与顺序设置后再认定确定性入口。 |
| [PEP_MOP](<C:/Users/89526/PycharmProjects/PEP_MOP>) | 理论/验证辅助 | — | MGDA 性能估计、最坏情形与步长分析；本轮未按独立通用外层求解器登记。 |
| [PFWMOO](<C:/Users/89526/PycharmProjects/PFWMOO>) | 部分纳入 06 | [README.md](<C:/Users/89526/PycharmProjects/PFWMOO/README.md>) | 确定性 FW/AFW/PFW/BCG/BPCG/FCFW/SSC/QC；随机批量和 CLM 分支排除。 |
| [plotting_of_FWM](<C:/Users/89526/PycharmProjects/plotting_of_FWM>) | 作图/单目标参考 | [Sliver_stepsize.ipynb](<C:/Users/89526/PycharmProjects/plotting_of_FWM/Sliver_stepsize.ipynb>) | Silver 步长与单目标梯度示例；另一 notebook 为未完成片段。 |
| [point_to_set](<C:/Users/89526/PycharmProjects/point_to_set>) | 辅助笔记 | [Untitled.ipynb](<C:/Users/89526/PycharmProjects/point_to_set/Untitled.ipynb>) | QP 与点到 Pareto 集的实验笔记。 |
| [primal_dual_frankwolfe_mop](<C:/Users/89526/PycharmProjects/primal_dual_frankwolfe_mop>) | 纳入 06；原型 | [Lp_frank_wolfe.jl](<C:/Users/89526/PycharmProjects/primal_dual_frankwolfe_mop/Lp_frank_wolfe.jl>) | 早期 Julia/notebook 原始—对偶与不同可行域实现。 |
| [projMOP](<C:/Users/89526/PycharmProjects/projMOP>) | 纳入 01、04、05 | [Algorithms.py](<C:/Users/89526/PycharmProjects/projMOP/Algorithms.py>) | GD、proj_GD 以及加速 QP/LP 原型。 |
| [pythonProject](<C:/Users/89526/PycharmProjects/pythonProject>) | 无相关源码 | — | 未发现本轮目标源码。 |
| [RegularizedGD](<C:/Users/89526/PycharmProjects/RegularizedGD>) | 纳入 01—04 | [Algorithms.py](<C:/Users/89526/PycharmProjects/RegularizedGD/Algorithms.py>) | 基础、正则、自适应、CG、lazy、非精确及 NAG；训练示例单列排除。 |
| [saddle_point_FW](<C:/Users/89526/PycharmProjects/saddle_point_FW>) | 纳入 06；notebook 原型 | [Untitled.ipynb](<C:/Users/89526/PycharmProjects/saddle_point_FW/Untitled.ipynb>) | 鞍点、optimistic/extra、平滑化子问题和 FWMOP_L2 外层。 |
| [Scalarization](<C:/Users/89526/PycharmProjects/Scalarization>) | 纳入 01、07 | [algorithms/linear_scralarization.py](<C:/Users/89526/PycharmProjects/Scalarization/algorithms/linear_scralarization.py>) | 固定权重梯度及 MGDA；Tchebycheff_scalarization.py 当前为空文件。 |
| [secant_LS](<C:/Users/89526/PycharmProjects/secant_LS>) | 纳入 02、04 | [Algorithms.py](<C:/Users/89526/PycharmProjects/secant_LS/Algorithms.py>) | 割线/Armijo 的 SGDM 和 SGDM_Nestrov；进化对照不收录。 |
| [secant_LS_FW](<C:/Users/89526/PycharmProjects/secant_LS_FW>) | 纳入 02、06；notebook 原型 | [LS_FrankWolfe.ipynb](<C:/Users/89526/PycharmProjects/secant_LS_FW/LS_FrankWolfe.ipynb>) | FW 与割线、BBDMO、BBlin 线搜索。 |
| [SGA](<C:/Users/89526/PycharmProjects/SGA>) | 鞍点/对抗学习参考 | — | 补充材料以 Alt-GDA/IQC/GAN 为主；未定位直接确定性 MOP 外层入口。 |
| [SGDMOO](<C:/Users/89526/PycharmProjects/SGDMOO>) | 随机算法排除 | [SGD.py](<C:/Users/89526/PycharmProjects/SGDMOO/SGD.py>) | 通用 mini-batch SGD；本目录名称本身不能作为确定性证据。 |
| [Sliver](<C:/Users/89526/PycharmProjects/Sliver>) | 单目标步长参考 | [Untitled.ipynb](<C:/Users/89526/PycharmProjects/Sliver/Untitled.ipynb>) | Silver 步长与普通/加速梯度示例。 |
| [smoothing](<C:/Users/89526/PycharmProjects/smoothing>) | 纳入 05；模型内嵌原型 | [CB3_LQ.py](<C:/Users/89526/PycharmProjects/smoothing/CB3_LQ.py>) | 平滑化、动量和近端梯度；另有 BK1。 |
| [SmoothsetFW](<C:/Users/89526/PycharmProjects/SmoothsetFW>) | 单目标参考 | [frank_wolfe.py](<C:/Users/89526/PycharmProjects/SmoothsetFW/frank_wolfe.py>) | 单目标 FW 接口与可行域/步长模块。 |
| [sp_fw_mop](<C:/Users/89526/PycharmProjects/sp_fw_mop>) | 纳入 06 | [Algorithm/algo.py](<C:/Users/89526/PycharmProjects/sp_fw_mop/Algorithm/algo.py>) | FWMOP、sp_FWMOP。 |
| [Stochastic GD](<C:/Users/89526/PycharmProjects/Stochastic GD>) | 随机算法排除 | [SGD.py](<C:/Users/89526/PycharmProjects/Stochastic GD/SGD.py>) | 随机梯度代码。 |
| [supplement](<C:/Users/89526/PycharmProjects/supplement>) | 鞍点/对抗学习参考 | [Alt-GDA-GAN/README.md](<C:/Users/89526/PycharmProjects/supplement/Alt-GDA-GAN/README.md>) | Alt-GDA/IQC/GAN 补充材料；不作为 MOP 梯度家族。 |
| [test](<C:/Users/89526/PycharmProjects/test>) | 纳入 08；notebook 原型 | [Untitled1.ipynb](<C:/Users/89526/PycharmProjects/test/Untitled1.ipynb>) | 约束 MOP 增广拉格朗日、有限差分和投影梯度内层。 |
| [zfista](<C:/Users/89526/PycharmProjects/zfista>) | 纳入 05 | [README.md](<C:/Users/89526/PycharmProjects/zfista/README.md>) | 多目标 ISTA/FISTA 与新动量因子接口。 |

## F:\Algorithms

| 项目目录 | 本轮归类 | 入口 | 说明 |
|---|---|---|---|
| [adaptive_step_MOP](<F:/Algorithms/adaptive_step_MOP>) | 纳入 01、02 | [deterministic/README.md](<F:/Algorithms/adaptive_step_MOP/deterministic/README.md>) | 只收录确定性算法及完整梯度应用；stochastic 分支排除。 |
| [adaptive_step_MOP-source-backup](<F:/Algorithms/adaptive_step_MOP-source-backup>) | 备份，不重复计数 | [README.md](<F:/Algorithms/adaptive_step_MOP-source-backup/README.md>) | 保留来源记录；未对主目录与备份做全量逐文件差异审计。 |
| [ADMMMOO](<F:/Algorithms/ADMMMOO>) | 纳入 08 扩展 | [README.md](<F:/Algorithms/ADMMMOO/README.md>) | Consensus-ADMM、one_step、MO-ALM 与应用分裂；内层 PG/FW/Newton 单独分层。 |
| [DROMOO](<F:/Algorithms/DROMOO>) | 部分纳入 01 | [literature_reproduction/LMO_MGDA.md](<F:/Algorithms/DROMOO/literature_reproduction/LMO_MGDA.md>) | 确定性 Entropic LMO-MGDA 已定位；其余随机主线排除，全梯度组合候选待核对。 |
| [MGDA-LP](<F:/Algorithms/MGDA-LP>) | 纳入 01 | [Algos/GDMOO.py](<F:/Algorithms/MGDA-LP/Algos/GDMOO.py>) | mgda_minimize；L2/PFW 与 L1/LP 两类方向。 |
| [MGDA-LP-acl-recovery](<F:/Algorithms/MGDA-LP-acl-recovery>) | 非算法材料 | — | 权限备份与恢复脚本。 |
