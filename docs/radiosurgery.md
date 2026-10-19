# Gamma Knife：四目标可分应用与 ADMM 求解

这是一个可以在本仓库独立运行的应用例子。剂量矩阵由合成高斯核生成，**不是患者数据或临床复现**。
默认固定一个等中心、8 个扇区、3 种准直器配置，包含 25 个时间/辅助变量、32 个剂量点和四个目标。
模块位于 [`mop_problems.applications.radiosurgery`](../src/mop_problems/applications/radiosurgery/)，
独立于要求标准采样和 Jacobian 接口的 343 个基准问题注册表。

来源：da Silva 等，*A parallel algorithm for generating Pareto-optimal radiosurgery treatment plans*，
[Medical Physics, 2026, DOI](https://doi.org/10.1002/mp.70454)；
明确公式依据[预印本 v1，§II.A–B 与附录 A](https://arxiv.org/html/2509.08602v1)。
代码是本项目对公式的实现，不是作者 Matlab/GPU 代码的复制。

## 安装与快速运行

在仓库根目录执行：

```sh
python -m pip install -e ".[radiosurgery]"
python examples/radiosurgery_demo.py --output outputs/radiosurgery-demo
```

运行求解器需要有效的 COPT 8 许可证。COPT 通过可选依赖安装，许可证需由使用者自行配置。
仅生成模型、评价目标和求独立参考 LP 时，安装 `.[algorithms]` 提供的 SciPy 即可，代码不会初始化 COPT。
核心基准库仍只依赖 NumPy。

默认比较三组权重下的原文标量化 ADMM、数值优化后的 MO-ADMM 与当前 MO-ALM 基线。
输出包含源代码哈希、原始末状态、迭代历史、独立 LP 审计以及支配性检查。
只运行 ADMM 或选择全部九组权重：

```sh
python examples/radiosurgery_demo.py --methods admm --weights 0 1 2 3 4 5 6 7 8 --output outputs/admm-nine
```

## 原生分离结构

令 $t\ge0$ 为照射配置时间，$\tau\ge0$ 为各等中心的最大扇区时间，$d$ 为剂量，
$x=(t,\tau)$。固定剂量影响矩阵 $K$ 后，

$$
[K\quad0]x-d=0,\qquad Ct-T\tau\le0,\qquad d\ge0,\quad d_O\le D_O.
$$

$C$ 对每个扇区的三个准直器时间求和，$T$ 将各等中心的 $\tau$ 复制到对应扇区。
四个最小化目标为

$$
F(x,d)=\bigl(U_T(d),O_R(d),O_{LD}(d),\mathbf1^\top\tau\bigr).
$$

其中

$$
U_T(d)=\frac1{N_{TS}}\sum_{j\in TS}(1-d_j/D_{TS})_+
       +\frac1{N_{TI}}\sum_{j\in TI}(1-d_j/D_{TI})_+,
$$

$$
O_R(d)=\frac1{N_R}\sum_{j\in R}(d_j/D_R-1)_+,\qquad
O_{LD}(d)=\frac1{N_{LD}}\sum_{j\in LD}(d_j/D_{LD}-1)_+.
$$

因此 $f=(0,0,0,\mathbf1^\top\tau)$，$g=(U_T,O_R,O_{LD},0)$，$A=[K\ 0],B=-I,c=0$。
该单轮固定几何/采样模型是凸的分段线性问题，没有增加人工的时间上界。
论文的临床等中心选择、两轮低剂量重采样与 GPU 实现未复现。

## 求解器与数值改动

- 原文算法固定权重 $w$，求解加权 LP 的对偶图分裂；多列共享矩阵分解并行更新。
- MO-ADMM 保留局部权重正则项与两个块的交替更新。时间块等价为一个凸 QP；
  剂量块对给定权重使用坐标 hinge 近端，再对四维权重采用带线搜索的分段 Newton 步。
- MO-ALM 联合求解两个原变量块的近端上图 QP。此基线与 MO-ADMM 的权重正则机制不同，
  不构成只改变分块方式的消融实验。

数值版缓存 $A^\top A$、二次目标和投影 QP，仅更新变化的系数与右端。
时间更新得到的 $Ax$ 在剂量更新、乘子更新和停止检查之间复用；$B=-I$ 以及坐标选择器用取负与索引计算。
必要的 $Ax$ 和 $A^\top v$ 保留。稠密 Gram 只在初始化形成，适合本例的小型时间变量块；
大规模情形仍应根据矩阵结构选择稀疏或算子实现。

局部 gap 阈值仍为 $2\times10^{-6}$，COPT 容差仍为 $10^{-9}$，剂量权重 gap 阈值仍为 $10^{-12}$。
加速没有放松这些精度要求。

## 罚参数更新与停止条件

参考 [Boyd 等 §3.3–3.4](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)，用

$$
r^{k+1}=Ax^{k+1}-d^{k+1},\qquad
s^{k+1}=-\rho_kA^\top(d^{k+1}-d^k)
$$

构造归一化残差

$$
R_k=\frac{\|r^{k+1}\|_2}{\max(1,\|Ax^{k+1}\|_2,\|d^{k+1}\|_2,\|c\|_2)},\qquad
S_k=\frac{\|s^{k+1}\|_2}{\max(1,\|A^\top y^{k+1}\|_2)}.
$$

默认 $\rho_0=1$，每 10 步检查：$R_k>10S_k$ 时加倍，$S_k>10R_k$ 时减半，
限制到 $[10^{-4},10^4]$，200 步后冻结。分母下限 1 是数值保护，不表示完全尺度不变；
尺度问题参见 [Wohlberg](https://arxiv.org/abs/1704.06209)。
使用未缩放乘子 $y$，改变 $\rho$ 时保持 $y$ 不变。$\eta=\kappa=1$ 保留。

物理残差仅用于调节罚参数。停止仍要求共同权重 KKT 与权重一致性残差都不超过 $10^{-4}$。
此更新是数值策略，不把经典单目标 ADMM 理论直接当作多目标变体的收敛证明。

```python
from mop_problems.applications.radiosurgery import make_case, initial, weight_grid
from mop_problems.applications.radiosurgery.numerics import solve_numeric_admm

case = make_case(202)
x, d = initial(case, 0)
result = solve_numeric_admm(case.problem, x, d, weight_grid()[4], penalty=1.0)
# 固定罚参数使用 balance=None；每次更新记录在 result["penalty_updates"]。
```

## 已保存的测量与可复核证据

完整数值对照包含两个测试种子、两个初值、九组权重、三次计时重复，每种设置 108 次输出。
所有权重顺序求解，单 CPU 线程；计时包括初始化和算法检查，不包括事后独立审计。

| 设置 | 每九次求解耗时中位数 | 单次迭代数中位数 | 数值验收 | 支配改进量大于 $10^{-3}$ |
|---|---:|---:|---:|---:|
| 原实现，固定 0.3 | 20.323 秒 | 150 | 108/108 | 0/108 |
| 数值版，固定 0.3 | 2.228 秒 | 150 | 108/108 | 0/108 |
| 数值版，固定 1 | 3.434 秒 | 190 | 108/108 | 0/108 |
| 数值版，从 1 自适应 | 2.251 秒 | 155 | 108/108 | 0/108 |

固定 0.3 的配对加速中位数为 8.994 倍，108 对迭代次数一致，末状态最大差约 $1.31\times10^{-9}$。
从 1 出发的自适应策略在 83/108 对中更快，最大迭代数从 990 降至 235。
它与验证选定的固定 0.3 耗时接近，并非每个配置都更快。

这些是**迁入前工作区的原始测量**，不重新标为本次移植的计时。
本次将末状态与元数据压缩整理，并在可独立运行的移植代码中重新审计：

- [数值对照摘要](experiments/radiosurgery/numerics/summary.json)、[CSV](experiments/radiosurgery/numerics/runs.csv)、
  [末状态](experiments/radiosurgery/numerics/states.npz)：444 次，含 12 次验证。
- [原始三算法摘要](experiments/radiosurgery/three-methods/summary.json)、
  [CSV](experiments/radiosurgery/three-methods/runs.csv)、[末状态](experiments/radiosurgery/three-methods/states.npz)：
  351 次，含 27 次验证。
- [原始来源、哈希和打包映射](experiments/radiosurgery/provenance.json)。三次计时重复不代表独立实例。

MO-ALM 在原实验中可能立即停在弱有效点：某个非负 hinge 指标已为零，使所有目标同时严格下降成为不可能；
近端步骤就可能选择当前点。KKT 小不代表不存在保持部分目标不变、改善其他目标的方案。
报告保留了独立支配 LP 的结果，不将快速停止解释为较好治疗方案。

重新运行完整数值对照，或重新审计压缩状态：

```sh
python -m mop_problems.applications.radiosurgery.numerics_bench --output outputs/radiosurgery-numerics --repeats 3
python examples/radiosurgery_demo.py --audit-reference docs/experiments/radiosurgery/numerics --output outputs/radiosurgery-audit
python examples/radiosurgery_demo.py --audit-reference docs/experiments/radiosurgery/three-methods --output outputs/radiosurgery-three-audit
python -m pytest tests/test_radiosurgery_model.py tests/test_radiosurgery.py tests/test_radiosurgery_numerics.py
```

模型、物理目标手算和原始/对偶 LP 一致性测试可在没有 COPT 时运行；求解器测试会在未安装或无许可证时明确跳过。
本结果只支持当前小规模合成实例，不作为临床或大规模性能结论。
