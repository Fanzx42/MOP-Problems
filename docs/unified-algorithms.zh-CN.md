# 确定性多目标梯度算法：统一实现与数学形式

本目录把本地算法整理成相同的 `solve(problem, x0) -> OptimizeResult` 接口。
输入是确定的起点、目标值及解析/自动微分 Jacobian；算法内部不抽样、不使用随机梯度。
`multistart` 是独立的外层实验工具；要完全确定起点，显式传入 `starts`。

原目录清点见[算法目录](deterministic-gradient-catalog.zh-CN.md)和
[项目索引](deterministic-gradient-project-index.zh-CN.md)。实现位于
[`src/mop_problems/algorithms`](../src/mop_problems/algorithms)，可运行示例为
[`examples/unified_algorithms.py`](../examples/unified_algorithms.py)。

## 1. 目录与调用

```text
algorithms/
├── __init__.py       # 注册、别名、minimize / get_algorithm / multistart
├── first_order.py    # MGDA、BBDMO、PG；保留独立名称的 TFY 加速
├── conjugate.py      # 多目标共轭梯度、beta、Wolfe 搜索、重启
├── conditional.py    # FW、convex-hull away、pairwise、Polytope
├── accelerated.py    # AMG-QP 完整递推及其复合近端扩展
├── augmented.py      # 多目标 PHR-ALM、乘子与罚参数更新
├── splitting.py      # SplitProblem、多目标权重一致性 ADMM
├── proximal.py       # 零项、逐目标不同权重/中心的 L1、近端项合同
├── _subproblem.py    # 带原始-对偶间隙检查的统一 minimax 近端子问题
└── _runtime.py       # 预算、轨迹、驻点残差、结果格式
```

| 类别 | 名称/别名 | 步长或更新策略 |
| --- | --- | --- |
| 最速下降 | `mgda`, `gd`, `steepest_descent` | `fixed`, `armijo`, `nonmonotone`, `nonmonotone_average` |
| BB 多目标下降 | `bbdmo` | 逐目标 BB 缩放 + 上述步长策略 |
| 共轭梯度 | `conjugate_gradient`, `cg` | `armijo`, `wolfe`, `strong_wolfe`；10 种 beta 选项 |
| 近端梯度 | `proximal_gradient`, `pg` | `fixed`, `backtracking` |
| Nesterov 光滑 | `nesterov`, `nag`, `amg_qp` | AMG-QP 的完整递推，给定固定 `L` |
| Nesterov 复合 | `nesterov_proximal`, `apg`, `amg_proximal` | 同一 AMG-QP 时钟及对应的 minimax 近端扩展 |
| 条件梯度 | `frank_wolfe`, `fw` | `diminishing`, `adaptive`, `armijo` |
| Away step | `away_frank_wolfe`, `away_fw` | `adaptive`, `armijo`；显式顶点集合 |
| Pairwise FW | `pairwise_frank_wolfe`, `pairwise_fw` | `adaptive`, `armijo`；显式顶点集合 |
| 增广拉格朗日 | `alm` | 向量近端梯度内层；约束残差驱动罚参数增加 |
| ADMM | `admm` | 两块权重一致性内层；原始/对偶残差平衡 |
| 历史 TFY 版本 | `tfy`, `tfy_proximal` | 既有 TFY/FISTA 时钟与目标差值模型，固定/回溯 |

注册表共有 13 个规范名称；别名不重复计数。
`nesterov*` 现在明确指 AMG-QP；旧 TFY 实验请改用
`TFYAcceleratedGradient` / `TFYAcceleratedProximalGradient` 或 `tfy*`。
原 [`gradient_comparison.py`](../examples/gradient_comparison.py) 已显式使用 TFY，保留原实验含义。
其图中的 NAG/APG 标签仍表示该脚本的 TFY 基线。

```sh
python -m pip install -e ".[algorithms]"
python examples/unified_algorithms.py
```

```python
from mop_problems import get_problem
from mop_problems.algorithms import L1, get_algorithm, minimize

p = get_problem("FON")
x0 = [0.2, -0.1, 0.3]
r = minimize(p, x0, algorithm="cg", beta_rule="PRP+", step_rule="strong_wolfe")
method = get_algorithm("nesterov_proximal", L=20, regularizer=L1([0.1, 0.2]))
apg_result = method.solve(p, x0)
print(r.status, r.fun, r.stationarity_bound, r.diagnostics)
```

`L` 是 `lipschitz` 的别名；二者均设 `step_size=1/L`，不能传入冲突的值。
AMG 使用这个固定光滑常数；其他支持步长的方法把它作为初始步长。
ADMM 的块步长由内层 L-BFGS-B 决定，`step_size/L` 对该内层无效，罚参数用 `penalty` 设置。
参数对象分别是 `SolverOptions`, `CGOptions`, `FWOptions`, `AMGOptions`, `ALMOptions`, `ADMMOptions`。
配置可重复使用，迭代状态均在 `solve()` 内新建。

## 2. 统一符号、问题和子问题

统一最小化向量目标

\[
 F(x)=(f_i(x)+g_i(x))_{i=1}^m,\quad J_i(x)=\nabla f_i(x)^T,
 \quad G(x)\le0,\ H(x)=0.
\]

`Problem` 提供光滑部分 `f` 和 `dF`，以及未归一化的 `G,H,dG,dH`。
`regularizer` 提供凸项 `g`。目前自带 `L1`：
\(g_i(x)=\sum_j c_{ij}|x_j-a_{ij}|\)，允许每个目标不同的系数和中心。
自定义项须提供 `values(x)` 和**加权和的精确近端算子**
`prox(v, step, weights, lower, upper)`；分别对每个目标做 prox 再平均通常不正确。

共享子问题为

\[
 P_h(v,J,b)=\arg\min_{u\in C}
 \left\{\max_i\big[J_i(u-v)+g_i(u)+b_i\big]
             +\frac{\|u-v\|^2}{2h}\right\}.
\]

双目标对偶可区间求根，多目标使用 SciPy 并按原始-对偶间隙检查。
`C` 为变量盒或全空间；FW 使用自己的紧凸域，ALM 另外处理 `G/H`。
只有 ALM 接受一般 `G/H`；ADMM 接受明确的两块线性耦合模型。
其他方法遇到显式约束会报错，避免误把带约束问题当成无约束问题。

## 3. 最速下降和 BBDMO

全空间的最速下降方向满足

\[
 d_k=\arg\min_d\max_i J_i(x_k)d+\tfrac12\|d\|^2
     =-J(x_k)^T\lambda_k,\quad
 \lambda_k\in\arg\min_{\lambda\in\Delta_m}\tfrac12\|J^T\lambda\|^2.
\]

固定步长更新 `x+αd`；Armijo 要求每个目标
\(f_i(x+\alpha d)\le f_i(x)+c\alpha J_i(x)d\)。
每轮从 `step_size` 开始乘 `contraction` 回溯。
`nonmonotone` 使用各目标近 `nonmonotone_window+1` 个值的最大值；
`nonmonotone_average` 使用 Zhang–Hager 型递推平均参考值。
带盒时先把盒纳入方向模型，再沿可行位移回溯，具体尺度见[原基线说明](algorithms.md)。

BBDMO 用 `s=x_k-x_{k-1}`、`y_i=∇f_i(x_k)-∇f_i(x_{k-1})`：

\[
 a_i=\operatorname{clip}_{[a_{\min},a_{\max}]}
 \begin{cases}
 s^Ty_i/\|s\|^2,&s^Ty_i>0,\\
 \|y_i\|/\|s\|,&s^Ty_i<0,\\
 a_{\min},&s^Ty_i=0.
 \end{cases}
\]

将 `J_i` 换成 `J_i/a_i` 求方向，第一步用 MGDA。
它不是给全部目标共用一个 BB 标量。线搜索和最终驻点判断使用原始目标/梯度。
默认 `bb_min=1e-3, bb_max=1e3`；历史 `bb_scales` 记录实际缩放。

## 4. 多目标共轭梯度

定义 \(\phi_k(d)=\max_i J_i(x_k)d\)，`v_k` 为单位模型的最速下降位移：
\(d_k=v_k+\beta_kd_{k-1}\)。为了简洁，记

\[
 a=\phi_k(v_k),\ b=\phi_{k-1}(v_k),\ c=\phi_{k-1}(v_{k-1}),
 \quad d=\phi_k(d_{k-1}),\ e=\phi_{k-1}(d_{k-1}),\ f=\phi_k(v_{k-1}).
\]

| `beta_rule` | 多目标方向最大值形式 |
| --- | --- |
| `FR` | \((-a)/(-c)\) |
| `CD` | \((-a)/(-e)\) |
| `DY` | \((-a)/(d-e)\) |
| `PRP` | \((-a+b)/(-c)\) |
| `HS` | \((-a+b)/(d-e)\) |
| `LS` | \((-a+b)/(-e)\) |
| `HZ` | \((-a+b)/(d-e)-2d(-a+b+f-c)/(d-e)^2\) |
| `PRP+`, `HS+`, `HZ+` | 对相应 beta 取 `max(0,beta)` |

相同目标退化为单目标时，这些式子还原经典参数。
向量版没有唯一推广方式；这里明确选择方向最大值形式，不把动态标量化梯度差冒称为同一公式。
代码对分母做正性和尺度相关浮点退化检查，不用绝对值掩盖错误符号。

Armijo 为 \(f_i(x+\alpha d)\le f_i(x)+c_1\alpha\phi_k(d)\)。
弱 Wolfe 增加 \(\phi_{k+1}(d)\ge c_2\phi_k(d)\)；
强 Wolfe 增加 \(|\phi_{k+1}(d)|\le c_2|\phi_k(d)|\)。
使用扩张/括区间二分搜索，默认 `c1=1e-4, wolfe_c2=0.1`。

重启条件：

- 周期：`restart_interval=None` 表示每 `n_var` 次，`0` 禁用。
- Powell 正交性：\(|v_k^Tv_{k-1}|\ge\eta\|v_k\|^2\)，默认 `powell_restart=0.2`，`None` 禁用。
- 分母退化、非有限 beta、方向不满足 `φ(d) <= descent_fraction*φ(v)`，或边界无可行步长。
- Wolfe 搜索失败后用最速下降方向做 Armijo；仍失败则明确退出。

`diagnostics` 记录 `beta`, `restart`, `direction`, `restart_counts`, `total_restarts`。
默认 `bounds="none"`；`bounds="box"` 是显式的投影位移与边界重启扩展。

## 5. 近端梯度

\[
 x_{k+1}=P_h(x_k,J(x_k),-g(x_k)).
\]

`fixed` 用给定 `h=step_size`；默认 `backtracking` 从给定步长开始，
要求每个光滑目标满足

\[
 f_i(x^+)\le f_i(x)+J_i(x)(x^+-x)+\frac{\|x^+-x\|^2}{2h}.
\]

否则令 `h *= contraction` 并重新求近端子问题。
这是多目标模型上界回溯；正则项通过子问题精确处理，而不是对不可微项做梯度。

## 6. Nesterov：AMG-QP 完整递推

本次按要求直接采用本地 `LNAG/Algorithms.py::AMG_QP` 的整套光滑递推。
默认 \(z_0=x_0,\gamma_0=1,\mu=0\)，给定共同光滑常数 \(L>0\)：

\[
 \tau_k=\frac{\gamma_k+\sqrt{\gamma_k^2+4L\gamma_k}}{2L},\quad
 y_k=\frac{x_k+\tau_kz_k}{1+\tau_k},\quad
 q_k=\mu(y_k-x_k)+\frac{\gamma_k}{\tau_k}(z_k-x_k).
\]

\[
 \lambda_k\in\arg\min_{\lambda\in\Delta_m}
 \frac12\|J(y_k)^T\lambda-q_k\|^2,\qquad w_k=J(y_k)^T\lambda_k,
\]

\[
 z_{k+1}=\frac{\gamma_kz_k+\mu\tau_ky_k-\tau_kw_k}{\gamma_k+\mu\tau_k},\quad
 x_{k+1}=\frac{x_k+\tau_kz_{k+1}}{1+\tau_k},\quad
 \gamma_{k+1}=\frac{\gamma_k+\mu\tau_k}{1+\tau_k}.
\]

注意更新中使用的是 **`w=J.T@lambda`，不是 `w-q`**。
QP 使用确定的均匀初始权重和 pairwise FW，检查其 FW 对偶间隙。
`mu` 需满足 `0<=mu<=L`；如设正值，应由问题的共同强凸性支撑。
默认 `L=1` 只是配置默认值，不是自动估计；实际问题应给定有效的共同上界。
固定 `L` 路径没有 TFY/FISTA 时钟、目标差值项或隐式回溯。

### 带正则项的对应扩展

原 AMG-QP 源码是光滑算法。本次复合版本是下面明确给出的扩展，不声称原源码已经实现它，
也不把光滑算法的理论收敛率未经证明地搬到复合情形。
保持同一 \(\tau,\gamma,y\)，定义

\[
 h_k=\frac{\tau_k^2}{(1+\tau_k)(\gamma_k+\mu\tau_k)},\qquad
 v_k=y_k+h_k\mu(y_k-z_k).
\]

\[
 x_{k+1}=\arg\min_{u\in C}
 \left\{\max_i[J_i(y_k)(u-x_k)+g_i(u)-g_i(x_k)]
       +\frac{\|u-v_k\|^2}{2h_k}\right\},\quad
 z_{k+1}=\frac{(1+\tau_k)x_{k+1}-x_k}{\tau_k}.
\]

即调用 `P_h(v, J(y), J(y)@(v-x)-g(x))`，之后按原式更新 `gamma`。
因 \(q_k=(v_k-x_k)/h_k\)，无正则且全空间时，该模型的对偶恰好还原上述偏移 QP；
`mu=0` 时 \(h_k=1/L\)。零 L1 项直接走同一光滑代码路径，逐步结果一致。

默认 `bounds="none", domain_restart=False` 与原光滑递推一致。
`bounds="box"` 把盒加入复合模型，是约束扩展；`domain_restart=True` 在外推点出盒时重置 `z=x,gamma=gamma0`。
接受点有盒约束并不能保证外推点也在盒内；域受限函数应明确选择保护策略。
每步诊断记录 `tau,gamma,y,z,objective_weights,direction_gap,smoothness`。
最终仍以接受点的单位近端驻点残差判断，代替原脚本的梯度组合范数退出条件。

## 7. 条件梯度、away 和 pairwise

经典多目标 FW 在紧凸集 \(C\) 上解

\[
 s_k\in\arg\min_{s\in C}\max_i J_i(x_k)(s-x_k),\quad
 d_k=s_k-x_k,\quad \theta_k=\max_i J_i(x_k)d_k.
\]

多目标 minimax 解可能在边或内部，因此不能简单选一个顶点代替。
默认域为问题盒，用 LP 直接求解，不枚举盒的全部顶点。
对显式多面体用 `Polytope(vertices)` 表示 \(C=\operatorname{conv}\{v_j\}\)，
`x=V.T@a, a>=0, sum(a)=1`，初始化通过线性可行性求分解。

```python
from mop_problems.algorithms import Polytope, get_algorithm

domain = Polytope([[0, 0], [1, 0], [0, 1]])
method = get_algorithm("pairwise_fw", domain=domain, step_rule="adaptive")
# problem.n_var 必须为 2，起点必须在三角形内
# result = method.solve(problem, [0.4, 0.4])
```

令 `S={j:a_j>0}`。多目标 away 使用活动顶点**凸包**中的分布 `b`：
\(\min_{b\in\Delta_S}\max_i J_i(x)(x-V_S^Tb)\)。
比较 away 与 FW 的最大方向导数，选择更小者。
这是本地 PFWMOO 中 CH-AFW 的形式，单目标时可取顶点解。

Pairwise 联合求
\(\min_{b\in\Delta,\ c\in\Delta_S}\max_i J_i(x)(V^Tb-V_S^Tc)\)。
三种系数变化依次为 `Δa=b-a`、`Δa=a-b`、`Δa=b-c`。
Away/pairwise 的上限为
\(\alpha_{\max}=\min_{\Delta a_j<0}-a_j/\Delta a_j\)，经典 FW 上限为 1。
因此不会使用不适用于一般混合 away 点的单顶点上限公式。

步长策略：

- `diminishing`：经典 FW 使用 \(\alpha_k=2/(k+2)\)，不保证每步逐目标单调。
- `adaptive`：试探曲率 `M`，取 \(\alpha=\min(\alpha_{max},-\phi(d)/(M\|d\|^2))\)，检查所有目标的二次上界；失败则增大 `M`。默认每轮先将上轮 `M` 乘 `contraction=0.5`，失败再除以它。
- `armijo`：从 `min(1,alpha_max)` 缩短，检查每个目标的充分下降。

Away/pairwise 需显式顶点集合，只接受后两种策略。
LP 结果检查原始-对偶间隙；终止统一采用**经典公共 FW gap 的上界**，不以 away gap 代替。
历史记录 `weights,direction_type,step_cap,smoothness`。

## 8. 多目标 PHR 增广拉格朗日（ALM）

对每个目标添加相同的 PHR 罚项：

\[
 \mathcal L_i(x;\bar\lambda,\bar\nu,\rho)
 =f_i(x)+g_i(x)+\bar\nu^TH(x)+\tfrac\rho2\|H(x)\|^2
 +\frac{\|[\bar\lambda+\rho G(x)]_+\|^2-\|\bar\lambda\|^2}{2\rho}.
\]

内层用向量近端梯度和模型回溯，将该向量增广问题解到规定的驻点精度。
光滑梯度是 `J_i + λ_next @ dG + ν_next @ dH`。
初始乘子为零，外层更新

\[
 \lambda_{k+1}=[\bar\lambda_k+\rho_kG(x_{k+1})]_+,\quad
 \nu_{k+1}=\bar\nu_k+\rho_kH(x_{k+1}).
\]

传入下一内层前，乘子分别保护到 `[0,multiplier_bound]` 和
`[-multiplier_bound,multiplier_bound]`。用

\[
 V_k=\max\{\|H(x_{k+1})\|_\infty,
 \|\max(G(x_{k+1}),-\bar\lambda_k/\rho_k)\|_\infty\}
\]

监测进展；若 `V_k > feasibility_contraction*V_previous`，
令 `rho=min(penalty_max,penalty_growth*rho)`，否则保持。
初始 `V_previous=inf`，第一轮不强制增罚。
罚参数达到上限仍不能取得规定进展时明确返回 `penalty_limit`。

默认 `penalty=1, penalty_growth=2, penalty_max=1e8, feasibility_contraction=0.5`，
`multiplier_bound=1e6, max_iter=80, max_inner=300`。
第 k 个内层精度为 `max(0.1*tol, inner_tol*inner_decay**k)`，默认 `inner_tol=0.1, inner_decay=0.5`。
初始罚参数应按约束量级设置；程序不擅自归一化约束。

终止需同时满足单位近端 Pareto-KKT 残差、原始可行性、
\(\|\min(-G,\lambda)\|_\infty\) 互补残差。
`diagnostics` 提供目标权重、两类乘子、内层精度、使用/下轮罚参数及各残差。
一般非凸问题这里只判断一阶条件，不保证全局 Pareto 最优。

## 9. 多目标 ADMM：权重一致性与自适应罚参数

ADMM 不凭空把任意 `Problem` 拆块，显式使用

\[
 F_i(x,z)=f_i(x)+g_i(z),\qquad Ax+Bz=c.
\]

本节 `g` 表示**第二块光滑目标**，与前文的非光滑正则项符号用途不同。
当前实现要求两个光滑 `Problem`，目标数相等，允许各自变量盒，不允许块内额外 `G/H`。
不接受非光滑 ADMM 块或一般非线性耦合约束；这些需要另一个明确的块近端合同。

```python
from mop_problems.algorithms import SplitProblem, MultiobjectiveADMM
# first / second 均是光滑 Problem；A,B,c 维度分别为 (p,nx),(p,nz),(p,)
# split = SplitProblem(first, second, A, B, c)
# result = MultiobjectiveADMM(penalty=1, residual_ratio=10).solve(split, [*x0, *z0])
```

使用本地 ADMMMOO 的 consistency-only 权重一致性形式，
初始目标权重 \(\alpha_0=\beta_0=\mathbf1/m\)，未缩放乘子 `y0=0`。
记 \(r(u,z)=Au+Bz-c\)，`κ=weight_prox>0`。第一块解

\[
 \min_{u\in C_x}\max_{a\in\Delta_m}
 y_k^Tr(u,z_k)+\tfrac{\rho_k}{2}\|r(u,z_k)\|^2
 +a^T[f(u)-f(x_k)]-\tfrac\kappa2\|a-\beta_k\|^2.
\]

固定 u 时，内层最大值的权重为
\(a(u)=\Pi_\Delta(\beta_k+[f(u)-f(x_k)]/\kappa)\)。
对消去 a 的光滑包络用有盒 L-BFGS-B 求解，再检查投影 KKT 残差；
得到 `x_next,alpha_next`。第二块同理，固定 `x_next`，用
\(\beta(z)=\Pi_\Delta(\alpha_{next}+[g(z)-g(z_k)]/\kappa)\)。
两个块都不通过单一固定目标权重来替代多目标权重更新。
块目标凸时，块一阶条件可作为块最优性条件；非凸时仅提供局部一阶检查。

\[
 r_{k+1}=Ax_{k+1}+Bz_{k+1}-c,\quad
 y_{k+1}=y_k+\rho_kr_{k+1},\quad
 s_{k+1}=\rho_kA^TB(z_{k+1}-z_k).
\]

经典残差平衡（[Boyd 等，§3.4.1、式 3.13](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)）：

\[
 \rho_{k+1}=\begin{cases}
 \min(\rho_{max},\tau_{inc}\rho_k),&\|r\|_2>\eta\|s\|_2,\\
 \max(\rho_{min},\rho_k/\tau_{dec}),&\|s\|_2>\eta\|r\|_2,\\
 \rho_k,&\text{其他}.
 \end{cases}
\]

默认 `penalty=1, residual_ratio=10, penalty_increase=penalty_decrease=2`，
`penalty_min=1e-8, penalty_max=1e8, weight_prox=1, inner_tol=1e-8, max_inner=300`。
实现保存**未缩放 y**，改变 rho 时保持 y 不变；如外部使用 `u=y/rho`，需相应重缩放 u。
残差平衡是相对本地固定 rho 版本增加的策略，不据此宣称多目标变罚参数 ADMM 的新收敛定理。

终止除 `r,s` 外，还检查 `||alpha-beta||_inf` 和以
`(alpha+beta)/2` 为共同目标权重的两个块投影 KKT 残差。
所以“仅可行”或“步子很小”不会被误报成功。
诊断保留 `alpha,beta,multiplier,penalty_used,penalty_next` 以及所有分项残差。

## 10. 统一结果、边界与成本口径

全部返回 `OptimizeResult`，包含 `x,fun,smooth_fun,status,message,success,n_iter`、
求值/导数/近端/子问题计数、耗时、`history`、`diagnostics`。
`success` 仅表示所选方法规定的一阶停止条件通过。
`stationarity_bound` 是停止指标；**不同类别的定义和量纲不同，不能直接拿其数值排名**：

| 类别 | `stationarity` | `stationarity_bound` |
| --- | --- | --- |
| MGDA/BBDMO/CG/PG/AMG/TFY | 原始目标单位近端残差 `||P_1(x)-x||inf` | 残差加 `sqrt(2*gap)` |
| FW/away/pairwise | 公共 FW gap 的数值估计 | 由 LP 对偶下界得到的保守 gap 上界 |
| ALM | 使用当前约束乘子的单位近端 KKT 位移 | 其间隙修正上界、可行性、互补残差的最大值 |
| ADMM | 共同权重下的两个块投影 KKT 残差 | KKT、原始/对偶、权重一致性残差的最大值 |

迭代预算的单位：普通方法是接受步；ALM/ADMM 是接受的**外层**步。
内层精度不达标、线搜索失败、不可微、预算耗尽都返回明确失败状态和最后接受点。
新点尚未完成检查时不继承旧点的驻点指标。
参数/维度/不支持的模型错误抛出异常；首次目标求值本身的非法域也可能直接抛错。

`nfev` 包含纯值及联合值/导数调用，`njev` 是其中的子集；两者不要再次相加。
ADMM 中一次完整 `SplitProblem` 调用计算两块，各块内层单独调用计一次，
`block_oracle_calls` 单独记录内层及参考值的单块调用数。
因此 ADMM 的实际底层块公式遍历数是 `2*nfev-block_oracle_calls`；导数同理为
`2*njev-block_oracle_calls`。此处块调用均为联合值/导数调用。
`nsubproblem` 包含诊断与方向模型；ADMM 每个块优化计一次；`nprox` 不包括 LP/光滑 ADMM。
墙钟时间更适合跨不同子问题类型评估计算成本。

默认 MGDA/PG/FW/ALM/ADMM 使用变量盒；CG/AMG 默认全空间以保持对应原递推。
显式 `bounds="none"` 会忽略问题盒，但 `feasible` 仍按原问题边界及约束判断。
所以无约束延拓的驻点也可能 `feasible=False`。
固定步长需要曲率信息；非凸、不可微或不满足方法假设的问题不自动获得收敛保证。

## 11. 来源与验证

本次按数学公式独立实现，未把其他项目整体复制进本仓库。
本地来源的具体文件与检索日期见目录文档；主要对应关系为：

| 本模块 | 本地依据及本次选择 |
| --- | --- |
| `first_order.py` | 仓库已有 MGDA/BBDMO/PG；保留其 BB 三分支和精度认证 |
| `conjugate.py` | `Approximate GD/Algorithms.py` 等方向最大值 beta；增补 LS/HZ 与显式重启；无原脚本 FR/CD/DY 的隐式阻尼 |
| `conditional.py` | `PFWMOO` 的 FW / CH-AFW / 联合 pairwise 形式；统一 LP 原始-对偶检查及可行系数 |
| `accelerated.py` | `C:\Users\89526\PycharmProjects\LNAG\Algorithms.py::AMG_QP`；复合近端模型是本次推导的扩展 |
| `augmented.py` | 本地 ALM 结构；统一为一般 `G<=0,H=0` 的 PHR 型向量增广模型 |
| `splitting.py` | `F:\Algorithms\ADMMMOO\consensus2026\projected.py` 的权重一致性结构；使用隐式块权重包络，并增加经典残差平衡 |

公式核对参考：已有 [MGDA/BBDMO/PG/TFY 文献](references.md#first-order-algorithms)；
[Lucambio Pérez–Prudente 多目标 CG 原论文](https://doi.org/10.1137/17M1126588)及
[作者实现 CGMOP](https://github.com/lfprudente/CGMOP)；
[PHR 保护型增广拉格朗日资料](https://optimization-online.org/2004/09/957/)；
[Boyd 等 ADMM 原始综述](https://stanford.edu/~boyd/papers/admm_distr_stats.html)。

[`test_unified_algorithms.py`](../tests/test_unified_algorithms.py) 使用独立的经典单目标 beta、
闭式梯度凸包投影、独立 epigraph 优化、具有已知乘子的约束二次问题、可解析的两块 ADMM 问题，
并核查 Wolfe 条件、系数/步长上限、罚参数增减、乘子缩放、预算及模型校验。
原有 [`test_algorithms.py`](../tests/test_algorithms.py) 继续检查 MGDA/BBDMO/PG/TFY。
这些是公式和接口验证，不是大规模性能实验或理论收敛证明。
