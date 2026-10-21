# 一阶多目标算法 / First-order baselines

本页说明原有 MGDA/GD、BBDMO、PG 和 TFY 五类基线。
新增 CG、FW/away/pairwise、ALM、ADMM 及 AMG-QP 完整递推，见
[确定性多目标算法统一说明](unified-algorithms.zh-CN.md)。
下表中的历史加速算法现以 `tfy` / `tfy_proximal` 显式命名，`nesterov*` 已改为 AMG-QP。

| 调用名 | 类 | 定义与实现选择 |
| --- | --- | --- |
| `mgda` / `gd` | `MGDA` / `GD` | 最小范数梯度组合，也称多目标最速下降 |
| `bbdmo` | `BBDMO` | Chen–Tang–Yang 的逐目标 BB 缩放，Eq. (22)、Algorithm 5；不是给 MGDA 统一乘一个 BB 步长 |
| `proximal_gradient` / `pg` | `ProximalGradient` | Tanabe–Fukuda–Yamashita 多目标近端梯度 |
| `tfy` | `TFYAcceleratedGradient` | TFY 加速近端算法中 `g_i=0` 的光滑特例 |
| `tfy_proximal` | `TFYAcceleratedProximalGradient` | TFY 加速近端梯度 Algorithm 2，保留目标差值项 |

算法出处及固定版本见[算法参考文献](references.md#first-order-algorithms)。
Nesterov 形式存在不同的多目标推广；本页的 NAG/APG 讨论均指保留的 TFY 版本。
AMG-QP 的递推、固定 L 和复合扩展在统一说明中单独给出。

## 安装与最小使用

```sh
python -m pip install -e ".[algorithms,plot]"
```

问题求值和前向导数只依赖 NumPy。双目标内层求解使用区间求根；
三目标及以上的基线额外使用 SciPy，放在可选的 `algorithms` 依赖中。

```python
from mop_problems import get_problem
from mop_problems.algorithms import MGDA, BBDMO, minimize, list_algorithms

p = get_problem("FON")
x0 = [0.2, -0.1, 0.3]

fixed = MGDA(step_rule="fixed", step_size=0.1, max_iter=500)
search = MGDA(step_rule="armijo", step_size=1.0, armijo=0.1, contraction=0.5)
r = search.solve(p, x0)
print(r.x, r.fun, r.status, r.stationarity_bound)
print(r.nfev, r.njev, r.elapsed_seconds)

bb = BBDMO(bb_min=1e-3, bb_max=1e3).solve(p, x0)
nag = minimize(p, x0, algorithm="tfy", step_rule="backtracking")
print(list_algorithms())
```

`x0` 必须是单个决策向量。配置对象可重复调用 `solve`，每次运行重新建立迭代状态。
算法不自动归一化目标或变量；目标缩放本身会影响多目标下降方向，应作为实验设置记录。

## MGDA 与 BBDMO

设 `J` 每一行是一个目标梯度。没有变量约束时，MGDA 解

\[
\min_{\lambda\ge0,\;\mathbf1^T\lambda=1}\frac12\|J^T\lambda\|^2,
\qquad d=-J^T\lambda,
\quad x^+=x+\alpha d.
\]

BBDMO 使用 `s=x_k-x_{k-1}`、`y_i=grad f_i(x_k)-grad f_i(x_{k-1})`，
将每行梯度除以正数 `a_i` 后求同一类最小范数组合：

\[
a_i=\begin{cases}
\operatorname{clip}(s^Ty_i/s^Ts,a_{min},a_{max})&s^Ty_i>0,\\
\operatorname{clip}(\|y_i\|/\|s\|,a_{min},a_{max})&s^Ty_i<0,\\
a_{min}&s^Ty_i=0.
\end{cases}
\]

第一步使用 MGDA；线搜索充分下降条件仍使用原始目标与原始梯度。
`step_rule` 可选：

- `fixed`：固定 `step_size`，不自动保证下降；步长需要与问题曲率匹配。
- `armijo`：所有目标同时满足 `F_new <= F_old + armijo*alpha*J*d`。
- `nonmonotone`：参考值改为各目标最近 `nonmonotone_window+1` 个值的最大值。
- `nonmonotone_average`：参考值按 `Q_new=eta*Q+1`、`C_new=(eta*Q*C+F_new)/Q_new` 更新。

默认 `step_rule="auto"` 对 MGDA/BBDMO 选择 Armijo。每轮从 `step_size` 开始，
乘 `contraction` 回溯。历史中的 `step_size` 是实际接受的步长。
为控制浮点比较误差，充分下降判断允许约 16 个机器精度乘目标量级的松弛。

## 近端问题与加速问题

近端算法求解 `F_i=f_i+g_i`，其中 `f_i` 由问题库提供值与梯度，
`g_i` 是显式提供的闭凸可近端项。一步模型为

\[
\min_{z\in C}\max_i\{\nabla f_i(x)^T(z-x)+g_i(z)-g_i(x)\}
+\frac{\|z-x\|^2}{2t}.
\]

加速版本在外推点 `y` 计算梯度，模型中的第 i 项改为

\[
\nabla f_i(y)^T(z-y)+g_i(z)+f_i(y)-F_i(x).
\]

这里 `x` 是上一个接受点，`y` 是当前外推点。不能省掉 `f_i(y)-F_i(x)`。
动量从 `q=1` 开始，按 `q_new=(1+sqrt(1+4*q*q))/2` 更新，
下一外推点为 `x_new + (q-1)/q_new*(x_new-x)`。

PG/NAG/APG 支持 `fixed` 与 `backtracking`。后者逐步减小 `t`，直到各光滑目标满足
`f_i(z) <= f_i(y)+grad f_i(y)·(z-y)+||z-y||²/(2*t)`。
接受后的步长保留到下一轮，确保逆步长不减少；默认初始 `t=1`。
加速算法不要求每轮每个目标都单调下降。
文献的凸性、Lipschitz 连续梯度等前提不自动适用于 ZDT/WFG 等非凸测试问题，
因此本库不将文献中的收敛速率声明为所有基准上的保证。

内置 `L1` 支持不同目标的权重与中心：

```python
from mop_problems import get_problem
from mop_problems.algorithms import L1, ProximalGradient, TFYAcceleratedProximalGradient

p = get_problem("FON", n_var=3)
reg = L1(coefficients=[0.1, 0.2], centres=[0.0, 0.5])
pg = ProximalGradient(regularizer=reg)
apg = TFYAcceleratedProximalGradient(regularizer=reg)
r = apg.solve(p, [0.2, -0.1, 0.3])
print(r.fun)  # f_i(x) + g_i(x)
print(r.smooth_fun)  # 原问题 f_i(x)
```

`g_i=sum_j coefficients[i,j]*abs(x_j-centres[i,j])`。
标量广播到所有位置，一维数组按目标广播，二维数组提供每个目标/变量的参数。
添加 L1 会改变优化问题及其 Pareto 前沿；不能拿原始问题前沿作为复合问题真值。
不提供正则项时 `g_i=0`，PG 与 MGDA 可能产生相同步骤，NAG 与 APG 也应一致。

自定义近端项需提供 `values(x) -> (n_obj,)`，以及
`prox(v, step, weights, lower, upper) -> (n_var,)`，后者必须精确求解
`argmin_{z in box} sum_i weights_i*g_i(z)+||z-v||²/(2*step)`。
仅对每个目标分别调用 prox 后求平均通常不正确。
可选 `bind(n_obj,n_var)` 用于参数展开。若用户的 prox 不是凸模型的正确解，
内层对偶间隙的证明也不成立；此合同需由自定义实现负责。

## 边界、一般约束与不可微点

默认 `bounds="box"` 将变量盒 **放在方向/近端子问题内部**，属于投影或约束近端扩展。
MGDA/BBDMO 中初始步长为 `a` 时，模型是
`min_{z in box} max_i J_scaled_i·(z-x)+||z-x||²/(2*a)`，
线搜索沿 `z-x` 缩短步长。非活动边界时与普通 MGDA/BBDMO 相同；
活动边界时不能先求无约束权重再简单裁剪来替代这个模型。

`bounds="none"` 显式选择忽略变量盒，使用公式的无约束延拓；
仅当公式在访问位置有定义时有效，此时 `result.feasible` 仍按原始盒判定。
默认 `domain_restart=True` 在外推点离开盒时重启动量，防止查询分数幂等公式的非法区域。
可设为 `False` 允许外推点越界求值，同时接受点仍在近端盒内；需确认公式可延拓。
若惯性恰好抵消近端位移而普通残差仍非零，也会重启动量；重启次数单独记录。
这些有限域保护措施是明确的实现选择，不宣称它们拥有未经证明的同一理论速率。

五类基线直接支持变量盒和显式近端凸项，**不直接求解带非线性 `G/H` 的问题**。
对 BNH、TNK、C-DTLZ、Welded-Beam 等传入 `.solve()` 会明确报错；
这些问题可使用新增的 `AugmentedLagrangian`，或通过 `dF/dG/dH` 接入其他约束算法。
算法不会忽略约束、偷偷加入罚项，或在梯度奇异时改用差分继续运行。

## 统一停止、精度和成本

本页五类基线在接受点使用同一个未进行 BB 缩放的单位步长近端模型 `p_1(x)`，
报告 `stationarity=||p_1(x)-x||_inf`。
内层解有对偶间隙 `gap` 时，停止依据保守上界
`stationarity_bound = stationarity + sqrt(2*gap)`，不是实际步长的大小。
因此很小的回溯步长或很大的 BB 缩放不会自动被当成收敛。
`success=True` 只代表该一阶条件通过，不代表全局 Pareto 最优；
在非凸问题、退化目标或边界处，驻点仍可能被其他可行点支配。

内层双目标使用有界对偶求根；多目标使用 SLSQP，必要时用局部曲率与可行对偶方向改进。
接受标准为原始-对偶间隙 `gap <= subproblem_tol*max(1,abs(primal),abs(dual))`，
不会仅凭 SciPy 的 success 标志判定内层成功。
近端映射、内层计算、残差诊断及失败的线搜索试探均计入成本。

| 默认选项 | 值 |
| --- | --- |
| `step_size`, `step_rule` | `1.0`, `auto` |
| `max_iter`, `tol` | `500`, `1e-5` |
| `max_nfev`, `max_njev` | `10000`, `2000` |
| `armijo`, `contraction` | `0.1`, `0.5` |
| `max_line_search`, `min_step` | `60`, `1e-16` |
| `nonmonotone_window`, `average_eta` | `10`, `0.8` |
| `bb_min`, `bb_max` | `1e-3`, `1e3` |
| `subproblem_tol`, `subproblem_max_iter` | `1e-12`, `500` |
| `bounds`, `domain_restart`, `store_history` | `box`, `True`, `True` |

`subproblem_max_iter` 分别限制初始 SLSQP 和必要的改进循环；不是两个阶段相加的总限制。
过小的 `tol` 还需要相匹配的内层精度，否则可能达到预算但不能认证驻点。

| 结果 | 定义 |
| --- | --- |
| `x`, `fun`, `smooth_fun` | 最后接受点、完整目标、光滑目标；只读数组 |
| `status`, `message`, `success` | 驻点、迭代/求值预算、不可微、内层失败或线搜索失败等，均明确区分 |
| `n_iter` | 已接受更新次数 |
| `nfev` | 全部公式遍历次数，包含普通求值及联合值/Jacobian 遍历 |
| `njev` | 联合值/Jacobian 遍历次数，是 `nfev` 的子集，不能再次相加 |
| `nprox`, `nsubproblem` | 近端调用及子问题尝试次数，包含失败尝试 |
| `subproblem_iterations` | 已完成内层求解的迭代数之和；失败求解的工作反映在 `nprox` 中 |
| `elapsed_seconds` | 整个运行的墙钟时间，含诊断与内层求解；依赖机器与环境 |
| `history` | 每个接受点的 x/F/残差/计数/上一步步长/上一步 BB 缩放 |

若预算在最后接受点的梯度或残差检查前耗尽，对应残差为 `None`，不会返回上一个点的残差。
已知不是公共下降方向时报告 `no_descent`，不会用非下降步伪装成标准 MGDA。

## 多起点与对比实验

```python
from mop_problems import get_problem
from mop_problems.algorithms import MGDA, BBDMO, multistart

p = get_problem("DTLZ2", n_var=8, n_obj=3)
starts = p.sample(30, seed=42)
gd_runs = multistart(p, MGDA(max_iter=500), starts=starts)
bb_runs = multistart(p, BBDMO(max_iter=500), starts=starts)
print(sum(r.nfev for r in gd_runs), sum(r.nfev for r in bb_runs))
```

同一 `starts` 可交给你的改进算法。预算按每个起点生效；与进化算法的最终解集对比时，
要汇总所有起点的成本，同时报告梯度成本、总时间、可行率、驻点率和解集质量。
对比 HV/IGD 等指标还需统一目标缩放、参考集与参考点。有限起点不保证覆盖整个前沿。

```sh
python examples/gradient_comparison.py --problem JOS1 --starts 12 --seed 42
python examples/gradient_comparison.py --problem FON --starts 12 --seed 42 --output outputs/gradient_fon
python examples/gradient_comparison.py --problem DTLZ2 --starts 12 --seed 42 --output outputs/gradient_dtlz2
```

默认 JOS1 为示例文件中明确定义的凸二次模型，不增加注册问题数。
FON 的初始点限制在 `[-1,1]^n`，避免远处指数饱和；DTLZ2 使用完整盒。
每次同时生成光滑比较和不同 L1 项的复合比较，固定步长 MGDA 使用 `0.1`。
固定步长方法可能在默认预算内达不到停止精度；这种情况保留为 `max_iter`。

输出包括共享初始点和终点的 NPZ、逐次运行及历史 JSON、汇总 CSV、
环境/参数/源码哈希 manifest，以及目标空间、残差和计算成本图。
曲线展示共同起点 #0；柱状图汇总全部起点。未收敛与失败结果保留在导出中。
`composite_l1` 使用改变后的目标，不叠加原始 Pareto 前沿。
添加 `--no-plots` 可关闭绘图。已记录的运行见[梯度实验记录](gradient-validation.md)。
