# 梯度与 Jacobian / First derivatives

0.4.0 为全部 **343 个已实现问题**提供目标与原始约束的一阶导数。
采用前向自动微分：在原有公式上逐运算传播链式法则，使用 NumPy 双精度计算，
不以有限差分作为默认梯度，也不依赖 PyTorch、JAX 或 SciPy。
公式参数、目标方向、约束符号和变量边界保持原有定义。

## 统一调用

```python
from mop_problems import get_problem, check_derivatives

p = get_problem("BNH")
X = [[1.0, 1.0], [2.0, 1.0]]
ev, jac = p.evaluate_with_jacobian(X)
print(ev.F)  # (2, 2)
print(jac.dF)  # (2, 2, 2): batch, objective, variable
print(jac.dG)  # (2, 2, 2): batch, inequality, variable
print(jac.dH)  # (2, 0, 2): no equality constraints
print(p.derivative_info)

# 已有自己的算法时，只取单个点的目标向量和 Jacobian：
x = [1.0, 1.0]
F = p.evaluate(x).F[0]
J = p.jacobian(x).dF[0]  # 每一行是一项目标的梯度
report = check_derivatives(p, X)
assert report.passed, report
```

| 接口或字段 | 含义 |
| --- | --- |
| `p.jacobian(X)` | 返回 `Jacobian`，单点也保留批次维度 |
| `p.evaluate_with_jacobian(X)` | 一次公式遍历返回 `(Evaluation, Jacobian)` |
| `dF` | `(N, n_obj, n_var)` |
| `dG`, `dH` | `(N, n_ieq_constr, n_var)`、`(N, n_eq_constr, n_var)` |
| `method` | 默认 `forward`；显式差分为 `finite_difference` |
| `regular` | `(N,)`，此次运算是否通过分支规则检查；差分返回 `None` |
| `check_bounds=True` | 与目标求值相同，默认拒绝超出变量边界的输入 |
| `nondifferentiable="raise"` | 默认拒绝检测到的不可微运算；可显式选择 `"branch"` |

导数数组和 `regular` 数组均为只读副本。支持空批次。`dG` 是原始 `G <= 0`
残差的导数，`dH` 是 `H = 0` 的导数；不会对约束归一化或改变符号。
变量上下界没有附加进 `dG`。若外部算法需要其导数，分别使用 `-I` 和 `I`。
不对 `CV` 这个含正部运算的汇总量提供 Jacobian。
LSCMOP 是例外的来源约定：其 `G` 本身就是作者定义的聚合违反量，
可行分支内梯度为零，不能当作原始约束梯度使用。
FDA/JY 返回固定时间下对 `x` 的导数，不计算时间导数。
详见[新增问题的约定与核验](expanded-validation.md)。

## 光滑性与适用范围

每个问题的 `derivative_info` 与默认参数目录元数据记录方法、光滑性及注意事项。
实例信息反映当前参数，例如 `DTLZ4(alpha=0.5)` 与默认 `alpha=100` 不同。

| 类型 | 例子与注意事项 |
| --- | --- |
| `smooth` | SCH、FON、POL、ZDT2、DTLZ1–5/7（DTLZ4 的小 alpha 除外）、UF8/10、Omni-test；光滑不等于凸 |
| `interior` | ZDT1/3/4、UF1/2/3/7 的零位置变量；ZDT6/DTLZ6 的零距离变量可能导致分数幂导数奇异 |
| `piecewise` | WFG、UF4/5/6/9 的变换或绝对值/max；C2-DTLZ2 的约束 min/max |
| `piecewise_discontinuous` | SYM-PART 的目标在各块内为二次函数，跨块可能跳跃 |
| 约束单独考虑 | TNK 的目标为线性，约束在原点角度未定义；C2-DTLZ2 的目标光滑，约束分段 |

`regular=True` 仅说明此次链式计算没有遇到已检测的异常，并不是光滑性的数学证明。
`abs` 的零点、`min/max` 的不同斜率交点以及相关整数分支会触发检查。
检查在约 32 个机器精度的邻域内保守执行；复杂变换中即使外层最终抵消了折点，也可能被拒绝。
WFG 的强偏置还可能将不同中间量压缩成双精度下的同一个值。
SYM-PART 对实际的块边界单独检查，中心块原点及已被截断的外层整数点不会误报。

```python
from mop_problems import NonDifferentiableError

p = get_problem("SYM-PART")
try:
    p.jacobian([6.0, 1.0])
except NonDifferentiableError:
    pass

branch = p.jacobian([6.0, 1.0], nondifferentiable="branch")
assert not branch.regular[0]
```

`branch` 返回当前选中分支的导数，**不保证是原函数的经典导数或有效次梯度**。
NaN/无穷导数始终报错，不会被替换为零或通过截断隐藏。
内置算法使用严格模式。非光滑的原始 WFG/UF 函数不会因为选择 PG 就自动变成
“光滑项 + 已知凸近端项”；PG 的分解需要显式提供。

## 独立数值核验

```python
from mop_problems import finite_difference, check_derivatives

p = get_problem("DTLZ2", n_obj=3, n_var=8)
X = p.sample(4, seed=238)
approx = finite_difference(p, X, relative_step=1e-6)
report = check_derivatives(p, X, rtol=1e-4, atol=1e-6, relative_step=1e-6)
print(report.max_absolute_error, report.max_scaled_error)
```

差分在内部点使用中心三点公式，在边界附近使用二阶单侧三点公式，扰动留在变量盒内。
默认相对步长为机器精度的立方根，尺度为 `min(max(1, abs(x_j)), xu_j-xl_j)`，
再限制到盒宽的四分之一；微量级工程变量不会受到过大的差分扰动。
每个点的成本为 `1 + 2*n_var` 次向量函数求值，包含共享的中心点。
误差使用 `abs(exact-approx)/(atol + rtol*max(abs(exact),abs(approx)))`，全部不超过 1 才通过。
差分无法证明可微性，也不会作为内置算法失败时的隐式备用梯度。

强振荡、大数抵消及边界奇异会影响核验。例如 C1-DTLZ3 的约束可能达到 `1e12`
量级，而某些位置方向的理论导数为零；有限差分的相减误差可能远大于真实导数。
应同时使用良态测试点、步长敏感性分析与解析恒等式，不能仅调大容差后宣称正确。
当前测试包括所有问题的独立差分、不同维数、手工解析导数、约束符号、
乘积含零因子的情形、奇异点和分支边界。

## 接入自己的算法与扩展

一阶算法可直接使用 `evaluate_with_jacobian` 返回的 `F[0]`、`dF[0]`、`dG[0]`、`dH[0]`。
这套 NumPy 接口不建立 PyTorch/JAX 计算图，也不提供 Hessian。
自定义 `Problem._evaluate` 若仅用内部支持的 NumPy 运算，可以自动传播导数；
不支持的运算会明确失败，不会静默丢失梯度。复杂自定义问题可覆写
`evaluate_with_jacobian` 返回同样的两个数据对象，或在外部提供自己的导数适配器。

内置实现使用的运算包括切片、加减乘除、幂、sin/cos、exp/expm1、log、sqrt、atan2、
sum/mean/prod/max、列拼接、roll、常数矩阵乘法及部分分段运算。
它是供本库公式使用的有限运算集合，不是通用自动微分框架。
前向模式同时传播所有变量方向，计算量随变量数增加；输入种子的内存为 `O(N*n_var²)`。
大变量数时应减小批次，并单独报告导数耗时，不能把一次 Jacobian 调用视为一次普通目标求值的等价成本。
算法调用、预算和文献版本详见[算法指南](algorithms.md)。
