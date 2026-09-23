# MOP-Problems：多目标优化问题库

[English](README.md) · [完整分类](problems/README.md) · [调用接口](docs/api.md) ·
[数学公式](docs/formulas.md) · [可复现实验](docs/experiments.md) · [GitHub 上传说明](docs/publishing.md)

[梯度与光滑性](docs/gradients.md) · [内置算法与对比指南](docs/algorithms.md)

这个仓库采用 Python + NumPy，统一管理问题定义、变量边界、约束、参考文献和测试。
0.3.0 版保留清单中的 **43 个问题族或独立问题条目**，其中 **15 个条目下的 47 个具体问题已经实现**。
其余条目明确标为 `planned`，有来源和待核验说明，不代表已经可以运行。
全部 47 个问题已提供一阶导数接口，并内置五类梯度/近端算法作为研究基线。

## 安装与调用

需要 Python 3.10 或更高版本。在仓库根目录执行：

```sh
python -m pip install -e .
```

代码托管于 [GitHub](https://github.com/Fanzx42/MOP-Problems)，安装方式为从源码安装，尚未发布到 PyPI。

```python
from mop_problems import get_problem, list_problems, list_suites

problem = get_problem("ZDT1", n_var=30)
X = problem.sample(100, seed=42)
result = problem.evaluate(X)
print(result.F)  # 目标矩阵 (100, 2)
print(result.feasible)  # 可行性 (100,)

problem = get_problem("BNH")
result = problem.evaluate([1.0, 1.0])
print(result.G)  # [[-8.0, -57.3]]，全部 <= 0，所以可行

problem = get_problem("DTLZ2", n_obj=5)
print(problem.n_var)  # 14；按 n_obj + k - 1 确定默认维度
print(list_problems())  # 仅返回可运行问题
print(list_suites(status="planned"))  # 查询待实现的问题族
```

## 当前实现范围

| 分类 | 已实现问题 |
| --- | --- |
| ZDT | ZDT1、ZDT2、ZDT3、ZDT4、ZDT6 |
| DTLZ | DTLZ1–DTLZ7 |
| 无约束经典问题 | SCH、FON、POL |
| 约束经典问题 | BNH、TNK、OSY、SRN |
| 变换与非可分问题 | WFG1–WFG9 |
| 变量耦合问题 | UF1–UF10 |
| 多个等价最优解集 | Omni-test、SYM-PART、SYM-PART-Rotated |
| 基于 DTLZ 的约束问题 | C1-DTLZ1、C1-DTLZ3、C2-DTLZ2、C3-DTLZ1、C3-DTLZ4 |
| 工程设计问题 | Welded-Beam（明确采用 pymoo 0.6.2 的双目标、四约束版本） |

## 梯度算法与统一对比

```sh
python -m pip install -e ".[algorithms,plot]"
python examples/gradient_comparison.py --starts 12 --seed 42
```

```python
from mop_problems import get_problem
from mop_problems.algorithms import MGDA, BBDMO, multistart

p = get_problem("FON")
x0 = [0.2, -0.1, 0.3]
value, jac = p.evaluate_with_jacobian(x0)
print(jac.dF.shape)  # (1, 2, 3)，每行是一个目标的梯度

gd = MGDA(step_rule="armijo", max_iter=500)
fixed = MGDA(step_rule="fixed", step_size=0.1)
result = gd.solve(p, x0)
print(result.status, result.nfev, result.njev, result.stationarity_bound)

starts = [[0.2, -0.1, 0.3], [-0.4, 0.1, 0.2]]
gd_runs = multistart(p, gd, starts=starts)
bb_runs = multistart(p, BBDMO(), starts=starts)
```

提供 MGDA/GD、BBDMO、多目标 proximal gradient、Nesterov 加速梯度、
Nesterov 加速 proximal gradient。加速方法明确采用 TFY 多目标版本。
GD 支持固定步长、Armijo、最大值非单调和平均值非单调线搜索；
近端方法支持固定步长/模型回溯，以及各目标不同的 L1 权重和中心。

导数采用前向自动微分，不默认使用有限差分。不可微点、奇异点、内层求解失败及预算耗尽
均有明确状态。输出包含轨迹、函数/导数调用次数、耗时与统一驻点指标。
默认把变量盒放入子问题；一般非线性约束算法需使用 `dG/dH` 自行接入。
停止条件通过不等于全局 Pareto 最优。
详见[完整调用与配置](docs/algorithms.md)、[导数核验及光滑性](docs/gradients.md)。

![共享起点的梯度算法对比](docs/assets/gradient_comparison.png)

所有目标默认最小化；不等式返回 `G ≤ 0`，等式返回 `H = 0`。
单个解也返回二维目标矩阵，无约束时 `G`、`H` 的形状为 `(N, 0)`。
接口校验维度、非有限数值和变量边界，不会自动裁剪输入或把约束混入目标值。
`CV` 是原始约束违反量的和，因此不能直接比较不同问题的 `CV` 大小。
随机采样只保证满足变量边界，不保证满足一般约束。

## 代表性计算与可视化实验

```sh
python -m pip install -e ".[plot]"
python examples/representative_suite.py --samples 2000 --seed 42
```

一次运行生成 **14 组固定种子采样结果、三张图、CSV 原始数据和 JSON 复现记录**，
涵盖 WFG 变换、UF 耦合和断裂结构、约束可行性，以及 5 目标和 8 目标问题。
另有解析构造展示 Omni-test 和两种 SYM-PART 各自的九个等价 Pareto 解集。
输出保存在 `outputs/representative/`，添加 `--no-plots` 可只生成数值数据。

详见[实验说明、复现步骤及已运行结果](docs/experiments.md)。图中明确区分采样非支配点、
解析前沿和不可行点；这个脚本仍是问题特征实验。
优化算法对比使用上面的 `gradient_comparison.py`，两类脚本各有明确的实验用途。

![多个等价 Pareto 解集及其共同前沿](docs/assets/multimodal_sets.png)

## 分类和来源约定

- `problems/` 完整保留你给的三级分类，可以直接在 GitHub 中浏览。
- 实际代码位于 `src/mop_problems/`；目录卡片由同一份 JSON 元数据生成，避免状态不一致。
- `unconstrained_smooth` 是分类名称，不保证每个问题处处可微。
- ZDT5 是二进制问题，本仓库没有把它混入连续 ZDT；DTLZ8/9 也没有混入无约束 DTLZ。
- `RE_continuous`、`RCM_continuous` 的连续成员要逐个核验，不会静默做连续松弛或取整。
- SYM-PART 已实现普通版和旋转版；扭曲版没有混入同一名称。MMF 仍待实现。
- 等式和动态问题的目录已经建立；当前尚未提供这些问题的公式实现。

来源包括原论文、作者提供的材料及参考实现。发现来源不等于完成数值验证。
每个问题从 `planned` 变为 `implemented` 前，都需要公式、边界、约束和独立测试。

## 开发与检查

```sh
python -m pip install -e ".[dev,reference,algorithms]"
python -m pytest --cov=mop_problems --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python scripts/build_catalog.py --check
python -m build
```

已经配置 GitHub Actions、Issue 表单、PR 模板、MIT 许可证和 `CITATION.cff`。
后续同步按[发布说明](docs/publishing.md)中的实际仓库地址执行。
论文实验请同时引用原始问题文献；本仓库名称不能代替原始基准问题的出处。
