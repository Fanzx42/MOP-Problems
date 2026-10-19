# MOP-Problems：多目标优化问题库

[English](README.md) · [完整分类](problems/README.md) · [调用接口](docs/api.md) ·
[数学公式](docs/formulas.md) · [可复现实验](docs/experiments.md) · [GitHub 上传说明](docs/publishing.md)

[梯度与光滑性](docs/gradients.md) · [内置算法与对比指南](docs/algorithms.md)

[有实际背景的多目标 ADMM 应用专题](docs/admm-applications.zh-CN.md)：新增 12 个文献/模型条目，
逐项列出目标分离、耦合等式、凸性条件、数据可用性及来源核验程度；另附四个小型分离结构数值检查。
这些条目属于研究目录，未计入可调用基准数量。

[Gamma Knife 四目标可运行例子](docs/radiosurgery.md)：提供合成剂量矩阵、
原文标量化 ADMM、多目标 ADMM/ALM、数值缓存优化、自适应罚参数及原始测试证据。
安装可选依赖 `.[radiosurgery]` 并配置 COPT 许可证即可运行；不改变 343 个基准问题的计数。

这个仓库采用 Python + NumPy，统一管理问题定义、变量边界、约束、参考文献和测试。
0.4.0 版覆盖清单中全部 **43 个问题族或独立问题条目，共 343 个可调用问题**。
原先 28 个 `planned` 问题族新增了 296 个实现，当前 `planned` 列表为空。
每个问题都有目标/约束 Jacobian、光滑性说明、原始论文和参考实现版本。
新增问题全部具有独立参考数值测试；库仍处于研究用 alpha 阶段。

[问题及论文目录](problems/README.md) · [文献索引](docs/problem-references.md) ·
[新增问题调用示例](docs/expanded-benchmarks.md) · [核验与版本差异](docs/expanded-validation.md)

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
print(list_suites(status="planned"))  # 当前返回 []
```

## 当前实现范围

| Family | Callable members |
| --- | --- |
| ZDT / DTLZ | ZDT1/2/3/4/6; DTLZ1–7 |
| Classics | SCH, FON, POL; BNH, TNK, OSY, SRN |
| Transformations | WFG1–9; IDTLZ1/2; SDTLZ1/2 |
| Linkage | UF1–10; LZ09_F1–9 |
| Irregular / many-objective | IMOP1–8; MaF1–15; ZCAT1–20 |
| Large-scale | LSMOP1–9; LSCMOP1–12 (author names LSCM1–12) |
| Multimodal | MMF1–8; Omni-test; SYM-PART / SYM-PART-Rotated; CMMOP1–14 |
| Dynamic | FDA1–5; JY1–8, explicit fixed-time snapshots |
| Classical constraints | CTP1–8; CF1–10 |
| DTLZ constraints | Five C-DTLZ variants; six DC-DTLZ variants; Eq-DTLZ1–4; Eq-IDTLZ1/2 |
| Difficult constraints | MW1–14; LIRCMOP1–14; DASCMOP1–9; DASCMaOP1–9; DOC1–9; NCTP1–18; ZXH_CF1–16 |
| Engineering | 10 RE; 10 RWA; 6 CRE; Welded-Beam; 39 RCM continuous definitions, including two explicitly labelled disk-count relaxations |

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
接口校验维度、非有限数值和变量边界。RE 保留文献定义的违反量目标；
LSCMOP 保留作者聚合违反量约束。MaF9 采样还会排除原定义中的无效多边形。
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

新增问题的计算、梯度、动态快照和等式可行点示例：

```sh
python examples/expanded_benchmarks.py --samples 1000 --seed 42
```

生成 CSV、含论文/版本的 JSON 清单以及三张图。使用 `--no-plots` 可仅依赖 NumPy。
完整说明见[新增基准实验](docs/expanded-benchmarks.md)。

## 分类和来源约定

- `problems/` 完整保留你给的三级分类，可以直接在 GitHub 中浏览。
- 实际代码位于 `src/mop_problems/`；目录卡片由同一份 JSON 元数据生成，避免状态不一致。
- `unconstrained_smooth` 是分类名称，不保证每个问题处处可微。
- ZDT5 是二进制问题，本仓库没有把它混入连续 ZDT；DTLZ8/9 也没有混入无约束 DTLZ。
- `RE_continuous` 包含 10 个确定性连续问题；`RCM_continuous` 包含 39 个来源代码定义，RCM5/14 的盘片数量连续松弛已明确标注。
- SYM-PART 已实现普通版和旋转版；扭曲版没有混入同一名称。MMF1–8 已实现。
- Eq-DTLZ/IDTLZ 使用真正的等式残差 `H`；FDA/JY 使用显式时间，不随求值次数变化。

来源包括原论文、作者提供的材料及参考实现，均列在每个问题的卡片中。
可通过 `get_problem_info("NCTP1")["references"]` 直接导出论文信息。
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
