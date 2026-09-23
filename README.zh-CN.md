# MOP-Problems：多目标优化问题库

[English](README.md) · [完整分类](problems/README.md) · [调用接口](docs/api.md) ·
[数学公式](docs/formulas.md) · [GitHub 上传说明](docs/publishing.md)

这个仓库采用 Python + NumPy，统一管理问题定义、变量边界、约束、参考文献和测试。
首版保留清单中的 **43 个问题族或独立问题条目**，其中 **19 个具体问题已经实现**。
其余条目明确标为 `planned`，有来源和待核验说明，不代表已经可以运行。

## 安装与调用

需要 Python 3.10 或更高版本。在仓库根目录执行：

```sh
python -m pip install -e .
```

当前只提供本地安装，不宣称已在 PyPI 或 GitHub 发布。

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

## 首版实现范围

| 分类 | 已实现问题 |
| --- | --- |
| ZDT | ZDT1、ZDT2、ZDT3、ZDT4、ZDT6 |
| DTLZ | DTLZ1–DTLZ7 |
| 无约束经典问题 | SCH、FON、POL |
| 约束经典问题 | BNH、TNK、OSY、SRN |

所有目标默认最小化；不等式返回 `G ≤ 0`，等式返回 `H = 0`。
单个解也返回二维目标矩阵，无约束时 `G`、`H` 的形状为 `(N, 0)`。
接口校验维度、非有限数值和变量边界，不会自动裁剪输入或把约束混入目标值。
`CV` 是原始约束违反量的和，因此不能直接比较不同问题的 `CV` 大小。
随机采样只保证满足变量边界，不保证满足一般约束。

## 分类和来源约定

- `problems/` 完整保留你给的三级分类，可以直接在 GitHub 中浏览。
- 实际代码位于 `src/mop_problems/`；目录卡片由同一份 JSON 元数据生成，避免状态不一致。
- `unconstrained_smooth` 是分类名称，不保证每个问题处处可微。
- ZDT5 是二进制问题，本仓库没有把它混入连续 ZDT；DTLZ8/9 也没有混入无约束 DTLZ。
- `RE_continuous`、`RCM_continuous` 的连续成员要逐个核验，不会静默做连续松弛或取整。
- MMF、SYM-PART 等有多种版本；问题卡片记录首版目标与尚待核对的变体。
- 等式和动态问题的目录已经建立；首版尚未提供这些问题的公式实现。

来源包括原论文、作者提供的材料及参考实现。发现来源不等于完成数值验证。
每个问题从 `planned` 变为 `implemented` 前，都需要公式、边界、约束和独立测试。

## 开发与检查

```sh
python -m pip install -e ".[dev,reference]"
python -m pytest --cov=mop_problems --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python scripts/build_catalog.py --check
python -m build
```

已经配置 GitHub Actions、Issue 表单、PR 模板、MIT 许可证和 `CITATION.cff`。
实际上传前，按[发布说明](docs/publishing.md)填写你自己的 GitHub 仓库地址。
论文实验请同时引用原始问题文献；本仓库名称不能代替原始基准问题的出处。
