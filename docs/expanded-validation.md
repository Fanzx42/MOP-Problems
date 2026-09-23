# 0.4.0 扩展核验与版本约定

本次为原先 28 个 planned 问题族增加 296 个实现；总计 343 个问题、43 个目录条目。
每个新增问题均有独立参考数值，保存在
[expanded_reference.json](../tests/data/expanded_reference.json) 的 **482 组案例**中。
每组包含输入、构造参数、目标/约束参考值、来源类别和约束比较方式。
这些参考值不由本库实现生成；测试运行时无需 MATLAB、网络或参考平台。

## 参考计算的来源

核验使用 Windows、Python 3.14、MATLAB R2023b 和 pymoo 0.6.2。
参考文件的版本在每个问题卡片中列出，作者下载包的 SHA-256 另存在测试数据文件顶层。

本地完整测试 **3269 项通过**，分支覆盖率统计合计 **97%**。
`expanded_benchmarks.py --samples 1000 --seed 42` 已生成并检查 14 组数据与三张图。
Ruff、396 页目录生成一致性、sdist/wheel 元数据检查均已通过；独立临时环境仅安装
NumPy 和 wheel，完成全部 343 个问题的求值/Jacobian 及双目标 MGDA 检查。

| 来源 | 实际核验方法 |
| --- | --- |
| PlatEMO `d25e65d1ffba58dbf4d7e1b5259786187d12968a` | 在 MATLAB 中执行原 `CalObj/CalCon`，用最小 `PROBLEM` 容器承接属性；MaF8 的欧式距离提供等价 `pdist2` 辅助函数 |
| jMetalPy `fe3cd718748bea9ae59404fb70bf0e0bfe4167c9` | 原 LZ09、ZCAT、RWA Python 方程运行于最小 solution/problem 容器；ZCAT 另检查 1–6 级、bias、imbalance、复杂 Pareto 集组合 |
| Tanabe RE/CRE `7876b4e465eac381a256e461d1310b8bb2b92846` | 直接运行原 Python 公式；CRE 比较正部违反量 |
| Eq-DTLZ/IDTLZ 作者补充材料 | MATLAB 执行一条/两条等式版本，比较 `abs(H)-1e-4`；本库另检验有符号解析等式 Jacobian |
| CMMOP、LSCM、CEC2021 RCM 作者包 | MATLAB 原公式输出；LSCM 包括 100 和 1000 维；RCM 逐行调用以避开原 RCM50 批处理缺陷 |
| NCTP 作者 2016 压缩包 | MATLAB 执行三类六个成员；仅在输出处暴露正部截断和群体归一化前的原残差 |
| JY 作者 October 2016 `JY.h` | 将 C++ 数学表达式和循环机械转为 C# 执行，保留映射；JY8 仅修复缺失 `pow`。检查 0、0.3、0.7、1.1 四个时间；这不是原生 C++ 编译核验 |
| pymoo 0.6.2 | MW1–14、CTP1–8、DASCMOP1–9、连续 MaF10；CTP 消去参考输出的缩放后比较原残差 |
| DASCMaOP 原论文 Eq. (8) | 独立标量约束公式，加 pymoo WFG 目标与中间变换，检查 3/5 目标下全部九个问题；不是作者软件结果复现 |

参考数组容差为 `rtol=5e-10, atol=2e-8`，适用于这里保存的双精度样例。
约束比较明确区分 raw、positive violation 和 equality band；不会将符号差异当作误差忽略。
DOC 部分来源输出已截断，因此对应冻结案例仅比较正部；其原始符号由公式与其他约束测试检查。
MaF10 参考 WFG1 的 `k=M-1`，该数学分组有效；测试绕开了 pymoo 构造器过严的最小 `k` 限制。
下载的外部代码不随本仓库分发，参考数值也不代表完整论文实验或最优前沿证明。

## 影响复现的版本差异

| 问题 | 本库明确采用的约定 |
| --- | --- |
| MaF10 | 连续 WFG1；不采用某个 PlatEMO 版本中 `b_flat` 后的四位小数取整。该差异也关系到导数能否用于研究 |
| MaF9 | 原无效多边形是定义域排除区；求值时明确拒绝，采样时可复现地重采样，不随机修复算法提交的输入 |
| ZCAT | 采用 jMetalPy 的 `theta` 分母 `n-p`，不与 PlatEMO 的 `n` 混用 |
| FDA2 / JY3 / JY8 | FDA2 用修正指数；JY3 用原作者 header；JY8 恢复缺失幂函数，与作者参与的 RJY8 定义交叉核对 |
| MW | 使用 pymoo 0.6.2；MW4/6/11/12/13/14 的边界/公式与 PlatEMO 存在差异 |
| DC-DTLZ3 | 来源代码的多峰距离系数为 10 |
| DASCMOP1–3 | pymoo 的 tail-only 距离，区别于 PlatEMO all-variable 和式；停用难度项保持非正值 |
| DASCMaOP | 使用论文 Eq. (8)，默认 `k=2*(M-1)`；论文部分实验写的 `k=10` 不总满足 WFG 分组整除条件，不宣称复现那组参数 |
| LIRCMOP | 保留来源 LIRCMOP2–4 尾变量平移、13/14 半径偏移 |
| Eq-DTLZ/IDTLZ | `H` 是真正等式；原包 `abs(H)-1e-4` 是容差带。DOC 的既定等式带则仍作为 `G` |
| NCTP | I/II/III 各六个成员，对应编号 1–18；所有变量边界 `[0,5]`；去掉算法相关的群体归一化和正部截断 |
| LSCMOP | 对应作者 LSCM1–12；保留聚合违反量、阈值和余数耦合。可行分支 `dG=0`，不能替代原始单个约束的 Jacobian |
| CRE / RE | CRE 暴露有符号残差；RE 保留作者定义的违反量目标。RE21 的 `sqrt(x3)` 体积项保留来源版本，与 RCM9 的版本不同 |
| RWA | 按实际 evaluator 的负号将最大化转为最小化；不用与方程不一致的方向标签推断符号 |
| RCM | 保留来源编号、工程常数和有效约束列；RCM9 无有效约束，RCM10 删除恒零 H 占位，RCM24 返回实际三目标 |
| RCM5/14 | 盘片数量采用来源的连续松弛，已在元数据标注；不是整数设计问题 |
| RCM50 | 原 `P(j)` 线性索引导致批次相关损耗；改成逐行二次损耗，匹配原始逐点计算 |

RCM2/4/8/14/17/40/41 的来源公式含容易与其他工程版本混淆的项，均记录在各自卡片。
本库不以“物理上更合理”为由静默替换已选定的基准定义。
RCM20 的相同内外半径、RCM47–49 的零电压、CF8–10 的零约束分母均可能导致未定义值；
接口会报错，不截断分母或伪造有限值。

## 导数证据和边界

全部 343 个问题在两组种子的内部点上比较前向 Jacobian 与独立差分，包括 `F/G/H`。
对强振荡、强偏置或病态大数抵消，使用明确写入测试的良态点：
NCTP 的 Rosenbrock 谷、DOC 的分量最优点附近、LSCM 的 CEC6 小半径分量、
WFG9/MaF12/DASCMaOP9 的中部决策、MW13 的较小距离等。
DOC5/7 使用更细差分步长。未通过简单放宽容差来掩盖振荡混叠。
微量级工程变量的差分尺度受其变量盒宽度约束。

另外检查动态时间不可变、Eq 等式解析导数、批次独立性、来源别名、被排除的离散成员、
MaF9 的定义域、LSCM 的维度限制以及不可微/奇异点的错误行为。
正则点数值核验不能证明全域可微；`regular=True` 只表示没有触发已实现的分支检查。
大规模前向 AD 的种子内存为 `O(N*n_var²)`，不是低成本的反向模式。
一般约束问题需要能处理 `G/H` 的算法；已有盒约束 MGDA 等基线没有因此获得通用约束求解能力。

## 运行核验

```sh
python -m pip install -e ".[dev,reference,algorithms,plot]"
python -m pytest --cov=mop_problems --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python scripts/build_catalog.py --check
python examples/expanded_benchmarks.py --samples 1000 --seed 42
python -m build
python -m twine check dist/mop_problems-0.4.0*
python scripts/check_wheel.py dist/mop_problems-0.4.0-py3-none-any.whl
```

原有 0.3.0 梯度算法的核验记录保留在 [gradient-validation.md](gradient-validation.md)。
CI 配置了 Linux Python 3.10–3.14 及 Windows/macOS；本地通过不等于这些托管任务已全部通过。
