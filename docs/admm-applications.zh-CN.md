# 有实际背景的多目标 ADMM 分离问题

检索与核验日期：2026-10-18。英文入口：[ADMM application catalogue](admm-applications.md)。

本专题补充 **12 个应用模型/文献条目**，用于选择多目标 ADMM 实验问题。
条目属于研究目录；尚未注册为 `get_problem()` 可调用问题，不增加现有 343 个基准的数量。
[机器可读索引](admm-applications.json)记录出处、证据层级、分离关系和数据限制；
[BibTeX](admm-applications.bib)可直接导入文献管理器。
[数值示例](../examples/admm_application_splits.py)只核验四种小规模构造的分离等价性。

## 先核对算法要求

统一考察

$$
\min_{x\in X,\ z\in Z}
F(x,z)=\bigl(f_i(x)+g_i(z)\bigr)_{i=1}^m,
\qquad Ax+Bz=c.
$$

每个目标都需要可分，不能仅凭论文写了“multi-agent”或“ADMM”判断。
$X,Z$ 内可以保留局部约束；跨块非线性约束不能冒充线性耦合。
没有自然分块时，可以增加复制变量或物理输出变量，但须给出等价关系。

以下公式是本目录统一符号后的表达；注明“本目录改写”的公式是我们的数学建模推导，
不是论文原式或作者已经提出的多目标算法。

- **A：论文明确处理多目标，并使用 ADMM。** 仍需区分加权和、满意度标量化、字典序以及直接向量迭代。
- **B：应用论文提供标量模型/算法，本目录将多个损失或性能项保留成向量。** 这是可研究的多目标扩展，不宣称原论文研究了该向量版本。
- **全文核验**表示已查到正文中的模型或分离步骤；**摘要核验**只确认背景和方法，详细参数与公式仍需核对。

经典凸两块 ADMM 一般用于固定权重的标量问题。求解一组权重不等于直接实现多目标 ADMM；
正权重的全局最优解是 Pareto 解，允许零权重时一般只能保证弱 Pareto 性。
非凸问题还存在加权和无法覆盖非支撑 Pareto 点的问题。
这些结论需要最优解存在及相应凸性/全局求解条件。

**与有限凸、全空间逐目标水平集有界理论的关系：**
可分结构、凸性、可行集有界是不同条件。多数下列原始模型带局部硬约束，
且复制后部分 $f_i$ 或 $g_i$ 为零；这通常不满足“每个目标在整个乘积空间上水平集有界”。
不能直接套用该类分离权重 ADMM 定理。使用指标函数需要扩展值/定义域理论；
增加小二次项、去掉整数条件或采用软罚都会改变原问题，必须使用新变体名称。
一般的多块顺序 ADMM 也不能仅凭分块数增加而继承两块收敛结论。

## 总览

“优先”指模型与复现便利程度，不是算法已经通过实验。

| ID | 应用、目标 | 主要耦合 | 类型与适用性 | 证据 |
| --- | --- | --- | --- | --- |
| A01 | 家庭储能：削峰 / 保留调节灵活性 | 家庭功率平均值与全局轨迹一致 | 凸；需局部可行域处理；优先 | 全文 |
| A02 | AC/DC 配电网：网损 / 电压偏差 / 调节成本 | 区域边界功率、电压副本一致 | 连续锥松弛可凸；原模型有离散设备 | 全文 |
| A03 | 园区综合能源：运行成本 / 碳排放 | 电、热、冷供需平衡 | 固定转换效率下有凸化途径；需核对原设备约束 | 全文 |
| A04 | 联网微电网：经济性 / 网络运行指标 | DMS 与参与者的节点功率平衡 | 未完成逐约束凸性核验 | 摘要 |
| A05 | Gamma Knife：靶区剂量 / 正常组织剂量 / 照射时间 | 剂量映射、变量复制 | 固定采样与几何下为 LP；优先 | 全文 |
| A06 | 交通 MPC：边界通行量 / 网内运行质量 | 相邻子网流量副本一致 | 论文凸模型；字典序，不能直接替换为 Pareto 扫权 | 全文 |
| A07 | IPM 电机：平均转矩 / 铁损 | 连续设计中的离散坐标与整数副本一致 | 混合变量非凸；ADMM 启发式 | 全文 |
| B01 | 稀疏投资：风险 / 收益，另设持仓数上限 | 持仓副本一致 | 基数约束非凸；有连续对照 | 全文 |
| B02 | 图像恢复：数据拟合 / 总变差 | 图像梯度与辅助变量一致 | 凸非光滑；优先 | 全文 |
| B03 | 鲁棒 PCA：低秩 / 稀疏 | 低秩项 + 稀疏项 = 观测 | 凸非光滑；优先 | 全文 |
| B04 | 高光谱解混：重建 / 稀疏 / 空间 TV | 丰度副本与空间差分一致 | 固定字典时凸；非光滑；优先 | 作者文献目录及参考代码 |
| B05 | 连续投资：风险 / 收益 / 交易量 | 新持仓 − 交易量 = 旧持仓 | 凸；需局部投资约束；优先 | 作者页面及链接全文 |

## A01　家庭储能的削峰与灵活性

**来源：** Philipp Sauerteig and Karl Worthmann, *Towards multiobjective optimization and
control of smart grids*, Optimal Control Applications and Methods 41, 128–145 (2020；
2019 年在线发表)。[DOI](https://doi.org/10.1002/oca.2532) ·
[开放全文](https://d-nb.info/1211604306/34)。核验位置：§2.1–2.4、§3。

家庭有光伏、负载和电池。$p_j\in\mathbb R^T$ 是第 $j$ 个家庭的净购电轨迹，
$q=N^{-1}\sum_jp_j$ 是平均功率，$r$ 为削峰参考，
$\ell,u$ 为期望调节区间。令 $a_+=\max(a,0)$，逐坐标运算：

$$
F_1=\frac1T\|q-r\|_2^2,\qquad
F_2=\|(\ell-q)_+\|_2^2+\|(q-u)_+\|_2^2.
$$

取 $x=(p_1,\ldots,p_N)$，$z=q$，则 $f_1=f_2=0$，
$g_1(z)=T^{-1}\|z-r\|^2$，
$g_2(z)=\|(\ell-z)_+\|^2+\|(z-u)_+\|^2$，
$A=N^{-1}[I\ \cdots\ I]$，$B=-I$，$c=0$。
每个家庭的 SOC 动力学、充放电功率与容量限制保留在局部集合 $D_j$。

这是消去论文松弛变量后的等价结构表达；论文也给出带松弛变量的形式。
论文采用标量化后 ADMM，并讨论 Pareto 前沿。适合作为最先复现的能源案例。
需要家庭负荷/光伏时间序列、初始 SOC、效率及电池参数；论文引用澳大利亚配电数据，
本专题没有取得或重新授权该数据。零局部目标不满足全空间逐目标强制性。

## A02　混合 AC/DC 配电网协调

**来源：** Xiaoxue Wang, Liting Gu and Dong Liang (2021),
*Decentralized and Multi-Objective Coordinated Optimization of Hybrid AC/DC Flexible
Distribution Networks*. [DOI / 全文](https://doi.org/10.3389/fenrg.2021.762423)。
核验位置：多目标模型、式 (21)–(23)、锥松弛段落。

三个目标是网损、电压偏差和设备调节成本。区域 $r$ 的变量 $v_r$
包含潮流、电压、无功和设备动作。将每项物理损失/成本只计入一个区域，
可写为 $F_i=\sum_r\phi_{ir}(v_r)$。
取 $x=(v_r)_r$，$z$ 为公共边界变量，
$f_i(x)=\sum_r\phi_{ir}(v_r)$，$g_i(z)=0$，耦合 $Ex-Rz=0$。
这里 $E$ 提取各区域边界量，$R$ 将同一公共边界量复制到相邻区域对应位置。
边界副本的方向必须统一，不能重复计入耦合支路损耗。

论文采用加权和与区域 ADMM。原 AC 潮流非凸，论文使用 SOCP 等处理；
OLTC 等离散动作还需固定、枚举或显式保留为混合整数问题。
“SOCP 松弛”不等于证明所得解满足原 AC 等式。
算例为修改的双 IEEE 33 节点系统；拓扑、设备、边界划分和松弛精确性均要复核。

## A03　园区综合能源的经济—碳排放权衡

**来源：** Kaiwen Xu, Zeyu Liu and Weichen Sun (2022),
*Optimal operation of park-level integrated energy system based on multi-agent
cooperative game*. [DOI / 全文](https://doi.org/10.3389/fenrg.2022.970344)。
核验位置：§2–3、§4.1–4.2。

设备包括储能、冷热源和 CHP。保留设备调度 $v_j$ 的运行成本
$C_j(v_j)$ 与排放 $E_j(v_j)$ 为两个目标：
$F=(\sum_j C_j,\sum_j E_j)$。
**本目录分组：** 将设备分为两个非重叠集合 $J_x,J_z$，
$f=(\sum_{J_x}C_j,\sum_{J_x}E_j)$，
$g=(\sum_{J_z}C_j,\sum_{J_z}E_j)$。
固定转换效率时，电、热、冷平衡堆叠成 $Ax+Bz=d$；
局部集合保留设备容量与储能时间耦合。

论文使用 AA-ADMM 并计算权重/权衡，之后用 Shapley 值分配合作收益。
收益分配不应当作新的物理调度目标混入。
正文给出设备表和典型日曲线，但没有在这里取得完整机器可读实例。
含启停、互斥或非线性效率时不能声称整个问题凸；两块内的设备可并行计算
也须依据实际子问题，而不是直接使用无保证的多块顺序更新。

## A04　联网微电网与配电管理系统

**来源：** Guodong Liu, Thomas B. Ollis, Maximiliano F. Ferrari, Aditya Sundararajan
and Yang Chen, *Distributed Energy Management for Networked Microgrids Embedded
Modern Distribution System Using ADMM Algorithm*, IEEE Access 11, 102589–102604
(2023)。[DOI](https://doi.org/10.1109/ACCESS.2023.3316513) ·
[作者机构摘要](https://www.ornl.gov/publications/distributed-energy-management-for-networked-microgrids-embedded-modern-distribution-system-using-admm-algorithm/)。

**证据限制：** 本次核验到机构摘要；机构页面日期与 DOI/卷年记录有差异，
引用采用 DOI 对应 2023 卷年，不把机构入库日期当作论文首发时间。
摘要确认多参与者运行指标、AHP 权重及 ADMM 节点平衡协调。

可研究的分块为参与者计划 $x=(v_j)_j$ 与 DMS 网络变量 $z$：
$F_i=\sum_j C_{ij}(v_j)+C_{i,\mathrm{DMS}}(z)$，
节点注入一致性为 $Px+Bz=d$。
**这只是本目录提出的结构模板**，不是已逐式复现的论文模型。
必须先从全文核对每个 $C_{ij}$、潮流模型和整数条件，才能进入代码问题库。
原始实例、约束凸性和具体目标清单尚待全文审计，故不列为立即可复现实验。

## A05　Gamma Knife 放射外科计划

**来源：** Joakim da Silva, Daniel Hernández Escobar, Tor Kjellsson Lindblom,
Håkan Nordström and Jens Sjölund,
*A parallel algorithm for generating Pareto-optimal radiosurgery treatment plans*。
[2025 年预印本 v1](https://arxiv.org/html/2509.08602v1) ·
[2026 年期刊 DOI](https://doi.org/10.1002/mp.70454)。
核验位置：§II.A–II.B、附录 A；本目录结构依据预印本 v1。

物理变量为各照射配置的持续时间 $t\ge0$，剂量 $d=Kt$。
**本目录的通用计划模板**取
$x=t$，$z=d$，$A=K,B=-I,c=0$，并保留
$F=(U_T(d),O_N(d),b(t))$：靶区欠剂量、正常组织超剂量/低剂量代理和
照射时间代价。其中 $f=(0,0,b)$，$g=(U_T,O_N,0)$；
可用分段线性剂量罚及线性辅助变量构造 LP。

论文的实际实现从加权 LP 的**对偶**出发，增加变量复制并作 ADMM 图投影；
上面是便于向量建模的原问题表达，不是照抄其更新式。
固定采样、几何和计划阶段时模型为凸；不能把两阶段低剂量点重采样、
DVH 或离散射束选择都看作一个固定凸模型。
论文使用 20 个病例；本次没有获得病例剂量矩阵。
如使用其他公开放疗数据替代，须核验许可，并使用独立实例名称；
不能将常规 IMRT 数据称为该文的 Gamma Knife 病例。

现已增加[四目标合成实例与独立求解器](radiosurgery.md)，明确区分环区过剂量与低剂量外溢，
以扇区最大照射时间辅助变量实现第四目标；包含原文固定权重对偶 ADMM、MO-ADMM、MO-ALM
及数值优化/罚参数对照。该应用模块不计入 343 个标准基准问题，使用 COPT 的部分需要可选依赖和许可证。

## A06　城市交通的字典序 MPC

**来源：** Viet Hoang Pham and Hyo-Sung Ahn (2025),
*Distributed MPC-based Coordination of Traffic Perimeter and Signal Control:
A Lexicographic Optimization Approach*。
[预印本 v1](https://arxiv.org/html/2510.04038v1)。
核验位置：§III.B–III.C、§IV、附录 A。期刊发表状态本次未另行确认。

一级目标最大化可接纳的边界流量，二级目标改善网内交通状态。
以最小化形式记为 $(-\sum_r I_r(v_r),\sum_r Q_r(v_r))$。
区域内包含车流守恒、道路容量和绿灯时间约束。
**统一分离模板：** $x=(v_r)_r,z=$ 边界流量副本，
$f=(-\sum I_r,\sum Q_r)$，$g=(0,0)$，$Ex-Rz=0$。
$E$ 提取各子网的边界副本，$R$ 复制公共边界值，使相邻副本一致。

论文在修改的 store-and-forward 凸模型上分级求解，并用近端 ADMM 分布式协调。
复现必须保留一级最优值对二级问题的限制及数值容差；
改用一组普通权重会改变偏好规则。真实交通过程与优化中的凸预测模型也应区分。
需要网络、转向比例、饱和流率、预测需求和仿真设置；论文使用 VISSIM/MATLAB，
这里未取得完整复现包。

## A07　混合变量 IPM 电机设计

**来源：** Tatsuya Asanuma and Yoshihiro Kanno (2025),
*Multi-objective surrogate optimization of mixed-variable IPM motor design problem
with ADMM-based approach*, Optimization and Engineering 26, 1699–1724。
[DOI / 全文](https://doi.org/10.1007/s11081-024-09954-9)。
核验位置：§3.3、§3.6，式 (24)，数据可用性段落。

原文两个目标是最大化平均转矩、最小化铁损；
转矩脉动在文中属于未来扩展，不能误列为已实验的第三目标。
连续/整数设计坐标共同记为 $x$，$z$ 复制离散坐标，
$S$ 为选取矩阵。向量表达为
$f=(-\widehat T(x),\widehat L_{\mathrm{iron}}(x))$，
$g=(0,0)$，$Sx-z=0,z\in\mathcal D$。
连续块保留几何约束。

论文采用 RBF 代理与 STOM 满意度标量化，ADMM 用于混合变量启发式处理。
代理非凸、离散集合非凸，即便耦合是线性的，也不属于凸 ADMM 定理适用范围。
设计数据需要向作者申请；昂贵电磁 FEA 和代理更新也是成本的一部分。
适合作为后续非凸拓展案例，不宜直接用于验证凸算法的全局结论。

## B01　带持仓数限制的风险—收益模型

**来源：** Zhang-Lei Shi, Xiao Peng Li, Chi-Sing Leung and Hing Cheung So,
*Cardinality Constrained Portfolio Optimization via Alternating Direction Method of
Multipliers*, IEEE TNNLS 35(2), 2901–2909 (2024；2022 年在线发表)。
[DOI](https://doi.org/10.1109/TNNLS.2022.3192065) ·
[作者全文](https://li-x-p.github.io/files/publications/CCPO-ADMM.pdf)。
核验位置：式 (5)、(12)–(14)。

**本目录保留为双目标：** $F=(x^\top\Sigma x,-\mu^\top x)$，$\Sigma\succeq0$。
取 $f=(x^\top\Sigma x,-\mu^\top x)$，$g=(0,0)$，
$x-z=0$，局部条件 $\mathbf1^\top x=1,\|z\|_0\le k$。
论文原模型允许的持仓符号需遵守原式；不能默认为 long-only。

论文求解的是带风险偏好参数的标量化；实现还将预算等式改成二次罚。
有限罚系数一般不保证预算精确成立，本目录的硬等式版和论文罚函数版应分别命名。
基数约束使问题非凸；只保留均值—方差及凸投资约束才有凸对照。
论文使用真实金融数据，但本次没有取得时间窗口、预处理与使用许可。
不要把非负且预算为一的 $\ell_1$ 持仓范数当作有效稀疏目标，它恒为一。

## B02　图像去噪/去模糊的拟合—TV 权衡

**来源：** Manya V. Afonso, José M. Bioucas-Dias and Mário A. T. Figueiredo (2010),
*Fast Image Recovery Using Variable Splitting and Constrained Optimization*。
[DOI](https://doi.org/10.1109/TIP.2010.2047910) ·
[2009 年预印本](https://arxiv.org/pdf/0910.4887)。
核验位置：式 (2)、(5)、变量分裂与 ADMM 部分。

论文研究数据拟合加正则的标量问题。**本目录向量化**为
$F=(\frac12\|Hx-b\|^2,\|Dx\|_{2,1})$，
其中 $H$ 是固定成像算子，$D$ 是空间差分。
令 $z=Dx$：
$f=(\frac12\|Hx-b\|^2,0)$，$g=(0,\|z\|_{2,1})$，
$A=D,B=-I,c=0$。
二维 TV 的分组方式及边界条件必须固定；一维例子使用 $\ell_1$ 差分。

两个目标均凸，第二个非光滑。$H$ 为欠采样时数据项可能不强制；
TV 还有常数方向零空间。不能从凸性推断所有目标在全空间水平集有界。
实验需要有许可的原图、噪声种子、模糊核与边界规则。
随权重变化的是 Pareto 权衡候选，不应以 PSNR 单一指标代替双目标结果。

## B03　鲁棒 PCA 的低秩—稀疏分解

**来源：** Emmanuel J. Candès, Xiaodong Li, Yi Ma and John Wright,
*Robust Principal Component Analysis?*。
[作者预印本](https://arxiv.org/abs/0912.3599)。
ADMM 结构背景另见
[Boyd et al. (2011)](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)。

视频背景与运动前景、遮挡人脸等可写为 $M=L+S$。
**本目录向量化**是
$F=(\|L\|_*,\|S\|_1)$，
$f=(\|L\|_*,0)$，$g=(0,\|S\|_1)$，
$A=I,B=I,c=\operatorname{vec}(M)$。
$\|S\|_1$ 为元素绝对值和，核范数使用奇异值之和。

原文 PCP 是 $\|L\|_*+\lambda\|S\|_1$ 标量模型；
这里不声称原文提出直接多目标 ADMM。
两块凸非光滑；固定权重下分别出现奇异值阈值与元素软阈值。
有稠密噪声时，可另建 $\|M-L-S\|_F\le\delta$ 变体，
但不要未经证明直接改成三块顺序 ADMM。
本专题不打包人脸/视频数据；小矩阵仅用于验证分离和范数计算。

## B04　高光谱丰度的重建—稀疏—空间平滑

**来源：** Marian-Daniel Iordache, José M. Bioucas-Dias and Antonio Plaza (2012),
*Total Variation Spatial Regularization for Sparse Hyperspectral Unmixing*。
[DOI](https://doi.org/10.1109/TGRS.2012.2191590) ·
[作者文献目录](https://it.pt/Members/Index/396) ·
[参考实现文件](https://github.com/ricardoborsoi/MUA_SparseUnmixing/blob/master/sunsal_tv.m)。
本次以作者目录及代码定义交叉核验，未取得出版社全文逐式审计；
参考代码是第三方项目中保留的实现，不代表本次核验了其完整版权链或执行结果。

固定光谱字典 $H$，观测 $Y$，丰度 $U\ge0$，空间差分 $D$。
**本目录三目标扩展：**
$F=(\frac12\|HU-Y\|_F^2,\|U\|_1,\|DU\|_1)$。
将丰度向量化，令 $x=\operatorname{vec}(U)$，$z=(v,w)$，
$v=x,w=Dx$，故 $A=[I;D],B=-I,c=0$；
$f=(\frac12\|HU-Y\|_F^2,0,0)$，
$g=(0,\|v\|_1,\|w\|_1)$，$X=\{U\ge0\}$。

固定字典时凸；若同时估计字典则出现双线性，不能沿用此判定。
非负加每像素和为一会使 $\|U\|_1$ 恒定，失去稀疏权衡；
须明确选择无和为一约束的版本，或改用有区分力的稀疏指标并重新分析凸性。
复现需字典版本、波段预处理、真实影像许可及 TV 边界；
这里不下载或重新分发影像，也不把三目标扩展称为原 SUnSAL-TV 多目标算法。

## B05　连续投资的风险—收益—交易量

**来源：** Stephen Boyd et al. (2017),
*Multi-Period Trading via Convex Optimization*。
[作者页面与代码入口](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html) ·
[DOI](https://doi.org/10.1561/2400000023)。

**本目录单期简化与三目标改写：** 给定旧持仓 $h$，新持仓 $x$，交易量 $z$，
保留 $F=(x^\top\Sigma x,-\mu^\top x,\|z\|_1)$，
其中 $\Sigma\succeq0$。
$f=(x^\top\Sigma x,-\mu^\top x,0)$，
$g=(0,0,\|z\|_1)$，$x-z=h$，
$X=\{x\ge0,\mathbf1^\top x=1\}$。
这将持仓/风险块与交易量/换手块分离。

原文讨论收益、风险与交易/持有成本的凸优化框架；
此处的三目标模型与 ADMM 分块是本目录推导，不是原文已有的多目标 ADMM。
$\ell_1$ 交易量是简化的换手代理；忽略自融资交易费用、现金账户和市场冲击时，
不能宣称完整复现原多期交易模型。
可使用经许可的历史数据计算 $\mu,\Sigma$，记录估计窗口；
数值示例只给明确的小型合成数据，不作收益预测或交易策略评价。

## 实验落地顺序与记录要求

建议先做 B02、B03、B05 的可核验小型实例，再做 A01。
B04 适合增加第三目标和空间结构；A05 的应用价值高，但真实剂量矩阵获取更费力。
A02、A03 需要把设备约束与凸化边界说清楚；A07、B01 留作非凸扩展；
A04 先完成全文审计，A06 保留字典序偏好。

每个正式实现至少保存：原文/改写版本、数据许可与版本、矩阵维度、
$f_i,g_i,A,B,c,X,Z$、尺度与单位、可行点、独立数值检查、
标量化方法、原始及对偶残差、共同权重驻点指标、内外层成本。
约束罚函数、加权和和向量模型不能共用同一个未标注的实例名。

运行结构示例：

~~~sh
python examples/admm_application_splits.py
~~~

输出四个案例的目标值、分离等价误差、可行残差及扰动后的非零残差。
这是构造检查，未运行多目标 ADMM，未复现上述论文的前沿、速度或真实数据结果。
