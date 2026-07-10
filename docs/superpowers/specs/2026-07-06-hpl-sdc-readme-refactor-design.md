# HPL SDC README.md 深度重构设计规范 (Design Spec)

**日期**：2026-07-06  
**主题**：`hpl-sdc-readme-refactor`  
**目标**：深度结合 HPL SDC 源码（尤其是 `sdc/` 目录下的 4 个核心文件及 `pgesv/` 中的调用链路），全面重构项目主文档 `readme.md`，缓解原文档中概念、函数接口、数据结构相互割裂的问题，实现 **源码对齐、结构清晰、简洁有力**。

> **SDC 定位更新**：本文档中的 SDC 模块应统一理解为 **low-overhead heuristic SDC detection and reporting layer**；各检测路径只能报告可观测异常，不承诺形式化容错或完整检出保证。

---

## 一、 背景与重构动因 (Background & Motivation)

当前 `readme.md` 共 507 行，虽内容详尽，但在结构与技术叙事上存在以下痛点：
1. **理论与实现割裂**：在“第 4 章：SDC 检测增强模块”中，4.2 节介绍了“四类检测路径”概念，但对应的核心函数接口被分散在 4.3 节，故障枚举定义被分散在 4.4 节，面板结构体修改被分散在 4.5 节。读者无法在阅读某一类检测路径时立即形成“因果机制 - 数学边界 - 源码实现”的闭环。
2. **冗余与冗长**：部分基础背景（如传统的 HPL 介绍、重复的 Look-ahead 描述）占用较多篇幅，稀释了本项目最具创新性的 SDC 增强成果（如 Kahan 无权值求和、JIT 面板准入记录、按字段独立聚类汇聚等）。
3. **技术冲击力可提升**：重构需对齐开源重磅工程项目与顶级系统顶会论文的风格，强调“痛点 -> 极简创新 -> 编译期开关隔离 -> 集群诊断排障”。

---

## 二、 重构后的总体架构 (Proposed Structure)

重构后的 `readme.md` 将保持 8 个主要章节，重点对 **第二章（目录结构）** 和 **第四章（SDC 检测增强模块）** 进行深度重构：

```text
# HPL (High Performance Linpack) — 含 SDC 静默数据损坏检测增强
├── 一、 项目概述与 SDC 挑战 (保留 FLOPS 计算公式，精炼 Exascale 痛点与解决思路)
├── 二、 项目目录结构 (精炼树状图，准确标注 SDC 核心文件与 4 种 Makefile 构建体系)
├── 三、 HPL 核心算法与分布式流水线 (保留 2D 块循环、右瞻 LU、Look-ahead 机制、6 种广播拓扑表)
├── 四、 SDC 检测增强模块 —— 启发式检测路径体系 (★ 本次重构的核心，深度闭环)
│   ├── 4.1 底层算法基石：Kahan 补偿求和与自适应双模断言
│   ├── 4.2 四类检测路径与源码映射 (4 Subgraph 图 + 逐层闭环剖析)
│   ├── 4.3 数据结构精简与编译期开关隔离 (分析 CS_TRAIL 废弃原因与宏隔离)
│   └── 4.4 分布式运维：按字段独立聚类汇聚与故障定位 (解决异构内存对齐难题)
├── 五、 编译构建说明 (清楚列出 4 种 make arch 及依赖)
├── 六、 运行与故障注入测试 (xhpl_sdc_test 的 7 大测试组与环境变量注入)
├── 七、 关键参数与调优建议 (精简表述)
└── 八、 开销分析 (保留 <0.5% 开销对比表)
```

---

## 三、 核心章节深度设计 (Detailed Section Specifications)

### 第四章：SDC 检测增强模块 —— 启发式检测路径体系 (Section 4 Spec)

本章是本次重构的灵魂。打破原先“概念”、“函数”、“结构体”平铺割裂的格局，采用**深度闭环**结构：

#### 4.1 底层算法基石：Kahan 补偿求和与自适应双模断言
* **Kahan 补偿求和 (Compensated Summation)**：
  * **痛点**：在百亿亿次超算规模下，上百万维度的标准浮点累加 $\sum A[i]$ 会因舍入误差累积产生 $O(\sqrt{m}\varepsilon)$ 甚至 $O(m\varepsilon)$ 的数值漂移，足以引发虚警。
  * **解决方案**：呈现 `HPL_sdc_checksum.c` 中使用的 Kahan 算法核心代码与公式，说明其利用补偿余量 `c` 捕获低位截断误差，降低累加漂移。
* **自适应双模断言 (Adaptive Hybrid Thresholding)**：
  * **痛点**：当待比对的参考指纹极小或接近于零（如数值奇异或消元尾部）时，标准相对偏差 $|cs_{comp} - cs_{exp}| / |cs_{exp}|$ 会因除零发生剧烈发散。
  * **解决方案**：呈现 `HPL_sdc_verify.c` 中 `HPL_sdc_verify_checksum` 的核心逻辑公式：
    $$\text{Judgment} = \begin{cases} |dev| > \max(\text{threshold}, 10^{-12}), & \text{if } |cs_{exp}| < 10^{-4} \\ \frac{|dev|}{|cs_{exp}|} > \text{threshold}, & \text{otherwise} \end{cases}$$

#### 4.2 四类检测路径与源码映射
首先放置更新后的 4 个 Subgraph 模块化 Mermaid 架构图。随后，**对每一类检测路径进行“因果机制 - 数学边界 - 源码映射”三维一体的深度剖析**：

1. **路径 1：JIT 面板准入检测 (Cache-Resident Panel Entry Check)**
   * **因果机制**：在 Look-ahead 右瞻求解中，第 $k$ 步待分解的面板（宽度 $jb=192$），正是前序所有 `DGEMM`（尾矩阵更新）累积计算的结果。由于 `DGEMM` 占据了 ~99% 的浮点计算负载，部分发生于 FPU、寄存器堆或内存中的静默比特翻转，可能最终留存在待分解的面板数据中。
   * **数学边界**：
     * **SIMD IEEE 754 异常扫描**：扫描并记录 `NaN`、`+Inf` 和 `-Inf`。
     * **动态收敛包络线断言**：高斯消元未消元矩阵元素的绝对范围随深度严格递减，上限公式锁定为 $10^{150} \times (1 - \frac{j}{2N})$。
   * **源码映射**：
     * **调用链**：`HPL_pdgesvK2.c:L193`（以及 `K1.c:L187` / `0.c:L185`）在调用 `HPL_pdfact` 前夕执行。
     * **核心函数**：`HPL_sdc_verify.c` -> `HPL_sdc_verify_panel_entry(A, lda, m, n)`。
     * **故障枚举**：触发并记录 `HPL_SDC_FAULT_PANEL_ENTRY` (`type = 2`)。
     * **架构重构收益**：精确阐明：正是因为这类 JIT 准入检测路径能在消元前夕尽早暴露一部分传播到面板的可观测异常，项目**不再维护了原版设计中对 `LASWP` 敏感且内存沉重的尾矩阵校验和 (`CS_TRAIL`) 及加权向量**，极大地精简了数据结构与通信开销！

2. **路径 2：面板分解一致性检测 (Panel Factorization Check)**
   * **因果机制**：面板局部 LU 分解包含列选主元 (`IDAMAX`)、行置换 (`LASWP`) 和下三角求解 (`DTRSM`)，对 CPU 的分支预测与 L1/L2 缓存极其敏感。
   * **校验逻辑**：对分解完成的算子与主元索引进行完备性校验。
   * **源码映射**：
     * **调用链**：`HPL_pdfact` 面板分解引擎。
     * **故障枚举**：记录 `HPL_SDC_FAULT_PANEL_FACT` (`type = 1`)。

3. **路径 3：广播缓冲区检测 (Broadcast Buffer Check)**
   * **因果机制**：面板所有者将下三角因子 $L_1, L_2$ 及主元映射表 `DPIV` 沿列网格（`row_comm`）广播给各接收进程。若交换机、光纤链路、网卡 DMA 发生比特位翻转，错误将扩散至全网格。
   * **数学边界**：
     * **无权值 Kahan 指纹**：主进程对发送缓冲区执行补偿累加，计算出广播载荷指纹 `cs_bcast`。
     * **低开销同步与自适应比对**：通过 `MPI_Bcast` 将指纹发往接收端；接收端重算 `cs_recv`，在 `HPL_sdc_verify_checksum` 中进行自适应双模比对。
   * **源码映射**：
     * **指纹计算**：`HPL_sdc_checksum.c` -> `HPL_sdc_compute_bcast_checksum(L2, ldl2, ml2, L1, jb_l1, DPIV, jb, cs_out)`。
     * **通信与触发**：`HPL_pdgesvK2.c:L222-L252`，比对异常则记录 `HPL_SDC_FAULT_PANEL_BCAST` (`type = 0`)。

4. **路径 4：全局终态兜底 (Global Residual Gate)**
   * **因果机制与数学边界**：
     * **解向量 6-$\sigma$ 统计学离群点筛查 (Anomalous & Outlier Detection)**：在 `HPL_pdtrsv.c:L300-L348` 上三角回代及解向量广播中，系统不仅检查 `NaN/Inf`，还实时计算全局解向量 $X$ 的均值 $\mu$ 和标准差 $\sigma$。若发现超出 $6\sigma$ 阈值的离群点（$|x_i - \mu| / \sigma > 6.0$），立即判定计算异常！
     * **绝对闭环闸门**：全流程结束前，计算 HPL 官方标准的高精度缩放无穷范数残差：
       $$\frac{\|A \cdot x - b\|_\infty}{\varepsilon \cdot (\|A\|_\infty \|x\|_\infty + \|b\|_\infty) \cdot N} < 16.0$$
   * **源码映射**：
     * **调用链**：`HPL_pdtrsv.c`（触发 `HPL_SDC_FAULT_BACK_SOLVE`, `type = 3`）与主驱动 `HPL_pddriver.c`。

#### 4.3 数据结构精简与编译期开关隔离
* **`HPL_T_panel` 控制块重构精髓**：
  * 列出 `hpl_panel.h:L95-L98` 中的条件编译字段：
    ```c
    #ifdef HPL_SDC_CHECK
       double    cs_bcast;   /* checksum of broadcast buffer */
       int       sdc_step;   /* per-panel pdupdate call counter */
    #endif
    ```
  * 深刻对比说明：得益于路径 1 (`verify_panel_entry`) 与路径 3 (`cs_bcast`) 的组合使用，旧方案中用来记录尾矩阵增量的 `CS_TRAIL`、面板列校验和 `CS_PANEL` 以及模 16 权值向量 `CS_WEIGHTS` 被移除！这为每个面板节省了可观的内存，同时降低了加权计算的复杂性。
* **编译期开关隔离**：
  * 阐明 `HPL_SDC_CHECK` 编译宏的总开关作用。当不定义该宏（如构建 `WSL_OpenBLAS`）时，所有 SDC 数据结构、指纹计算与验证分支由编译期开关排除，避免在该构建中引入检测路径运行时分支。

#### 4.4 分布式运维：按字段独立聚类汇聚与故障定位
* **$O(1)$ 无阻塞链表日志**：
  * 解析 `hpl_sdc.h` 中的 `HPL_T_SDC_FAULT` 结构体，说明 `HPL_sdc_log_fault` 如何在主循环中以 $O(1)$ 头部插入方式记录 physical node name (`MPI_Get_processor_name`)、2D 网格坐标 (`myrow, mycol`) 及全局矩阵绝对切片位置 (`ia, ja`)。
* **按字段独立聚类汇聚技术 (Per-Field Independent Gathering)**：
  * **行业难题**：在异构分布式超算集群或不同编译器优化下，C 语言结构体的字节对齐与填充（Padding/Alignment）规则不同。若在 `MPI_Gather` 中直接传递打包结构体，极易触发解包错位或内存段错误。
  * **我们的突破**：详细解析 `HPL_sdc_report.c:L130-L298` 中的 `HPL_sdc_report_and_aggregate` 实现：系统摒弃了结构体打包，改为对底层 10 个基础字段（`mpi_rank`, `grid_row`, `grid_col`, `fault_type`, `step`, `global_row`, `global_col`, `cs_expected`, `cs_computed`, `deviation`, `node_name`）分配独立的类型缓冲区，利用各自准确的 MPI 基础类型（`MPI_INT`, `MPI_DOUBLE`, `MPI_CHAR`）独立发起 `MPI_Gatherv`，实现了跨架构 更高的跨架构汇聚可移植性！
* **故障聚合与诊断建议引擎**：
  * 展示标准的输出诊断报告，说明系统如何自动统计各故障类型与各节点的故障密度，并给出明确的硬件运维指令（如 `RECOMMENDATION: Replace nodes with >10 faults: compute-node-042`）。

---

## 四、 实施与验证计划 (Implementation & Verification Plan)

1. **文档重构实施**：
   * 严格按照本设计规范修改 `C:\Users\ubuntu\Documents\Linpack-HPL\readme.md`。
   * 确保 LaTeX 公式、Mermaid 流程图和 C 代码块语法完整规范。
2. **准确性核查**：
   * 逐字段核对 `hpl_sdc.h`、`hpl_panel.h` 与 `HPL_sdc_report.c` 中的命名与行号，检查是否存在虚构或过期描述。
3. **提交与推送**：
   * 在 WSL 环境下执行 `git add docs/superpowers/specs/2026-07-06-hpl-sdc-readme-refactor-design.md readme.md`，提交并推送至远端仓库。
