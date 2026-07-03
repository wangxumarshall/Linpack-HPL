# HPL (High Performance Linpack) — 含 SDC 静默数据损坏检测增强

## 一、项目概述

**HPL（High Performance Computing Linpack）** 是国际标准的高性能计算机浮点性能基准测试工具，由 University of Tennessee 开发（v2.3）。它通过求解大规模稠密线性方程组 $Ax = b$（双精度 64 位浮点运算）来衡量分布式内存系统的浮点计算性能。

**性能计算公式**：

$$R = \frac{\frac{2}{3}N^3 + \frac{3}{2}N^2}{T}$$

其中 $N$ 为矩阵维度，$T$ 为求解时间（秒），$R$ 的单位为 FLOPS。

**本项目增强**：在原版 HPL 基础上，新增了 **SDC（Silent Data Corruption，静默数据损坏）** 检测模块。SDC 是指宇宙射线或硬件静默故障引发的内存/寄存器比特翻转，传统 HPL 仅在求解结束后通过残差检验判断"是否出错"，无法定位故障发生的时间与节点。本增强模块基于 ABFT（Algorithm-Based Fault Tolerance）思想，在 LU 分解的关键路径上插入校验和检测点，实现运行时实时检测与节点级故障定位。

---

## 二、项目目录结构

```
Linpack-HPL/
├── hpl/                          # HPL 主目录
│   ├── src/                      # 核心源码
│   │   ├── auxil/                # 辅助工具函数（打印、内存等）
│   │   ├── blas/                 # 本地 BLAS 操作（dgemm, dtrsm, dgemv 等封装）
│   │   ├── comm/                 # MPI 通信层（广播 HPL_bcast、归约、屏障等）
│   │   ├── grid/                 # 处理器网格管理（HPL_grid_init）
│   │   ├── panel/                # 面板数据结构管理（init, new, free, disp）
│   │   ├── pauxil/               # 分布式辅助（索引映射、范数等）
│   │   ├── pfact/                # 面板分解（pdfact, pdpan{cr,rl,ll}{N,T}, pdrpan*）
│   │   ├── pgesv/                # 主求解器（pdgesv, pdgesvK2, pdupdate{NN,NT,TN,TT}, pdtrsv）
│   │   └── sdc/                  # ★ SDC 检测模块（校验和、验证、注入、报告）
│   ├── include/                  # 头文件
│   │   ├── hpl.h                 # 主头文件（汇总包含所有子头文件）
│   │   ├── hpl_sdc.h             # ★ SDC 模块头文件（数据结构、函数原型、宏定义）
│   │   ├── hpl_panel.h           # 面板结构体（含 SDC 扩展字段）
│   │   ├── hpl_grid.h            # 处理器网格结构体
│   │   ├── hpl_pgesv.h           # 求解器算法参数结构体
│   │   └── ...                   # 其他模块头文件
│   ├── testing/                  # 测试程序
│   │   ├── ptest/                # 主测试驱动（HPL_pddriver.c → xhpl）
│   │   │   └── HPL.dat           # 测试参数配置文件
│   │   └── sdc_test/             # ★ SDC 独立测试程序（HPL_sdc_test.c → xhpl_sdc_test）
│   ├── makes/                    # 各模块的 Makefile 模板
│   │   ├── Make.sdc              # SDC 模块构建规则
│   │   └── Make.sdc_test         # SDC 测试程序构建规则
│   ├── bin/                      # 可执行文件输出目录
│   ├── lib/                      # 库文件输出目录（libhpl.a）
│   ├── Make.WSL_OpenBLAS         # 构建配置：标准性能测试
│   ├── Make.WSL_OpenMPI          # 构建配置：标准 MPI
│   ├── Make.WSL_SDC_CHECK_ONLY   # ★ 构建配置：SDC 检测模式
│   ├── Make.WSL_SDC_INJECT       # ★ 构建配置：SDC 检测 + 故障注入
│   ├── Make.top                  # 顶层构建逻辑
│   └── Makefile                  # 构建入口
├── doc/                          # 项目文档
│   └── HPL_SDC_Enhancement_Plan.md  # SDC 增强方案设计文档
└── readme.md                     # 本文件
```

---

## 三、HPL 核心算法原理

### 3.1 问题定义

求解 $N \times N$ 稠密线性方程组 $Ax = b$，其中 $A$ 为随机生成的双精度稠密矩阵。算法采用 **带行部分选主元的 LU 分解**（$PA = LU$），将问题分解为：

1. **LU 分解**：$A \rightarrow LU$（带行置换 $P$）
2. **前代 + 回代**：先解 $Ly = Pb$，再解 $Ux = y$
3. **残差验证**：计算 $\|b - Ax\|_\infty / (\varepsilon \cdot (\|A\|_\infty \|x\|_\infty + \|b\|_\infty) \cdot N)$，与阈值比较判定 PASS/FAIL

### 3.2 二维分块数据分布

矩阵 $A$ 和增广矩阵 $[A|b]$ 采用 **2D Block-Cyclic 分布**，映射到 $P \times Q$ 处理器网格：

- 块大小 $NB$：矩阵被切分为 $NB \times NB$ 的数据块
- 分布方式：第 $(i, j)$ 个数据块分配给网格位置 $(i \bmod P, j \bmod Q)$ 的进程
- 本地矩阵维度：$mp = \text{HPL\_numroc}(N, NB, NB, myrow, 0, P)$，$nq = \text{HPL\_numroc}(N+1, NB, NB, mycol, 0, Q)$

处理器网格通过 `HPL_grid_init()`（[hpl/src/grid/HPL_grid_init.c](hpl/src/grid/HPL_grid_init.c)）创建，使用 `MPI_Comm_split` 生成行通信器 `row_comm`、列通信器 `col_comm` 和全局通信器 `all_comm`。

### 3.3 Right-looking LU 分解与 Look-ahead

HPL 采用 **Right-looking** 变体的 LU 分解，配合 **Look-ahead** 技术实现计算与通信重叠：

```
主循环（步长 NB）：
  ┌─────────────────────────────────────────────────────────────┐
  │  1. 面板分解 (HPL_pdfact)                                    │
  │     对当前 NB 列进行 LU 分解（含选主元）                        │
  │                                                              │
  │  2. 面板广播 (HPL_bcast)                                     │
  │     将分解后的 L/U 因子沿列方向广播到所有进程行                  │
  │     支持 6 种非阻塞广播拓扑，允许与步骤 3 重叠                  │
  │                                                              │
  │  3. 尾矩阵更新 (HPL_pdupdate)                                │
  │     A_trail -= L₂ × U  （DGEMM，占总计算量 ~2/3 N³）          │
  │     同时完成广播等待和行交换                                    │
  │                                                              │
  │  4. 循环至下一面板                                             │
  └─────────────────────────────────────────────────────────────┘
```

**Look-ahead**：维护 `depth+1` 个面板缓冲区，当前面板广播时，后续面板可提前分解。由 `HPL_pdgesvK2()`（[hpl/src/pgesv/HPL_pdgesvK2.c](hpl/src/pgesv/HPL_pdgesvK2.c)）实现，是性能最优路径。

### 3.4 关键函数调用链

```
main (HPL_pddriver.c)
  └─ HPL_pdinfo()                    // 读取 HPL.dat 参数
  └─ HPL_grid_init()                 // 创建 P×Q 处理器网格
  └─ for each (N, NB, P, Q, ...) 参数组合:
       └─ HPL_pdtest()               // 执行单次测试
            ├─ HPL_pdmatgen()         // 生成随机矩阵 [A|b]
            ├─ HPL_pdgesv()           // ★ 求解入口
            │    └─ HPL_pdgesvK2()    // 带 look-ahead 的主循环
            │         ├─ HPL_pdpanel_new/init()  // 创建面板
            │         ├─ HPL_pdfact()             // 面板 LU 分解
            │         │    └─ HPL_pdpancrN/rlN()  // Crout/Right-looking 变体
            │         ├─ HPL_binit/bcast/bwait()  // 面板广播
            │         └─ HPL_pdupdate()            // 尾矩阵 DGEMM 更新
            │              └─ HPL_pdupdateNN/NT/TN/TT()
            │                   ├─ HPL_pdlaswp01N()  // 分布式行交换
            │                   ├─ HPL_dtrsm()        // 三角求解
            │                   └─ HPL_dgemm()        // 矩阵乘更新
            ├─ HPL_pdtrsv()           // 上三角回代求解
            ├─ HPL_pdmatgen()         // 重新生成矩阵（用于验证）
            └─ 残差检查 → PASSED / FAILED
```

### 3.5 面板广播拓扑

面板广播沿处理器网格的**列方向**进行（`row_comm`），支持 6 种非阻塞拓扑（[hpl/src/comm/HPL_bcast.c](hpl/src/comm/HPL_bcast.c)）：

| 编号 | 拓扑 | 描述 |
|------|------|------|
| 0 | `1RING` | 单向环 |
| 1 | `1RING_M` | 修正单向环 |
| 2 | `2RING` | 双向环 |
| 3 | `2RING_M` | 修正双向环（推荐） |
| 4 | `BLONG` | 长消息拓扑 |
| 5 | `BLONG_M` | 修正长消息拓扑 |

非阻塞广播返回 `HPL_KEEP_TESTING` 表示未完成，允许在等待期间执行尾矩阵更新计算（计算-通信重叠）。

### 3.6 关键数据结构

| 结构体 | 文件 | 核心字段 |
|--------|------|----------|
| `HPL_T_grid` | hpl_grid.h | `nprow, npcol, myrow, mycol, row_comm, col_comm, all_comm` |
| `HPL_T_palg` | hpl_pgesv.h | `btopo, depth, pfact, pffun, rffun, upfun, fswap` |
| `HPL_T_pmat` | hpl_pgesv.h | `n, nb, A, ld, mp, nq` |
| `HPL_T_panel` | hpl_panel.h | `A, L1, L2, U, DPIV, jb, mp, nq, prow, pcol` |

---

## 四、SDC 检测增强模块

### 4.1 设计思想：基于加权校验和的 ABFT

在高并发、大模型及百万核超算集群中，宇宙射线或硬件静默故障极易引发寄存器或内存比特翻转（Bit Flip）。传统 HPL 仅在全流程结束后通过残差 $\|Ax-b\|_\infty$ 检验正确性，若发生 SDC，无法定位出错阶段与出错节点。

HPL SDC 模块采用 **ABFT（Algorithm-Based Fault Tolerance，算法级容错）**，通过对矩阵列打"数值指纹"（校验和），利用线性代数运算与校验和运算的同构性进行实时监测。

**加权校验和**：如果采用均匀权值（全 1 校验和），当矩阵同一列中发生一处 $+e$ 另一处 $-e$ 的复合错误时，和保持不变，造成漏报。为实现位置敏感的故障捕获，系统对第 $i$ 行赋以 2 的幂次加权：

$$w[i] = 2^{(i \bmod 16)}$$

采用对 16 取模的窗口（`HPL_SDC_WEIGHT_WINDOW = 16`），既保证了相邻行权值各不相同、极低碰撞率，又彻底避免了 64 位双精度浮点指数在幂次过大时发生的精度溢出或下溢。

列校验和公式为：

$$CS[j] = \sum_{i=0}^{m-1} w[i] \times A[i, j]$$

### 4.2 四个检测层级

```mermaid
graph TD
    A[HPL_pdgesvK2 主循环求解] --> B[步骤 1: HPL_pdfact 面板分解]
    B --> C[步骤 2: 计算 owner 列广播缓冲区 CS_bcast]
    C --> D[步骤 3: MPI_Allreduce MAX 全局传播参考值]
    D --> E[步骤 4: HPL_bcast 面板全局广播]
    E --> F[步骤 5: HPL_sdc_verify_checksum 广播后验证指纹]
    F -->|发现偏差 dev > 1e-10| G[HPL_sdc_log_fault 记录到节点日志]
    F -->|正常通过| H[步骤 6: HPL_pdupdate 尾矩阵 DGEMM 更新]
    H --> I[周期性/增量校验和验证 CS_trail]
    I --> J[全流程结束: HPL_sdc_report_and_aggregate 聚合输出定位报告]
```

#### L1：尾矩阵增量校验和（主力模块，占计算量 ~2/3 N³）

尾矩阵更新公式 $A_{\text{trail}} \leftarrow A_{\text{trail}} - L_2 \times U$。根据线性代数分配律，校验和的增量更新仅需：

$$CS_{\text{trail}}^{\text{new}}[j] = CS_{\text{trail}}^{\text{old}}[j] - \sum_{k=0}^{jb-1} CS_{L_2}[k] \times U[k, j]$$

其中 $CS_{L_2}[k] = \sum_i w[i] \times L_2[i][k]$。

- 增量更新开销：$O(mp \cdot jb + jb \cdot nn)$
- 主 DGEMM 开销：$O(mp \cdot jb \cdot nn)$
- **开销比**：$\approx 1/\min(mp, nn) < 0.1\%$

实现在 [HPL_pdupdateNN.c](hpl/src/pgesv/HPL_pdupdateNN.c) 等四个 update 文件中，dgemm 后调用 `HPL_sdc_update_trail_checksum()`。

#### L2：面板广播完整性检验（当前主力生效模块）

面板广播（`HPL_bcast`）将当前列主元和 $L$ 因子发往全网格进程。一旦广播损坏，错误将在后续尾矩阵更新中污染全局。

- **广播前指纹构建**：在 [HPL_pdgesvK2.c](hpl/src/pgesv/HPL_pdgesvK2.c) 中，只有当前拥有真实面板数据的所有者列进程（`mycol == icurcol`）计算校验和 `cs_bcast`；非所有者进程赋为 `0.0`。
- **零开销参考值同步**：调用 `MPI_Allreduce(..., MPI_MAX, comm)`。由于校验和实际幅值为正，`MPI_MAX` 在无额外握手开销下将合法参考值 `cs_ref` 同步至所有进程。
- **广播后接收验证**：所有进程执行 `HPL_bwait()` 后，立即重算校验和 `cs_recv`，根据相对偏差阈值断言：

$$\frac{|CS_{\text{recv}} - CS_{\text{ref}}|}{\max(|CS_{\text{ref}}|, 1.0)} > 1.0 \times 10^{-10} \implies \text{检测到通信 SDC！}$$

#### L3：面板分解校验

在 `HPL_pdfact()`（[hpl/src/pfact/HPL_pdfact.c](hpl/src/pfact/HPL_pdfact.c)）完成分解后，立即计算面板列校验和 $CS_{\text{panel}}[k] = \sum_i w[i] \times L_2[i][k]$，填充 `PANEL->CS_PANEL`，为后续广播验证提供基准。

#### L4：回代求解统计检测

在 `HPL_pdtrsv()`（[hpl/src/pgesv/HPL_pdtrsv.c](hpl/src/pgesv/HPL_pdtrsv.c)）完成后，对解向量 $X$ 执行两层检测：
1. **NaN/Inf 检测**：检查 $cs_x = \sum XR[i]$ 是否为 NaN 或溢出
2. **6-sigma 离群值检测**：统计超过 6 倍标准差的离群值数量

### 4.3 核心函数说明

#### 校验和计算（[HPL_sdc_checksum.c](hpl/src/sdc/HPL_sdc_checksum.c)）

| 函数 | 功能 | 复杂度 |
|------|------|--------|
| `HPL_sdc_init_weights(w, n)` | 初始化权值向量 $w[i] = 2^{i \bmod 16}$ | $O(n)$ |
| `HPL_sdc_col_checksum(A, lda, m, n, w)` | 计算矩阵列校验和 $\sum_j \sum_i w[i] A[i][j]$ | $O(mn)$ |
| `HPL_sdc_panel_checksum(A, lda, m, n, w, cs)` | 计算面板每列校验和 $cs[k] = \sum_i w[i] A[i][k]$ | $O(mn)$ |
| `HPL_sdc_update_trail_checksum(...)` | 增量更新尾矩阵校验和 | $O(mp \cdot jb + jb \cdot nn)$ |
| `HPL_sdc_compute_bcast_checksum(...)` | 计算广播缓冲区（L2+L1+DPIV）校验和 | $O(ml2 \cdot jb + jb^2)$ |

#### 验证逻辑（[HPL_sdc_verify.c](hpl/src/sdc/HPL_sdc_verify.c)）

| 函数 | 功能 |
|------|------|
| `HPL_sdc_verify_checksum(cs_expected, cs_computed, threshold)` | 相对阈值比较，返回 1=SDC / 0=正常 |
| `HPL_sdc_verify_panel(A, lda, m, n, w, cs_expected, threshold)` | 逐列重算面板校验和并比对 |
| `HPL_sdc_verify_trailing(A, lda, m, n, cs_expected, w, threshold)` | 逐列重算尾矩阵校验和并比对 |

**判定逻辑**：

```
deviation = |cs_computed - cs_expected|
denom     = max(|cs_expected|, 1.0)    // 防除零

若 deviation/denom > threshold → 返回 1（SDC 故障）
否则                          → 返回 0（正常）
```

#### 故障注入模型（[HPL_sdc_inject.c](hpl/src/sdc/HPL_sdc_inject.c)，需 `-DHPL_SDC_INJECT`）

| 函数 | 故障模型 | 描述 |
|------|---------|------|
| `HPL_sdc_inject_bitflip(A, index, bit_pos)` | 单位翻转 | 翻转 `A[index]` 的指定比特位 |
| `HPL_sdc_inject_random(A, n, rate)` | 随机替换 | 以 `rate` 概率将元素替换为随机值 |
| `HPL_sdc_inject_at(A, index, mode, value)` | 精确注入 | mode 0=替换, 1=漂移, 2=零值卡死, 3=符号翻转, 4=NaN, 5=Inf |

#### 故障日志与聚合报告（[HPL_sdc_report.c](hpl/src/sdc/HPL_sdc_report.c)）

| 函数 | 功能 |
|------|------|
| `HPL_sdc_log_init(log, comm)` | 初始化日志，获取物理节点名 |
| `HPL_sdc_log_fault(log, rank, row, col, type, step, ...)` | O(1) 链表插入故障记录 |
| `HPL_sdc_report_and_aggregate(log, comm, rank)` | MPI 聚合 + 输出报告 |
| `HPL_sdc_log_cleanup(log)` | 释放故障链表 |

### 4.4 关键数据结构

```c
/* SDC 故障类型枚举 (hpl_sdc.h) */
typedef enum {
   HPL_SDC_FAULT_PANEL_BCAST,    // 面板广播损坏
   HPL_SDC_FAULT_PANEL_FACT,     // 面板分解损坏
   HPL_SDC_FAULT_TRAIL_UPDATE,   // 尾矩阵更新损坏
   HPL_SDC_FAULT_BACK_SOLVE,     // 回代求解损坏
   HPL_SDC_FAULT_BROADCAST,      // 通信层广播损坏
   HPL_SDC_FAULT_UNKNOWN         // 未知类型
} HPL_T_SDC_FAULT_TYPE;

/* 单条故障记录（链表节点） */
typedef struct HPL_S_SDC_FAULT {
   int                  mpi_rank;      // MPI 全局秩
   int                  grid_row;      // 处理器网格行坐标
   int                  grid_col;      // 处理器网格列坐标
   char                 node_name[64]; // 物理节点主机名
   HPL_T_SDC_FAULT_TYPE fault_type;    // 故障类型
   int                  step;          // LU 分解步号
   int                  global_row;    // 全局矩阵行索引
   int                  global_col;    // 全局矩阵列索引
   double               cs_expected;   // 期望校验和
   double               cs_computed;   // 实际校验和
   double               deviation;     // 偏差量
   struct HPL_S_SDC_FAULT * next;      // 链表指针
} HPL_T_SDC_FAULT;

/* 故障日志（每个进程维护一个） */
typedef struct HPL_S_SDC_LOG {
   HPL_T_SDC_FAULT * head;     // 链表头
   int               count;    // 故障计数
   int               enabled;  // 启用标志
   char              node_name[64]; // 物理节点主机名
} HPL_T_SDC_LOG;
```

### 4.5 面板结构体中的 SDC 扩展字段

在 `HPL_T_panel`（[hpl/include/hpl_panel.h](hpl/include/hpl_panel.h)）中新增：

```c
#ifdef HPL_SDC_CHECK
   double  * CS_PANEL;   // jb 个面板列校验和
   double  * CS_WEIGHTS; // mp 个权值向量
   double    cs_bcast;   // 广播缓冲区校验和
   double  * CS_TRAIL;   // nq 个尾矩阵列校验和
   int       sdc_step;   // 验证步数计数器
#endif
```

在 [HPL_pdpanel_init.c](hpl/src/panel/HPL_pdpanel_init.c) 中分配并初始化，在 [HPL_pdpanel_free.c](hpl/src/panel/HPL_pdpanel_free.c) 中释放。

### 4.6 编译宏控制

| 宏定义 | 作用 |
|--------|------|
| `HPL_SDC_CHECK` | 启用 SDC 检测的总开关，所有 SDC 代码均在 `#ifdef HPL_SDC_CHECK` 下 |
| `HPL_SDC_BCAST_VERIFY` | 启用面板广播校验和验证（默认 1） |
| `HPL_SDC_TRAIL_VERIFY` | 启用尾矩阵校验和验证（默认 0） |
| `HPL_SDC_INJECT` | 启用故障注入功能（仅测试用） |
| `HPL_SDC_THRESHOLD` | 校验和比对相对阈值（默认 `1.0e-10`） |
| `HPL_SDC_WEIGHT_WINDOW` | 权值窗口大小（默认 16） |

不启用 `HPL_SDC_CHECK` 时，所有 SDC 代码被编译器完全消除，**零开销**。

### 4.7 故障报告输出示例

```
===== SDC FAULT REPORT =====
Total faults detected: 42

--- Fault #1 ---
  Type:        TRAIL_UPDATE
  Step:        1536
  MPI Rank:    37
  Grid Pos:    (row=3, col=5)
  Node Name:   compute-node-042
  Location:    global A[294912, 295104]
  Deviation:   3.72e-08
  Severity:    HIGH

--- Summary by Node ---
  compute-node-042:  15 faults
  compute-node-017:  12 faults

--- Summary by Fault Type ---
  TRAIL_UPDATE: 28, PANEL_BCAST: 8, PANEL_FACT: 4

RECOMMENDATION: Replace nodes with >10 faults:
  compute-node-042, compute-node-017
==============================
```

---

## 五、编译构建说明

### 5.1 依赖

- **MPI**：OpenMPI（`mpicc` 编译器包装器）
- **BLAS**：OpenBLAS（`-lopenblas`）
- **操作系统**：Linux / WSL（Windows Subsystem for Linux）
- **编译器**：GCC（`mpicc` 包装）

### 5.2 构建配置

本项目提供四种构建配置，通过 `Make.<arch>` 文件定义：

| 配置名 | 编译宏 | 用途 |
|--------|--------|------|
| `WSL_OpenBLAS` | 无 SDC 宏 | 标准 HPL 性能测试 |
| `WSL_OpenMPI` | 无 SDC 宏 | 标准 MPI 构建 |
| `WSL_SDC_CHECK_ONLY` | `-DHPL_SDC_CHECK -DHPL_SDC_BCAST_VERIFY=1 -DHPL_SDC_TRAIL_VERIFY=1` | SDC 检测模式 |
| `WSL_SDC_INJECT` | 上述 + `-DHPL_SDC_INJECT` | SDC 检测 + 故障注入 |

### 5.3 构建步骤

```bash
# 进入 HPL 目录
cd hpl

# 标准构建（性能测试）
make arch=WSL_OpenBLAS

# SDC 检测模式构建
make arch=WSL_SDC_CHECK_ONLY

# SDC 检测 + 故障注入构建
make arch=WSL_SDC_INJECT

# 清理
make clean arch=WSL_SDC_INJECT
```

构建产物：
- 库文件：`lib/<arch>/libhpl.a`
- 可执行文件：`bin/<arch>/xhpl`（主程序）、`bin/<arch>/xhpl_sdc_test`（SDC 测试）

### 5.4 构建系统组织

```
Makefile (入口)
  └─ Make.top (顶层逻辑)
       ├─ startup: 创建目录结构 + 符号链接 Make.inc
       ├─ refresh: 复制 makes/Make.* → 各子目录 Makefile
       └─ build:
            ├─ build_src: 编译 auxil → blas → comm → grid → panel → pauxil → pfact → pgesv → sdc
            └─ build_tst: 编译 matgen → timer → pmatgen → ptimer → ptest → sdc_test
```

每个子目录通过符号链接 `Make.inc → Make.<arch>` 获取编译配置。

---

## 六、运行测试说明

### 6.1 标准 HPL 性能测试

```bash
# 运行（以 4 进程为例）
cd hpl
mpirun -np 4 ./bin/WSL_OpenBLAS/xhpl
```

程序从当前目录读取 `HPL.dat` 配置文件（模板见 [hpl/testing/ptest/HPL.dat](hpl/testing/ptest/HPL.dat)），遍历所有参数组合执行测试。

### 6.2 HPL.dat 关键参数

```
Ns              矩阵维度列表（如 2000 4000 8000）
NBs             分块大小列表（如 64 192）
PMAP            进程映射方式（0=行主序, 1=列主序）
Ps / Qs         处理器网格 P×Q（如 P=1,2  Q=4,2）
threshold       残差阈值（如 16.0）
PFACTs          面板分解算法（0=Left, 1=Crout, 2=Right）
RFACTs          递归分解算法（0=Left, 1=Crout, 2=Right）
BCASTs          广播拓扑（0=1rg, 1=1rM, 2=2rg, 3=2rM, 4=Lng, 5=LnM）
DEPTHs          Look-ahead 深度（≥0）
SWAP            行交换策略（0=bin-exch, 1=long, 2=mix）
NBMIN           递归停止条件（≥1）
NDIVs           递归面板数
Equilibration   均衡化（0=否, 1=是）
```

### 6.3 SDC 独立测试

```bash
# SDC 单元测试（需 WSL_SDC_INJECT 构建）
mpirun -np 4 ./bin/WSL_SDC_INJECT/xhpl_sdc_test
```

测试包含 7 组：

| 测试组 | 内容 |
|--------|------|
| Group 1 | 校验和计算正确性（权值初始化、列校验和、面板校验和） |
| Group 2 | 验证逻辑（真阴性/真阳性、阈值边界测试） |
| Group 3 | 6 种故障注入模型（位翻转、随机替换、零值卡死、小漂移、符号翻转、值替换） |
| Group 4 | 故障日志记录与 MPI 聚合报告 |
| Group 5 | 增量尾矩阵校验和更新 vs 全量重算交叉验证 |
| Group 6 | 广播校验和（L2+L1+DPIV 完整性） |
| Group 7 | 检测延迟模拟（注入后立即检测） |

### 6.4 SDC 运行时故障注入

在 `HPL_SDC_INJECT` 构建下运行主程序时，可通过环境变量在指定列注入故障：

```bash
export HPL_SDC_INJECT_COL=5       # 在第 5 列注入
export HPL_SDC_INJECT_VAL=999.0   # 注入值 999.0
mpirun -np 4 ./bin/WSL_SDC_INJECT/xhpl
```

---

## 七、关键参数与调优建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| $N$ | 接近内存 80% 容量 | 如 56000（视内存大小调整） |
| $NB$ | 192 | 常用大块尺寸，平衡计算效率与通信频率 |
| $P \times Q$ | 等于进程总数 | $P$ 推荐为 2 的幂且略小于 $Q$ |
| BCAST | 3（2rM） | 修正双向环，通常最优 |
| DEPTH | 1 | Look-ahead 深度，1 为常用值 |
| PFACT | 1（Crout） | Crout 变体通常性能最佳 |

**性能优化原则**：
- 增大 $N$ 可提高利用率，但不得超过物理内存
- $NB$ 过大会增加单步计算量但不利于广播并行，过小则通信频繁
- $P \times Q$ 应使每个进程有足够的本地数据（$mp, nq \gg NB$）

---

## 八、开销分析

| 操作 | 额外计算量 | 额外通信量 |
|------|-----------|-----------|
| 面板校验和（L3） | $O(mp \times jb)$ | 无 |
| 广播校验和（L2） | $O(mp \times jb)$ | 1 个 double 的 Allreduce |
| 尾矩阵增量更新（L1） | $O(mp \times jb + jb \times nn)$ | 无 |
| 回代检测（L4） | $O(n)$ | 无 |
| **总额外开销** | **$\sim O(N^2)$ vs 主计算 $O(N^3)$** | **可忽略** |

**相对开销 $\approx O(1/N)$**，当 $N$ 很大时趋近于零。不启用 `HPL_SDC_CHECK` 时开销严格为零（所有代码被预处理器消除）。
