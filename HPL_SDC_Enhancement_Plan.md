# HPL 静默数据损坏（SDC）检测增强方案

## Context

HPL（High Performance Linpack）是国际标准的高性能计算机浮点性能基准测试工具，通过求解大规模稠密线性方程组 Ax=b 测试系统浮点性能。经过对 HPL 源码的全面分析，**HPL 当前不包含任何运行时容错或 SDC 检测机制**——没有 ABFT、校验和、编码冗余。唯一的验证是测试驱动 `HPL_pdtest.c` 中求解完成后的残差检查 `||b-Ax||`，无法定位故障发生的时间和位置。

本方案旨在为 HPL 设计一套轻量级 SDC 检测增强系统，在不显著影响性能的前提下，实现对静默数据损坏的实时检测与定位。

---

## 一、HPL 现有容错能力边界分析

### 1.1 现有机制：几乎为零

| 维度 | 现状 |
|------|------|
| **检测覆盖率** | 0%——求解过程中无任何数据完整性检查 |
| **性能开销** | 无（因为没有检测） |
| **故障定位能力** | 仅能在求解结束后通过残差判断"是否出错"，无法定位到具体步骤/操作 |
| **漏报情况** | 所有不导致发散的发散型 SDC（如小幅度漂移、符号翻转、个别位翻转）均无法检测 |
| **误报情况** | 不存在检测机制，故无误报 |
| **MPI 错误处理** | MPI 调用返回码大多被忽略 |
| **内存分配检查** | 仅有 `malloc` 返回 NULL 时的 `HPL_pabort` |

### 1.2 脆弱性排序

| 风险等级 | 操作 | 源码位置 | 原因 |
|---------|------|---------|------|
| **极高** | dgemm 尾矩阵更新 | `HPL_pdupdateNN/NT/TN/TT.c` | 占总计算量 ~2/3 N³；损坏传播至后续所有步骤 |
| **高** | 面板分解 | `HPL_pdpan{ll,cr,rl}{N,T}.c`、`HPL_pdrpan*.c` | L/U 损坏污染所有后续更新；主元错误是灾难性的 |
| **高** | 面板广播 | `HPL_bcast.c`、`HPL_copyL.c` | 损坏面板数据发送到所有进程列，单点故障全局扩散 |
| **中** | dtrsm 三角求解 | `HPL_pdupdate*.c` 内部 | 三角求解误差向前传播 |
| **中** | 回代求解 | `HPL_pdtrsv.c` | 最终解向量计算，错误直接影响结果 |
| **低** | 选主 | `HPL_pdmxswp.c` | 错误选主导致数值不稳定，但可被后续残差检查发现 |

---

## 二、SDC 检测增强方案设计

### 设计原则：列校验和 ABFT + 分层选择性验证

采用**列校验和编码**作为核心机制。对矩阵块 A（m×n），维护校验和向量：
```
cs[j] = Σ w[i] * A[i][j],  i=0..m-1, j=0..n-1
```
其中 `w[i] = 2^(i mod 16)` 为 2 的幂次权值（用移位代替乘法，模窗口 16 防溢出）。

### 2.1 核心算法：分层校验和 ABFT

#### 第一层（关键）：尾矩阵 dgemm 更新 —— 增量校验和

**原理**：尾矩阵更新 `A_new = A_old - L2 * U` 后，校验和满足：
```
cs_trail_new[j] = cs_trail_old[j] - Σ_k cs_L2[k] * U[k][j]
```
其中 `cs_L2[k] = Σ_i w[i] * L2[i][k]`。

**实现**：在每次 dgemm 块更新后，同步执行一次小规模校验和更新（`O(mp*jb + jb*nn)` vs 主 dgemm 的 `O(mp*jb*nn)`），开销比约 `1/min(mp,nn)`，可忽略不计。

**验证频率**：每 K=8 步执行一次全量重算交叉验证，中间步骤仅做增量传播。

#### 第二层（重要）：面板广播 —— 附加校验和

在面板拷贝到广播缓冲区时（`HPL_copyL.c`），计算整个缓冲区（L2 + L1 + DPIV）的校验和，附加到广播数据末尾。接收端在 `HPL_bwait()` 后重算校验和并比对。

**开销**：计算 `O(mp*jb)` 一次遍历；通信仅增加 1 个 double，可忽略。

#### 第三层（中等）：面板分解 —— 分解前后校验

在 `HPL_pdfact()` 中，分解前计算面板列校验和，分解后验证 L*U 乘积的校验和与原始一致。由于选主会置换行，需跟踪置换调整权值。

**简化方案**：仅在 `HPL_pdfact()` 顶层入口/出口验证，不进入递归内部。

#### 第四层（基础）：回代求解 —— 终态校验

在 `HPL_pdtrsv.c` 完成后，计算解向量 X 的校验和，与通过 A、b 属性推导的期望值交叉比对。

### 2.2 选择性验证策略

| 层级 | 操作 | 方法 | 频率 | 开销占比 |
|------|------|------|------|---------|
| L1 | dgemm 尾矩阵更新 | 增量校验和 + 周期全量验证 | 每步增量，每 K=8 步全量 | <0.1% |
| L2 | 面板广播 | 校验和附加 + 接收端验证 | 每次广播 | ~0.05% |
| L3 | 面板分解 | 分解前后校验 | 每个面板 | ~0.1% |
| L4 | 回代求解 | 解向量校验 | 一次 | 可忽略 |

**自适应策略**：
- 前 10 步：每步全量验证（建立基线）
- 10-100 步：每 K=8 步验证
- 100 步后：每 K=16 步验证
- 检测到 SDC 后：回退到每步验证持续 20 步

### 2.3 阈值与误报控制

校验和比对使用相对阈值：
```
|cs_computed - cs_expected| / max(|cs_expected|, 1.0) < ε_threshold
```
推荐 `ε_threshold = 1e-10`（双精度）。考虑到浮点非结合性导致的累积舍入误差，阈值需随步数增长：
```
ε_effective = ε_threshold * sqrt(N/NB) * ε_machine
```

---

## 三、代码级修改计划

### Task 1：创建 SDC 检测模块（新文件）

**新建文件**：
- `hpl/include/hpl_sdc.h` —— SDC 头文件，函数原型与数据结构
- `hpl/src/sdc/HPL_sdc_checksum.c` —— 核心校验和计算
- `hpl/src/sdc/HPL_sdc_verify.c` —— 验证与比对逻辑
- `hpl/src/sdc/HPL_sdc_inject.c` —— 故障注入（测试用）
- `hpl/src/sdc/HPL_sdc_report.c` —— 节点级故障日志、聚合与报告输出
- `hpl/makes/Make.sdc` —— 构建规则

**核心函数**：
```c
// 校验和计算
void   HPL_sdc_init_weights(double *w, int n);
void   HPL_sdc_panel_checksum(HPL_T_panel *PANEL, double *cs, int m, int n);
void   HPL_sdc_trail_checksum(HPL_T_panel *PANEL, double *cs, int m, int n);

// 增量校验和更新（关键：与 dgemm 同步）
void   HPL_sdc_update_trail_checksum(HPL_T_panel *PANEL, double *cs_trail,
                                      const double *L2, int ldl2,
                                      const double *U, int ldu,
                                      int mp, int jb, int nn);

// 验证函数
int    HPL_sdc_verify_panel(HPL_T_panel *PANEL);
int    HPL_sdc_verify_broadcast(HPL_T_panel *PANEL);
int    HPL_sdc_verify_trailing(HPL_T_panel *PANEL, double *cs_trail);

// 广播校验和
void   HPL_sdc_compute_broadcast_checksum(HPL_T_panel *PANEL, double *cs);

// 故障注入（仅测试）
void   HPL_sdc_inject_at(HPL_T_pmat *A, int row, int col, int mode, int severity);
void   HPL_sdc_inject_random(HPL_T_pmat *A, double rate, int mode);
```

### Task 2：扩展 HPL_T_panel 数据结构

**修改文件**：`hpl/include/hpl_panel.h`（第 60-100 行）

在 `HPL_T_panel` 结构体中，`#ifdef HPL_CALL_VSIPL` 之前添加：
```c
#ifdef HPL_SDC_CHECK
   double  * CS_PANEL;     /* jb 个面板列校验和 */
   double  * CS_TRAIL;     /* nq 个尾矩阵列校验和 */
   double  * CS_WEIGHTS;   /* mp 个权值向量 */
   double    cs_bcast;      /* 广播缓冲区校验和 */
   int       sdc_step;      /* 自适应验证步数计数器 */
   int       sdc_verified;  /* 面板校验通过标志 */
#endif
```

### Task 3：面板初始化/释放时分配/回收 SDC 工作空间

**修改文件**：
- `hpl/src/panel/HPL_pdpanel_init.c` —— 在末尾添加 `#ifdef HPL_SDC_CHECK` 分配 `CS_PANEL`、`CS_TRAIL`、`CS_WEIGHTS` 并初始化权值
- `hpl/src/panel/HPL_pdpanel_disp.c` —— 添加 SDC 工作空间释放
- `hpl/src/panel/HPL_pdpanel_free.c` —— 同上

### Task 4：主循环集成 SDC 检测

**修改文件**：`hpl/src/pgesv/HPL_pdgesvK2.c`

在三个关键位置插入检测代码：

1. **`HPL_pdfact()` 之后、广播之前**（第 196 行后）：计算面板校验和 + 广播校验和
2. **`HPL_bwait()` 之后**（第 202 行后）：验证广播完整性
3. **尾矩阵更新后**（第 201 行后）：增量校验和更新 + 周期性全量验证

同时对 `hpl/src/pgesv/HPL_pdgesv0.c` 做类似修改。

### Task 5：尾矩阵更新中嵌入增量校验和

**修改文件**：
- `hpl/src/pgesv/HPL_pdupdateNN.c` —— 在每个 dgemm 块更新后调用 `HPL_sdc_update_trail_checksum()`
- `hpl/src/pgesv/HPL_pdupdateNT.c` —— 同上
- `hpl/src/pgesv/HPL_pdupdateTN.c` —— 同上
- `hpl/src/pgesv/HPL_pdupdateTT.c` —— 同上

### Task 6：广播校验和

**修改文件**：`hpl/src/comm/HPL_copyL.c` —— 在面板拷贝完成后计算广播缓冲区校验和

### Task 7：回代求解校验

**修改文件**：`hpl/src/pgesv/HPL_pdtrsv.c` —— 求解完成后计算解向量校验和

### Task 8：面板分解校验

**修改文件**：`hpl/src/pfact/HPL_pdfact.c` —— 在 `rffun` 调用前后计算校验和并验证

### Task 9：构建系统集成

**修改文件**：
- `hpl/Make.top` 或架构 Makefile —— 添加 `-DHPL_SDC_CHECK` 编译选项
- `hpl/makes/Make.sdc` —— 新模块构建规则
- `hpl/Makefile` —— 包含 SDC 模块

### Task 10：测试驱动集成

**修改文件**：`hpl/testing/ptest/HPL_pdtest.c` —— 在残差检查后添加 SDC 检测统计报告

---

## 四、开销估算

| 组件 | 额外内存 | 计算开销 | 通信开销 |
|------|---------|---------|---------|
| 面板校验和 | O(jb) doubles/面板 | O(mp*jb)/面板分解 | 无 |
| 尾矩阵增量校验和 | O(nq) doubles | O(mp*jb + jb*nn)/更新块 | 无 |
| 全量验证（每 K 步） | 无额外 | O(mp*nq)/验证 | 无 |
| 广播校验和 | 1 double/面板 | O(mp*jb)/广播 | 1 double 附带 |
| **总计** | **~4000 doubles ≈ 32KB/进程** | **增量 <0.1%，全量 1-3%** | **可忽略** |

（基于典型配置 N=100000, P=Q=100, NB=192, mp≈1000, nq≈1000, jb=192）

---

## 五、故障注入测试框架

支持 6 种故障模型注入：

| 故障模型 | 描述 | 检测难度 |
|---------|------|---------|
| 单位翻转 | 单个 double 的某一位翻转 | 中 |
| 零值卡死 | 值静默变为 0 | 高 |
| 随机替换 | 值被随机数据替换 | 低 |
| 小幅度漂移 | 值增加 ε（如 1e-15） | 极高 |
| 主元损坏 | 枢轴索引被篡改 | 关键 |
| 广播损坏 | MPI 传输中面板数据损坏 | 高 |

测试流程：编译开启 `-DHPL_SDC_CHECK -DHPL_SDC_INJECT` → 无注入基线运行验证零误报 → 逐模型注入验证检出率 → 测量检测延迟（注入到检出的步数）→ 性能对比测试。

---

## 六、风险与注意事项

1. **阈值调优**：过紧导致浮点舍入引发误报，过松漏报小幅度 SDC。推荐 `ε=1e-10` 并随步数自适应调整
2. **选主对校验和的影响**：行交换会置换权值，需跟踪置换或尾矩阵使用置换不变校验和（均匀权值）
3. **递归面板分解**：仅在 `HPL_pdfact()` 顶层验证，不进入 `HPL_pdrpan*` 递归内部
4. **非阻塞广播**：6 种广播拓扑完成语义不同，校验和验证必须在 `HPL_SUCCESS` 确认后进行
5. **浮点非结合性**：增量校验和与全量重算可能因舍入顺序不同而有微小差异，阈值需覆盖此误差

---

## 七、关键文件清单

| 文件 | 修改类型 | 用途 |
|------|---------|------|
| `hpl/include/hpl_panel.h` | 添加字段 | SDC 校验和存储 |
| `hpl/include/hpl_sdc.h` | **新建** | SDC 函数原型 |
| `hpl/src/sdc/HPL_sdc_checksum.c` | **新建** | 核心校验和计算 |
| `hpl/src/sdc/HPL_sdc_verify.c` | **新建** | 验证逻辑 |
| `hpl/src/sdc/HPL_sdc_inject.c` | **新建** | 故障注入测试 |
| `hpl/src/panel/HPL_pdpanel_init.c` | 添加分配 | SDC 工作空间 |
| `hpl/src/panel/HPL_pdpanel_disp.c` | 添加释放 | SDC 工作空间 |
| `hpl/src/panel/HPL_pdpanel_free.c` | 添加释放 | SDC 工作空间 |
| `hpl/src/pgesv/HPL_pdgesvK2.c` | 添加检测调用 | 主循环集成 |
| `hpl/src/pgesv/HPL_pdgesv0.c` | 添加检测调用 | 无前瞻版本集成 |
| `hpl/src/pgesv/HPL_pdupdateNN.c` | 添加增量更新 | 尾矩阵校验和 |
| `hpl/src/pgesv/HPL_pdupdateNT.c` | 添加增量更新 | 同上 |
| `hpl/src/pgesv/HPL_pdupdateTN.c` | 添加增量更新 | 同上 |
| `hpl/src/pgesv/HPL_pdupdateTT.c` | 添加增量更新 | 同上 |
| `hpl/src/comm/HPL_copyL.c` | 添加校验和 | 广播缓冲区校验 |
| `hpl/src/pfact/HPL_pdfact.c` | 添加校验 | 面板分解校验 |
| `hpl/src/pgesv/HPL_pdtrsv.c` | 添加校验 | 解向量校验 |
| `hpl/testing/ptest/HPL_pdtest.c` | 添加统计 | SDC 检测报告 |
| `hpl/makes/Make.sdc` | **新建** | 构建规则 |
| `hpl/src/sdc/HPL_sdc_report.c` | **新建** | 节点级故障日志、聚合与报告输出 |

## 八、节点级 SDC 故障定位与报告

### 8.1 设计目标

检测到 SDC 后，精确报告故障所在的 **MPI 进程（核）、处理器网格坐标、物理节点主机名**，便于运维快速定位并更换故障节点。

### 8.2 故障记录数据结构

在 `hpl/include/hpl_sdc.h` 中定义：

```c
/* SDC 故障类型枚举 */
typedef enum {
   HPL_SDC_FAULT_PANEL_BCAST,    /* 面板广播损坏 */
   HPL_SDC_FAULT_PANEL_FACT,     /* 面板分解损坏 */
   HPL_SDC_FAULT_TRAIL_UPDATE,   /* 尾矩阵更新损坏 */
   HPL_SDC_FAULT_BACK_SOLVE,     /* 回代求解损坏 */
   HPL_SDC_FAULT_BROADCAST,      /* 通信层广播损坏 */
   HPL_SDC_FAULT_UNKNOWN         /* 未知类型 */
} HPL_T_SDC_FAULT_TYPE;

/* 单条故障记录 */
typedef struct HPL_S_SDC_FAULT {
   int                  mpi_rank;      /* MPI 全局秩 (grid->iam) */
   int                  grid_row;      /* 处理器网格行坐标 (myrow) */
   int                  grid_col;      /* 处理器网格列坐标 (mycol) */
   char                 node_name[64]; /* 物理节点主机名 */
   HPL_T_SDC_FAULT_TYPE fault_type;    /* 故障类型 */
   int                  step;          /* 检测到的 LU 分解步号 */
   int                  global_row;    /* 全局矩阵行索引（如可定位） */
   int                  global_col;    /* 全局矩阵列索引（如可定位） */
   double               cs_expected;   /* 期望校验和 */
   double               cs_computed;   /* 实际校验和 */
   double               deviation;     /* 偏差量 */
   struct HPL_S_SDC_FAULT * next;      /* 链表指针 */
} HPL_T_SDC_FAULT;

/* 故障日志（每个进程维护本地链表） */
typedef struct HPL_S_SDC_LOG {
   HPL_T_SDC_FAULT * head;
   int               count;
   int               enabled;
   char              node_name[64];    /* 物理节点主机名 */
} HPL_T_SDC_LOG;
```

### 8.3 物理节点名获取

SDC 模块初始化时调用 `MPI_Get_processor_name()` 获取主机名：

```c
void HPL_sdc_log_init(HPL_T_SDC_LOG * log, HPL_T_grid * grid) {
   char name[MPI_MAX_PROCESSOR_NAME];
   int  namelen;
   MPI_Get_processor_name(name, &namelen);
   strncpy(log->node_name, name, 63);
   log->node_name[63] = '\0';
   log->head = NULL; log->count = 0; log->enabled = 1;
}
```

### 8.4 故障记录函数

```c
void HPL_sdc_log_fault(
   HPL_T_SDC_LOG * log, HPL_T_grid * grid,
   HPL_T_SDC_FAULT_TYPE type, int step,
   int global_row, int global_col,
   double cs_expected, double cs_computed
);
```

### 8.5 故障聚合与报告输出

所有进程通过 `MPI_Allreduce` 汇总故障总数，rank 0 收集所有故障详情并按节点分组统计：

```c
void HPL_sdc_report_and_aggregate(HPL_T_SDC_LOG * local_log, HPL_T_grid * grid);
```

**报告输出格式**（rank 0 输出到 stdout）：
```
===== SDC FAULT REPORT =====
Total faults detected: 42  (across 5 nodes)

--- Fault #1 ---
  Type:        TRAIL_UPDATE
  Step:        1536
  MPI Rank:    37
  Grid Pos:    (row=3, col=5) in 8x8 grid
  Node Name:   compute-node-042
  Location:    global A[294912, 295104]
  Deviation:   3.72e-08
  Severity:    HIGH

--- Summary by Node ---
  compute-node-042:  15 faults  (ranks 37,38,39,...)
  compute-node-017:  12 faults  (ranks 20,21,...)
  compute-node-088:   8 faults
  compute-node-003:   5 faults
  compute-node-061:   2 faults

--- Summary by Fault Type ---
  TRAIL_UPDATE: 28, PANEL_BCAST: 8, PANEL_FACT: 4, BROADCAST: 2

RECOMMENDATION: Replace nodes with >10 faults:
  compute-node-042, compute-node-017
==============================
```

### 8.6 集成点

| 检测点 | 文件 | 可定位信息 |
|--------|------|------------|
| 广播校验和不匹配 | `HPL_pdgesvK2.c` 第 202 行后 | 接收端 rank + 步号 |
| 尾矩阵校验和偏差 | `HPL_pdupdateNN.c` 等 | 本地 rank + 步号 + 列索引 |
| 面板分解校验失败 | `HPL_pdfact.c` | 分解端 rank + 步号 |
| 回代校验失败 | `HPL_pdtrsv.c` | 求解端 rank |

在主循环结束（`HPL_pdgesvK2.c` 第 227 行前）调用 `HPL_sdc_report_and_aggregate()` 输出汇总报告。

### 8.7 开销分析

- **内存**：每条故障记录 ~160 字节，典型 SDC 率下 <100 条，开销 <16KB
- **计算**：故障记录为链表插入 O(1)；聚合仅在检测结束时执行一次
- **通信**：仅在检测到故障时触发 `MPI_Allreduce`（1 个 int），无故障时零开销

---

## 九、验证方法

1. **编译验证**：`make` 开启/关闭 `-DHPL_SDC_CHECK` 均成功
2. **基线正确性**：无注入时运行结果与原版一致，无误报
3. **检出率测试**：对每种故障模型注入 1000 次，统计检出率（目标 >95%）
4. **性能测试**：对比开启/关闭 SDC 检测的执行时间（目标开销 <3%）
5. **定位精度**：注入到已知位置，验证检测到的步骤与注入点一致
6. **节点定位验证**：注入故障后检查报告中的 MPI rank、grid 坐标、主机名是否与注入位置一致
7. **聚合报告验证**：多节点注入时，rank 0 输出的按节点分组统计正确，更换建议准确
