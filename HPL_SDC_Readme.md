# SDC 检测实现原理（修正版）

## HPL SDC实现原理

### 一、基于加权校验和的ABFT

在高并发、大模型及百万核超算集群中，宇宙射线或硬件静默故障极易引发寄存器或显存/内存比特翻转（Bit Flip）。传统 HPL 仅在全流程结束后通过残差 $||Ax-b||_\infty$ 检验正确性，若发生 SDC，根本无法定位出错阶段与出错节点。

HPL SDC 模块采用 **ABFT（Algorithm-Based Fault Tolerance，算法级容错）**，通过对矩阵列打"数值指纹"（校验和），利用线性代数运算与校验和运算的同构性进行实时监测。

列校验和公式为：
$$CS[j] = \sum_{i=0}^{m-1} w[i] \times A[i, j]$$

其中权重 $w[i] = 2^{(i \bmod 16)}$（面板校验和用加权），或 $w[i] = 1$（尾矩阵和广播缓冲用均匀权值，因选主置换对位置相关权值有破坏）。

---

### 二、检测点架构

#### 检测点 1：面板分解正确性（CS_PANEL）

在 `HPL_pdfact` 分解完成后计算面板 L2 的加权列校验和，存入 CS_PANEL。

#### 检测点 2：面板广播完整性（BCAST_VERIFY）—— 主力检测模块

```
面板分解 (HPL_pdfact)
    ↓
计算广播缓冲校验和 cs_bcast (L2 + L1 + DPIV，均匀求和)
    ↓
MPI_Bcast 传播参考值到所有进程
    ↓
面板广播 (HPL_bcast) ← 可能引入SDC
    ↓
重算校验和 cs_recv
    ↓
比较 cs_bcast vs cs_recv → 检测通信损坏
```

**注意**：参考值同步使用 `MPI_Bcast`（非文档早期版本的 `MPI_Allreduce(MAX)`），由 owner 列进程计算参考值并广播。

**注意**：BCAST_VERIFY 使用均匀求和（非加权），因广播缓冲含 L2+L1+DPIV 异构数据，加权需按块分别处理，复杂度高。

#### 检测点 3：尾矩阵增量更新（TRAIL_VERIFY）

尾矩阵占总计算量 $\frac{2}{3}N^3$。每次矩阵块更新公式为 $A_{\text{trail}} \leftarrow A_{\text{trail}} - L_2 \times U$。

```
每次 pdupdate 调用:
  1. 重算 CS_TRAIL 基线（均匀权值列和，置换不变）
  2. 每个 dgemm 块后增量更新 CS_TRAIL
  3. 调用末尾全量验证 CS_TRAIL vs 重算值
```

**设计决策**：尾矩阵使用均匀权值（w[i]=1.0），因为 HPL 选主操作（laswp）对尾矩阵做行置换，位置相关权值（2的幂）会失效。均匀权值对行置换不变。

#### 检测点 4：回代求解校验

在 `HPL_pdtrsv` 完成后：
- NaN/Inf 检测（解向量校验和异常）
- 统计异常检测（偏离均值超过 6σ 的异常值）

---

### 三、校验和比较算法

```c
deviation = |cs_computed - cs_expected|
denom     = max(|cs_expected|, 1.0)

if( deviation / denom > 1.0e-10 )
    → SDC 故障 (返回 1)
else
    → 正常 (返回 0)
```

使用**相对误差**而非绝对误差，适应不同数量级的校验和。

---

### 四、故障记录与聚合报告

检测到 SDC 后，`HPL_sdc_report.c` 完成：
- 本地链表记录：MPI Rank、网格坐标、物理节点名、故障类型、步数、偏差值
- 跨节点聚合（MPI_Gatherv）：汇总至 Rank 0
- 按节点/类型统计 + 换件建议（>10 故障的节点）

---

### 五、编译开关

| 宏 | 默认 | 用途 |
|----|------|------|
| `HPL_SDC_CHECK` | - | SDC 总开关 |
| `HPL_SDC_BCAST_VERIFY` | 1 | 面板广播校验 |
| `HPL_SDC_TRAIL_VERIFY` | 0 | 尾矩阵校验（需显式启用） |
| `HPL_SDC_INJECT` | - | 故障注入（测试用） |

启用 TRAIL_VERIFY：`-DHPL_SDC_TRAIL_VERIFY=1`

故障注入环境变量：
- `HPL_SDC_INJECT_COL`：注入面板列号（默认 nb）
- `HPL_SDC_INJECT_VAL`：注入值（默认 1e6）

---

### 六、开销分析

| 检测点 | 计算开销 | 通信开销 |
|--------|---------|---------|
| BCAST_VERIFY | O(mp × jb)/面板 | 1 double MPI_Bcast |
| TRAIL_VERIFY | ~1.1%（重算+增量+验证） | 无 |
| 回代检测 | O(nq) | 无 |
| **总额外开销** | **<3%** | **可忽略** |
