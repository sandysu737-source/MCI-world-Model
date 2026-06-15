# Ch07 经济成本分析 (TCO) — 改进规划书

## 1. 章节概述

原报告第七章分析了三种方案在 1000 QPS × 1 年的 TCO：
- LLM-7B 自托管: ~$3.15M/年
- LLM-10B API: ~$15.8M/年
- **MCI V5.0.0 CPU: ~$0.03M/年** (100-500x 优势)

但 MCI 的经济优势仅在因果密集、安全关键的垂直领域成立。

## 2. 改进目标

| # | 目标 | 量化指标 | 优先级 |
|---|---|---|---|
| G1 | 建立 MCI 成本基准测试 | 单推理延迟 <10ms, 内存 <512MB | P2 |
| G2 | 构建混合架构成本模拟器 | 可计算任意混合比例的 TCO | P2 |
| G3 | 优化推理性能 | 单次推理 CPU 时间 <5ms | P3 |
| G4 | 领域级 TCO 对比报告 | 医疗/法律/工程各有 TCO 数据 | P3 |

## 3. 实施方案

### 3.1 成本基准测试 (`benchmarks/economics/`)

```python
class CostBenchmark:
    def benchmark_inference(self, wm: MCIWorldModel, n_runs=1000):
        """单次推理成本测量"""
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            wm.predict_effect("价格上涨")
            times.append(time.perf_counter() - start)
        return {
            "avg_ms": np.mean(times) * 1000,
            "p95_ms": np.percentile(times, 95) * 1000,
            "memory_mb": self._measure_memory(),
            "cost_per_inference": self._calc_cost(times),
        }
```

### 3.2 混合架构 TCO 模拟器

```python
@dataclass
class HybridTCOSimulator:
    mci_cost_per_query: float = 0.000001  # $/query
    llm_cost_per_query: float = 0.0001     # $/query (7B)
    mci_routing_ratio: float = 0.80        # 80% 走 MCI
    
    def simulate(self, qps=1000, days=365) -> dict:
        total_queries = qps * 86400 * days
        mci_queries = total_queries * self.mci_routing_ratio
        llm_queries = total_queries * (1 - self.mci_routing_ratio)
        return {
            "total_cost": mci_queries * self.mci_cost_per_query + llm_queries * self.llm_cost_per_query,
            "mci_cost": mci_queries * self.mci_cost_per_query,
            "llm_cost": llm_queries * self.llm_cost_per_query,
            "savings_vs_llm_only": (total_queries * self.llm_cost_per_query) - ...,
        }
```

### 3.3 推理性能优化

```python
# 优化热点: DoCalculus 的 O(V+E) 因果图遍历
# 优化: 缓存拓扑排序 + 增量更新
class CachedDoCalculus(DoCalculus):
    def __init__(self, causal_graph):
        super().__init__(causal_graph)
        self._topo_cache = None
    
    def estimate_ate(self, X, Y, **kwargs):
        if self._topo_cache is None:
            self._topo_cache = self._topological_sort()
        # 使用缓存拓扑，避免重复计算
        return self._fast_ate(X, Y, self._topo_cache, **kwargs)
```

## 4. 时间计划

| 周 | 任务 | 交付物 |
|---|---|---|
| W12-14 | 成本基准测试框架 | `benchmarks/economics/` |
| W15-18 | 混合 TCO 模拟器 | `tools/tco_simulator.py` |
| W19-24 | 推理性能优化 | 缓存 + numpy 优化 |
| W25-28 | 领域 TCO 对比报告 | 医疗/法律/工程 |
| W29-40 | 持续优化 + 边缘部署测试 | 树莓派/嵌入式验证 |

## 5. 资源配置

| 资源 | 数量 | 成本 |
|---|---|---|
| 后端工程师 | 1人 × 10周 | 20 人天 |
| SRE / 运维 | 0.5人 × 6周 | 10 人天 |
| 边缘设备 (树莓派) | 1 台 | $50 |

## 6. KPI 指标

| KPI | 基线 | 目标 |
|---|---|---|
| 单次推理延迟 | ~15ms | <5ms (CPU) |
| 内存占用 | ~256MB | <128MB |
| 混合架构 TCO 节省 | 未测量 | 量化数据 |
| 边缘部署可用 | 未测试 | 树莓派可运行 |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| 性能优化影响正确性 | 低 | 高 | 基准测试 + 回归测试 |
| 边缘设备内存不足 | 中 | 中 | 模型量化 + 懒加载 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 |
|---|---|---|
| 基准测试框架 | 8 | $0 |
| TCO 模拟器 | 6 | $0 |
| 推理优化 | 8 | $0 |
| 领域报告 | 4 | $0 |
| 边缘验证 | 4 | $50 |
| **小计** | **30** | **$50** |

## 9. 验收标准

- [ ] 单次推理 CPU 延迟 <5ms
- [ ] 内存占用 <128MB
- [ ] TCO 模拟器可输出任意混合比例的年度成本
- [ ] 树莓派 4B (4GB) 可运行核心推理

## 依赖关系

- **前置**: Ch04 (致命缺陷修复，否则成本数据无意义)
- **被依赖**: Ch14 (战略结论), Ch09 (替代性目标)
