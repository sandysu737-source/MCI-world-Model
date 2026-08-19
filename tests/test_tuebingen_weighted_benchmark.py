"""L3 真实数据基准测试 — Tübingen 官方加权评分。

验证方向推断在真实 Tübingen cause-effect pairs 上的表现, 使用 Mooij et al.
(2016) 的官方加权准确率口径, 与文献 SOTA 同口径对比。

这是 L3 层 (真实世界可证伪基准), 回应"循环验证"的批评:
之前用合成数据测试算法, 现在用 109 对真实世界数据 + 官方加权评分。
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.oracle, pytest.mark.realdata]

from benchmarks.real_world.tuebingen_pairs import (
    evaluate_direction_weighted,
    load_pairmeta_weights,
    load_tuebingen_pairs,
)


@pytest.fixture(scope="module")
def pairs():
    """加载真实 Tübingen pairs (module 级共享, 避免重复 IO)。"""
    return load_tuebingen_pairs()


@pytest.fixture(scope="module")
def weights():
    return load_pairmeta_weights()


class TestTubingenRealBenchmark:
    """真实 Tübingen 数据上的方向推断。"""

    def test_real_pairs_loaded(self, pairs):
        """应加载到真实 pairs (非合成)。"""
        assert len(pairs) >= 90, f"仅加载 {len(pairs)} 对, 期望 ≥90"

    def test_official_weights_loaded(self, weights):
        """应加载到官方 pairmeta 权重。"""
        assert len(weights) >= 100, f"仅 {len(weights)} 个权重, 期望 ≥100"
        # 权重应在合理范围 (0, 1]
        max_w = max(weights.values())
        assert 0 < max_w <= 1.0, f"最大权重 {max_w} 不在 (0,1]"

    def test_weighted_accuracy_above_random(self, pairs):
        """加权准确率必须显著优于随机 (50%)。"""
        result = evaluate_direction_weighted(pairs, method="hybrid")
        assert result["weighted_accuracy"] > 0.55, f"加权准确率 {result['weighted_accuracy']:.1%} 未显著优于随机 50%"

    def test_weighted_vs_unweighted_reported(self, pairs):
        """应同时报告加权与未加权, 便于与文献对比。"""
        result = evaluate_direction_weighted(pairs)
        assert "weighted_accuracy" in result
        assert "accuracy" in result
        # 两者都应有意义 (都优于随机)
        assert result["weighted_accuracy"] > 0.50
        assert result["accuracy"] > 0.50

    def test_weighted_accuracy_in_sota_ballpark(self, pairs):
        """加权准确率应在合理的 SOTA 区间内。

        文献参考 (加权口径, Mooij 2016 及后续):
          - IGCI: ~62-68%
          - CGNN: ~73%
          - 最佳方法: ~75-80%

        本项目 hybrid 方法 (IGCI + 残差不对称性投票) 应落在 55-75% 区间。
        低于 55% 说明方法失效, 高于 80% 可能过拟合或评测有误。
        """
        result = evaluate_direction_weighted(pairs, method="hybrid")
        acc = result["weighted_accuracy"]
        assert 0.55 <= acc <= 0.80, (
            f"加权准确率 {acc:.1%} 超出合理 SOTA 区间 [55%, 80%]。未加权={result['accuracy']:.1%}"
        )

    def test_all_pairs_have_weight(self, pairs, weights):
        """每个加载的 pair 都应有对应官方权重。"""
        missing = [p["pair_id"] for p in pairs if p["pair_id"] not in weights]
        # 允许少数 pair 缺权重 (用了默认 1.0), 但不应大面积缺失
        assert len(missing) <= 5, f"{len(missing)} 个 pair 缺官方权重: {missing[:5]}"
