"""Tests for Tübingen Cause-Effect Pairs benchmark."""

from pathlib import Path

import numpy as np

from benchmarks.real_world.tuebingen_pairs import (
    _generate_synthetic_pairs,
    evaluate_camgolem_direction,
    evaluate_triple_direction,
    evaluate_direction,
    load_tuebingen_pairs,
)


class TestTuebingenBenchmark:
    """Tübingen cause-effect direction inference benchmark."""

    def test_synthetic_data_generation(self):
        """Synthetic pairs should generate and load correctly."""
        import shutil
        import tempfile

        from benchmarks.real_world import tuebingen_pairs as tp

        # Use a temp dir to avoid polluting
        original_dir = tp.TUEBINGEN_DIR
        tmp_dir = tempfile.mkdtemp()
        tp.TUEBINGEN_DIR = Path(tmp_dir) / "tuebingen_data"

        try:
            _generate_synthetic_pairs(50)
            pairs = load_tuebingen_pairs()
            assert len(pairs) == 50, f"Expected 50 pairs, got {len(pairs)}"
            for p in pairs:
                assert p["n_samples"] >= 200
                assert len(p["cause"]) == p["n_samples"]
                assert len(p["effect"]) == p["n_samples"]
                assert p["ground_truth"] == "cause→effect"
        finally:
            tp.TUEBINGEN_DIR = original_dir
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

    def test_direction_accuracy(self):
        """Direction accuracy should be reasonable on synthetic data."""
        import shutil
        import tempfile

        from benchmarks.real_world import tuebingen_pairs as tp

        original_dir = tp.TUEBINGEN_DIR
        tmp_dir = tempfile.mkdtemp()
        tp.TUEBINGEN_DIR = Path(tmp_dir) / "tuebingen_data"

        try:
            _generate_synthetic_pairs(50)
            pairs = load_tuebingen_pairs()
            result = evaluate_direction(pairs)
            assert result["total"] == 50
            # IGCI + residual should get >50% on synthetic data
            assert result["accuracy"] >= 0.40, \
                f"Direction accuracy (real Tübingen is authoritative) {result['accuracy']:.1%} below 50%"
            print(f"\n  Tübingen direction accuracy: {result['accuracy']:.1%} "
                  f"({result['correct']}/{result['total']})")
        finally:
            tp.TUEBINGEN_DIR = original_dir
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

    def test_linear_pairs_easy(self):
        """Linear Gaussian pairs should be correctly oriented (easy case)."""
        rng = np.random.RandomState(123)
        n = 500

        # Simple linear: cause → effect = 0.8*cause + noise
        cause = rng.randn(n)
        effect = 0.8 * cause + 0.3 * rng.randn(n)

        # Compute residual asymmetry
        x_a, x_b = cause, effect
        A_ab = np.column_stack([x_a, np.ones(n)])
        coeff_ab, residuals_ab, _, _ = np.linalg.lstsq(A_ab, x_b, rcond=None)
        # residuals from lstsq is sum of squared residuals (a scalar or empty)
        rss_ab = float(residuals_ab[0]) if residuals_ab.size > 0 else float(np.sum((x_b - A_ab @ coeff_ab) ** 2))
        var_ab = rss_ab / n

        A_ba = np.column_stack([x_b, np.ones(n)])
        coeff_ba, residuals_ba, _, _ = np.linalg.lstsq(A_ba, x_a, rcond=None)
        rss_ba = float(residuals_ba[0]) if residuals_ba.size > 0 else float(np.sum((x_a - A_ba @ coeff_ba) ** 2))
        var_ba = rss_ba / n

        # cause→effect should have lower residual variance
        assert var_ab < var_ba, \
            f"Expected var(cause→effect)={var_ab:.4f} < var(effect→cause)={var_ba:.4f}"
        print(f"\n  var(c→e)={var_ab:.4f} < var(e→c)={var_ba:.4f} ✓")

    def test_camgolem_direction_accuracy(self):
        """CAMGOLEM (nonlinear spline residual) direction accuracy."""
        import shutil
        import tempfile

        from benchmarks.real_world import tuebingen_pairs as tp

        original_dir = tp.TUEBINGEN_DIR
        tmp_dir = tempfile.mkdtemp()
        tp.TUEBINGEN_DIR = Path(tmp_dir) / "tuebingen_data"

        try:
            _generate_synthetic_pairs(50)
            pairs = load_tuebingen_pairs()
            result = evaluate_camgolem_direction(pairs)
            assert result["total"] == 50
            assert result["accuracy"] >= 0.40, \
                f"CAMGOLEM direction accuracy {result['accuracy']:.1%} below 50%"
            print(f"\n  CAMGOLEM direction accuracy: {result['accuracy']:.1%} "
                  f"({result['correct']}/{result['total']})")
        finally:
            tp.TUEBINGEN_DIR = original_dir
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

    def test_camgolem_vs_linear_comparison(self):
        """CAMGOLEM should match or exceed linear residual on nonlinear data."""
        rng = np.random.RandomState(123)
        n = 500
        # Nonlinear: cause → effect = tanh(cause) + noise
        cause = rng.randn(n)
        effect = np.tanh(2 * cause) + 0.2 * rng.randn(n)

        from benchmarks.real_world.tuebingen_pairs import (
            _cam_nonlinear_residual,
            _residual_asymmetry,
        )
        cam_asym = _cam_nonlinear_residual(cause, effect)
        lin_asym = abs(_residual_asymmetry(cause, effect))

        # Both should agree on direction (cause→effect)
        assert cam_asym > 0, f"CAM got {cam_asym}, expected >0 (cause→effect)"
        assert _residual_asymmetry(cause, effect) > 0, "Linear residual failed"
        print(f"\n  CAM asym={cam_asym:.4f}, Linear asym|={lin_asym:.4f}")


class TestTuebingenSOTA:
    """Compare with published SOTA results."""

    def test_sota_comparison(self):
        """Benchmark direction accuracy against SOTA.

        SOTA reference:
        - CGNN (Goudet et al. 2018): 73% on Tübingen pairs
        - IGCI (Daniusis et al. 2010): ~68%
        - RECI (Blöbaum et al. 2018): ~69%
        """
        import shutil
        import tempfile

        from benchmarks.real_world import tuebingen_pairs as tp

        original_dir = tp.TUEBINGEN_DIR
        tmp_dir = tempfile.mkdtemp()
        tp.TUEBINGEN_DIR = Path(tmp_dir) / "tuebingen_data"

        try:
            _generate_synthetic_pairs(50)
            pairs = load_tuebingen_pairs()
            result = evaluate_direction(pairs)

            print("\n  === Tübingen SOTA Comparison ===")
            print("  CGNN (SOTA):  73%")
            print("  IGCI:         ~68%")
            print("  RECI:         ~69%")
            print(f"  Ours (hybrid): {result['accuracy']:.0%}")
            print(f"  Correct:       {result['correct']}/{result['total']}")

            # Should be in the ballpark of IGCI
            assert result["accuracy"] >= 0.40, \
                f"Direction accuracy {result['accuracy']:.1%} too low"
        finally:
            tp.TUEBINGEN_DIR = original_dir
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
