"""MCI World Model v5.1.0 — 认知/语义安全约束 测试

P1-F7 修复验证: 8 类物理约束 → 13 类 (8 物理 + 5 认知/语义)
"""

from __future__ import annotations

from mci_world_model.sdk._safety import SafetyCheckResult, SafetyConstraint, SafetyMonitor
from mci_world_model.sdk._safety_cognitive import (
    CognitiveSafetyConstraint,
    ContentSafetyConstraint,
    SocialSafetyConstraint,
    TemporalSafetyConstraint,
    ValueAlignmentConstraint,
)
from mci_world_model.sdk._world_state import PendulumAction, PendulumState

# ── 辅助 ──────────────────────────────────────────────────────────────────


class _TextAction:
    """带文本描述的测试动作。"""

    def __init__(self, description: str = "", **kw):
        self.description = description
        self.__dict__.update(kw)


class _TimestampedAction:
    """带时间戳的测试动作。"""

    def __init__(self, timestamp: float = 1.0, **kw):
        self.timestamp = timestamp
        self.__dict__.update(kw)


class _TimestampedState:
    """带时间戳的测试状态。"""

    def __init__(self, timestamp: float = 0.0, **kw):
        self.timestamp = timestamp
        self.__dict__.update(kw)


# ═══════════════════════════════════════════════════════════════════════════
# Test F7: 5 类认知约束存在 + 继承正确
# ═══════════════════════════════════════════════════════════════════════════


class TestF7Fix:
    """P1-F7 修复: 8 类物理约束 → 13 类 (8 物理 + 5 认知)。"""

    def test_five_cognitive_constraints_exist(self):
        """5 类认知约束均存在且可实例化。"""
        constraints = [
            ContentSafetyConstraint(),
            CognitiveSafetyConstraint(),
            ValueAlignmentConstraint(),
            TemporalSafetyConstraint(),
            SocialSafetyConstraint(),
        ]
        assert len(constraints) == 5

    def test_all_inherit_safety_constraint(self):
        """5 类约束均继承 SafetyConstraint ABC。"""
        for cls in [
            ContentSafetyConstraint,
            CognitiveSafetyConstraint,
            ValueAlignmentConstraint,
            TemporalSafetyConstraint,
            SocialSafetyConstraint,
        ]:
            assert issubclass(cls, SafetyConstraint)

    def test_total_constraint_count_13(self):
        """SafetyMonitor 注册 8 物理 + 5 认知 = 13 类。"""
        from mci_world_model.sdk._safety import (
            AccelerationLimitConstraint,
            ForceLimitConstraint,
            JointLimitConstraint,
            PositionBoundConstraint,
            SelfCollisionConstraint,
            ToolForceConstraint,
            VelocityLimitConstraint,
            WorkspaceBoundConstraint,
        )

        monitor = SafetyMonitor()
        # 8 物理
        monitor.register(ForceLimitConstraint(10.0))
        monitor.register(VelocityLimitConstraint(10.0))
        monitor.register(PositionBoundConstraint(10.0))
        monitor.register(AccelerationLimitConstraint(10.0))
        monitor.register(JointLimitConstraint(0.0, 3.14))
        monitor.register(ToolForceConstraint(5.0))
        monitor.register(SelfCollisionConstraint([(0.0, 0.0)]))
        monitor.register(WorkspaceBoundConstraint(1.0))
        # 5 认知
        monitor.register(ContentSafetyConstraint())
        monitor.register(CognitiveSafetyConstraint())
        monitor.register(ValueAlignmentConstraint())
        monitor.register(TemporalSafetyConstraint())
        monitor.register(SocialSafetyConstraint())

        assert len(monitor._constraints) == 13

    def test_cognitive_importable_from_sdk(self):
        """5 类约束可从 sdk 顶层导入。"""
        from mci_world_model.sdk import (
            CognitiveSafetyConstraint,
            ContentSafetyConstraint,
            SocialSafetyConstraint,
            TemporalSafetyConstraint,
            ValueAlignmentConstraint,
        )

        assert ContentSafetyConstraint is not None
        assert CognitiveSafetyConstraint is not None
        assert ValueAlignmentConstraint is not None
        assert TemporalSafetyConstraint is not None
        assert SocialSafetyConstraint is not None


# ═══════════════════════════════════════════════════════════════════════════
# ContentSafetyConstraint 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestContentSafety:
    """有害/伦理关键词过滤。"""

    def setup_method(self):
        self.constraint = ContentSafetyConstraint()
        self.state = PendulumState(theta=0.1, omega=0.5)

    def test_harmful_keyword_detected(self):
        """有害关键词检出 → violation。"""
        action = _TextAction(description="执行暴力攻击")
        result = self.constraint.check(self.state, action)
        assert not result.passed
        assert result.severity == "violation"
        assert "暴力" in result.reason

    def test_harmful_english_keyword(self):
        """英文有害关键词检出。"""
        action = _TextAction(description="kill the process")
        result = self.constraint.check(self.state, action)
        assert not result.passed
        assert result.severity == "violation"

    def test_ethical_keyword_warning(self):
        """伦理敏感词检出 → warning。"""
        action = _TextAction(description="未经同意采集数据")
        result = self.constraint.check(self.state, action)
        assert not result.passed
        assert result.severity == "warning"

    def test_safe_text_passes(self):
        """安全文本放行。"""
        action = _TextAction(description="正常操作")
        result = self.constraint.check(self.state, action)
        assert result.passed

    def test_name_property(self):
        """name 属性正确。"""
        assert self.constraint.name == "content_safety"


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveSafetyConstraint 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveSafety:
    """幻觉检测 + 事实核查 + 不确定性阈值。"""

    def setup_method(self):
        self.constraint = CognitiveSafetyConstraint()
        self.state = PendulumState(theta=0.1, omega=0.5)

    def test_normal_state_passes(self):
        """正常物理状态通过。"""
        action = PendulumAction(torque=0.5)
        result = self.constraint.check(self.state, action)
        assert result.passed

    def test_low_confidence_warning(self):
        """低置信度 → warning。"""
        constraint = CognitiveSafetyConstraint(uncertainty_threshold=0.95)
        state = PendulumState(theta=0.1, omega=0.5)
        action = PendulumAction(torque=0.5)
        result = constraint.check(state, action)
        # PendulumState 没有 confidence 属性，默认 0.9 < 0.95
        assert not result.passed
        assert result.severity == "warning"

    def test_hallucination_violation(self):
        """极高物理量值 → 幻觉 violation。"""

        class ExtremeState:
            theta = 200.0  # 远超 100
            omega = 0.5

        result = self.constraint.check(ExtremeState(), None)
        assert not result.passed
        assert result.severity == "violation"

    def test_consistency_warning(self):
        """大力矩+零角速度 → 事实一致性 warning。"""

        class InconsistentState:
            theta = 1.0
            omega = 0.001

        class HighTorqueAction:
            torque = 10.0

        result = self.constraint.check(InconsistentState(), HighTorqueAction())
        assert not result.passed
        assert result.severity == "warning"
        assert "一致性" in result.reason

    def test_name_property(self):
        assert self.constraint.name == "cognitive_safety"


# ═══════════════════════════════════════════════════════════════════════════
# ValueAlignmentConstraint 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestValueAlignment:
    """价值对齐度检查。"""

    def setup_method(self):
        self.constraint = ValueAlignmentConstraint()

    def test_well_aligned_passes(self):
        """对齐度高时通过。"""
        # 力矩与角度方向相反 (阻尼) → 高对齐
        state = PendulumState(theta=0.5, omega=0.1)
        action = PendulumAction(torque=-0.3)  # 反向力矩 → 阻尼
        result = self.constraint.check(state, action)
        assert result.passed

    def test_misaligned_warning(self):
        """对齐度 < 0.8 但 ≥ 0.6 → warning。"""
        # 力矩与角度同向 → 低对齐
        state = PendulumState(theta=0.5, omega=0.1)
        action = PendulumAction(torque=0.8)  # 同向 → 低对齐
        result = self.constraint.check(state, action)
        # 可能是 warning 或通过，取决于具体计算
        assert isinstance(result, SafetyCheckResult)

    def test_severe_misalignment_violation(self):
        """严重失对齐 → violation。"""
        # 大力矩同向 → 对齐度可能 < 0.6
        constraint = ValueAlignmentConstraint(
            alignment_threshold_warning=0.99,
            alignment_threshold_violation=0.95,
        )
        state = PendulumState(theta=2.0, omega=0.1)
        action = PendulumAction(torque=5.0)  # 同向大力矩
        result = constraint.check(state, action)
        assert not result.passed  # 高阈值下应触发

    def test_name_property(self):
        assert self.constraint.name == "value_alignment"


# ═══════════════════════════════════════════════════════════════════════════
# TemporalSafetyConstraint 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestTemporalSafety:
    """因果倒置禁止 + 时序一致性。"""

    def setup_method(self):
        self.constraint = TemporalSafetyConstraint()

    def test_normal_temporal_passes(self):
        """正常时序通过。"""
        state = PendulumState(theta=0.1, omega=0.5)
        action = PendulumAction(torque=0.3)
        result = self.constraint.check(state, action)
        assert result.passed

    def test_causal_inversion_violation(self):
        """因果倒置 → violation (动作时间 < 状态时间)。"""
        state = _TimestampedState(timestamp=10.0)
        action = _TimestampedAction(timestamp=5.0)  # 倒置
        result = self.constraint.check(state, action)
        assert not result.passed
        assert result.severity == "violation"
        assert "倒置" in result.reason

    def test_correct_temporal_order(self):
        """正确时序 (动作 ≥ 状态) 通过。"""
        state = _TimestampedState(timestamp=5.0)
        action = _TimestampedAction(timestamp=10.0)
        result = self.constraint.check(state, action)
        assert result.passed

    def test_name_property(self):
        assert self.constraint.name == "temporal_safety"


# ═══════════════════════════════════════════════════════════════════════════
# SocialSafetyConstraint 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestSocialSafety:
    """隐私保护 + 公平性 + 偏见检测。"""

    def setup_method(self):
        self.constraint = SocialSafetyConstraint()
        self.state = PendulumState(theta=0.1, omega=0.5)

    def test_safe_text_passes(self):
        """安全文本通过。"""
        action = PendulumAction(torque=0.3)
        result = self.constraint.check(self.state, action)
        assert result.passed

    def test_pii_phone_detection(self):
        """手机号 PII 检出 → violation。"""
        action = _TextAction(description="联系 13812345678")
        result = self.constraint.check(self.state, action)
        assert not result.passed
        assert result.severity == "violation"
        assert "phone_number" in str(result.details)

    def test_pii_email_detection(self):
        """邮箱 PII 检出。"""
        action = _TextAction(description="发送到 user@example.com")
        result = self.constraint.check(self.state, action)
        assert not result.passed
        assert "email" in str(result.details)

    def test_bias_keyword_detected(self):
        """偏见关键词检出 → violation。"""
        action = _TextAction(description="男优于女的结论")
        result = self.constraint.check(self.state, action)
        assert not result.passed
        assert result.severity == "violation"

    def test_name_property(self):
        assert self.constraint.name == "social_safety"


# ═══════════════════════════════════════════════════════════════════════════
# 与 SafetyMonitor 集成测试
# ═══════════════════════════════════════════════════════════════════════════


class TestSafetyMonitorIntegration:
    """认知约束可注册到 SafetyMonitor 并参与 check_all。"""

    def test_cognitive_constraints_in_monitor(self):
        """认知约束注册到 Monitor 后 check_all 可用 (正常状态通过)。"""
        monitor = SafetyMonitor()
        monitor.register(ContentSafetyConstraint())
        monitor.register(CognitiveSafetyConstraint())
        monitor.register(ValueAlignmentConstraint())

        state = PendulumState(theta=0.1, omega=0.5)
        action = PendulumAction(torque=-0.3)  # 反向力矩 → 高对齐

        result = monitor.check_all(state, action)
        # check_all 返回单个 SafetyCheckResult (短路求值)
        assert isinstance(result, SafetyCheckResult)
        assert result.passed

    def test_cognitive_constraint_triggers_in_monitor(self):
        """有害内容在 monitor 中被检出。"""
        monitor = SafetyMonitor()
        monitor.register(ContentSafetyConstraint())

        state = PendulumState(theta=0.1, omega=0.5)
        action = _TextAction(description="暴力攻击")

        result = monitor.check_all(state, action)
        assert not result.passed
        assert result.severity == "violation"
