from __future__ import annotations
"""MCI World Model v4.4.0 — Surgical Workflow. 清创手术四相状态机 + 五层安全门禁。"""
import json, logging, time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

class SurgicalPhase(IntEnum):
    IDLE=-1; EXPLORE=0; DEBRIDE=1; HEMOSTASIS=2; VERIFY=3; EMERGENCY_STOP=4; COMPLETE=5
    @property
    def display(self) -> str:
        return {-1:"空闲",0:"探查",1:"清创",2:"止血",3:"验证",4:"紧急停止",5:"完成"}.get(self.value,"未知")

@dataclass
class SurgicalState:
    phase: SurgicalPhase = SurgicalPhase.IDLE; tissue_type: int = 0; tool_force_n: float = 0.0
    tool_velocity: float = 0.0; depth_mm: float = 0.0; temperature_c: float = 37.0
    bleeding_level: int = 0; tissue_confidence: float = 0.0; phase_duration_s: float = 0.0
    safety_flags: list[str] = field(default_factory=list); timestamp: float = 0.0
    def to_dict(self) -> dict:
        return {"phase":self.phase.value,"phase_name":self.phase.display,"tissue_type":self.tissue_type,
                "tool_force_n":round(self.tool_force_n,2),"tool_velocity":round(self.tool_velocity,1),
                "depth_mm":round(self.depth_mm,2),"temperature_c":round(self.temperature_c,1),
                "bleeding_level":self.bleeding_level,"tissue_confidence":round(self.tissue_confidence,3),
                "phase_duration_s":round(self.phase_duration_s,1),"safety_flags":self.safety_flags,"timestamp":self.timestamp}

@dataclass
class PhaseTransition:
    from_phase: SurgicalPhase; to_phase: SurgicalPhase; reason: str = ""
    auto: bool = False; safety_checks_passed: int = 0; safety_checks_total: int = 0; timestamp: float = 0.0

@dataclass
class WorkflowReport:
    transitions: list[PhaseTransition] = field(default_factory=list); total_duration_s: float = 0.0
    phase_durations: dict[str, float] = field(default_factory=dict); safety_violations: int = 0
    emergency_stops: int = 0; completion_status: str = "incomplete"
    def to_dict(self) -> dict:
        return {"n_transitions":len(self.transitions),"total_duration_s":round(self.total_duration_s,1),
                "phase_durations":{k:round(v,1) for k,v in self.phase_durations.items()},
                "safety_violations":self.safety_violations,"emergency_stops":self.emergency_stops,"completion_status":self.completion_status}

class SurgicalWorkflow:
    VALID_TRANSITIONS = {
        SurgicalPhase.IDLE: {SurgicalPhase.EXPLORE, SurgicalPhase.EMERGENCY_STOP},
        SurgicalPhase.EXPLORE: {SurgicalPhase.DEBRIDE, SurgicalPhase.VERIFY, SurgicalPhase.EMERGENCY_STOP},
        SurgicalPhase.DEBRIDE: {SurgicalPhase.HEMOSTASIS, SurgicalPhase.EXPLORE, SurgicalPhase.EMERGENCY_STOP},
        SurgicalPhase.HEMOSTASIS: {SurgicalPhase.DEBRIDE, SurgicalPhase.VERIFY, SurgicalPhase.EMERGENCY_STOP},
        SurgicalPhase.VERIFY: {SurgicalPhase.EXPLORE, SurgicalPhase.DEBRIDE, SurgicalPhase.COMPLETE, SurgicalPhase.EMERGENCY_STOP},
        SurgicalPhase.EMERGENCY_STOP: {SurgicalPhase.IDLE},
        SurgicalPhase.COMPLETE: set(),
    }

    def __init__(self, enable_auto_transitions: bool = False) -> None:
        self._current_phase = SurgicalPhase.IDLE; self._state = SurgicalState()
        self._report = WorkflowReport(); self._phase_start_time = time.time(); self._enable_auto = enable_auto_transitions
        self._transitions: list[PhaseTransition] = []; self._phase_durations: dict[int, float] = {}

    @property
    def current_phase(self) -> SurgicalPhase: return self._current_phase
    @property
    def state(self) -> SurgicalState: return self._state

    def update_state(self, tissue_type: int | None = None, force_n: float | None = None, velocity: float | None = None,
                     depth_mm: float | None = None, temperature_c: float | None = None, bleeding_level: int | None = None,
                     tissue_confidence: float | None = None) -> SurgicalState:
        if tissue_type is not None: self._state.tissue_type = tissue_type
        if force_n is not None: self._state.tool_force_n = force_n
        if velocity is not None: self._state.tool_velocity = velocity
        if depth_mm is not None: self._state.depth_mm = depth_mm
        if temperature_c is not None: self._state.temperature_c = temperature_c
        if bleeding_level is not None: self._state.bleeding_level = bleeding_level
        if tissue_confidence is not None: self._state.tissue_confidence = tissue_confidence
        self._state.phase = self._current_phase; self._state.phase_duration_s = time.time()-self._phase_start_time
        self._state.timestamp = time.time(); self._check_safety(); return self._state

    def _check_safety(self) -> None:
        from mci_world_model.sdk._force_tissue_dynamics import TISSUE_PARAMS
        self._state.safety_flags = []; s = self._state
        if s.temperature_c > 45.0: self._state.safety_flags.append("L1: 温度>45°C")
        params = TISSUE_PARAMS.get(s.tissue_type, {}); mf = params.get("max_force", 5.0)
        if s.tool_force_n > mf*2: self._state.safety_flags.append(f"L2: 力严重超限 {s.tool_force_n:.1f}>{mf*2}N")
        elif s.tool_force_n > mf: self._state.safety_flags.append(f"L2: 力超限 {s.tool_force_n:.1f}>{mf}N")
        if s.tissue_confidence < 0.5 and s.tissue_confidence > 0: self._state.safety_flags.append(f"L3: 置信度低({s.tissue_confidence:.2f})")
        if s.bleeding_level >= 3: self._state.safety_flags.append("L4: 严重出血")
        if len(self._state.safety_flags) >= 2: self._state.safety_flags.append("L5: 多重违规→急停")
        if self._state.safety_flags: logger.warning("Safety: %s", " | ".join(self._state.safety_flags))

    def transition_to(self, target: SurgicalPhase, reason: str = "", auto: bool = False, force: bool = False) -> tuple[bool, str]:
        if not force and target not in self.VALID_TRANSITIONS.get(self._current_phase, set()):
            return False, f"非法切换: {self._current_phase.display}→{target.display}"
        elapsed = time.time()-self._phase_start_time
        self._phase_durations[self._current_phase.value] = self._phase_durations.get(self._current_phase.value, 0)+elapsed
        cp = 0; ct = 0
        if not force: cp, ct = self._run_checks(target)
        tr = PhaseTransition(from_phase=self._current_phase, to_phase=target, reason=reason or f"切换到{target.display}", auto=auto, safety_checks_passed=cp, safety_checks_total=ct, timestamp=time.time())
        self._transitions.append(tr); old = self._current_phase; self._current_phase = target; self._phase_start_time = time.time()
        if target == SurgicalPhase.EMERGENCY_STOP: self._report.emergency_stops += 1
        logger.info("Transition: %s→%s (%d/%d)", old.display, target.display, cp, ct)
        return True, f"切换: {old.display}→{target.display}"

    def _run_checks(self, target: SurgicalPhase) -> tuple[int, int]:
        checks = [not bool(self._state.safety_flags), self._state.temperature_c < 43.0,
                  self._state.tissue_confidence >= 0.5 or self._state.tissue_confidence == 0, self._state.bleeding_level < 3]
        return sum(checks), len(checks)

    def emergency_stop(self, reason: str = "手动触发") -> bool:
        return self.transition_to(SurgicalPhase.EMERGENCY_STOP, reason=reason, force=True)[0]

    def export_report(self) -> WorkflowReport:
        final_elapsed = time.time()-self._phase_start_time
        self._phase_durations[self._current_phase.value] = self._phase_durations.get(self._current_phase.value,0)+final_elapsed
        total = sum(self._phase_durations.values()); pn = {}
        for val, dur in self._phase_durations.items():
            try: pn[SurgicalPhase(val).display] = dur
            except ValueError: pn[f"Phase({val})"] = dur
        self._report = WorkflowReport(transitions=self._transitions, total_duration_s=total, phase_durations=pn,
            safety_violations=sum(1 for t in self._transitions if t.safety_checks_passed<t.safety_checks_total),
            emergency_stops=self._report.emergency_stops,
            completion_status="complete" if self._current_phase==SurgicalPhase.COMPLETE else "incomplete")
        return self._report

    def export_trajectory(self, path: str) -> None:
        self.export_report(); os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        traj = {"version":"4.4.0","workflow":"debridement","report":self._report.to_dict(),
                "transitions":[{"from":t.from_phase.display,"to":t.to_phase.display,"reason":t.reason,"auto":t.auto,"checks":f"{t.safety_checks_passed}/{t.safety_checks_total}","timestamp":t.timestamp} for t in self._transitions],
                "current_phase":self._current_phase.display,"state":self._state.to_dict()}
        with open(path,"w") as f: json.dump(traj, f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        s = self._state; return f"SurgicalWorkflow(phase={self._current_phase.display}, force={s.tool_force_n:.1f}N, depth={s.depth_mm:.1f}mm, temp={s.temperature_c:.1f}°C, flags={len(s.safety_flags)})"
import os
