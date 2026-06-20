from __future__ import annotations
"""MCI World Model v4.4.0 — Surgical Workflow."""
import json, logging, os, time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any
import numpy as np
logger = logging.getLogger(__name__)
class SurgicalPhase(IntEnum):
    IDLE=-1; EXPLORE=0; DEBRIDE=1; HEMOSTASIS=2; VERIFY=3; EMERGENCY_STOP=4; COMPLETE=5
    @property
    def display(self): return {-1:"空闲",0:"探查",1:"清创",2:"止血",3:"验证",4:"紧急停止",5:"完成"}.get(self.value,"未知")
@dataclass
class SurgicalState:
    phase=SurgicalPhase.IDLE; tissue_type:int=0; tool_force_n:float=0.0; tool_velocity:float=0.0
    depth_mm:float=0.0; temperature_c:float=37.0; bleeding_level:int=0; tissue_confidence:float=0.0
    phase_duration_s:float=0.0; safety_flags:list=field(default_factory=list); timestamp:float=0.0
    def to_dict(self): return {"phase":self.phase.value,"phase_name":self.phase.display,"tissue_type":self.tissue_type,"tool_force_n":round(self.tool_force_n,2),"depth_mm":round(self.depth_mm,2),"temperature_c":round(self.temperature_c,1),"bleeding_level":self.bleeding_level,"tissue_confidence":round(self.tissue_confidence,3),"safety_flags":self.safety_flags}
@dataclass
class PhaseTransition:
    from_phase:SurgicalPhase; to_phase:SurgicalPhase; reason:str=""
    auto:bool=False; safety_checks_passed:int=0; safety_checks_total:int=0; timestamp:float=0.0
@dataclass
class WorkflowReport:
    transitions:list=field(default_factory=list); total_duration_s:float=0.0
    phase_durations:dict=field(default_factory=dict); safety_violations:int=0; emergency_stops:int=0; completion_status:str="incomplete"
    def to_dict(self): return {"n_transitions":len(self.transitions),"total_duration_s":round(self.total_duration_s,1),"phase_durations":{k:round(v,1) for k,v in self.phase_durations.items()},"safety_violations":self.safety_violations,"emergency_stops":self.emergency_stops,"completion_status":self.completion_status}
class SurgicalWorkflow:
    VALID_TRANSITIONS={
        SurgicalPhase.IDLE:{SurgicalPhase.EXPLORE,SurgicalPhase.EMERGENCY_STOP},
        SurgicalPhase.EXPLORE:{SurgicalPhase.DEBRIDE,SurgicalPhase.VERIFY,SurgicalPhase.EMERGENCY_STOP},
        SurgicalPhase.DEBRIDE:{SurgicalPhase.HEMOSTASIS,SurgicalPhase.EXPLORE,SurgicalPhase.EMERGENCY_STOP},
        SurgicalPhase.HEMOSTASIS:{SurgicalPhase.DEBRIDE,SurgicalPhase.VERIFY,SurgicalPhase.EMERGENCY_STOP},
        SurgicalPhase.VERIFY:{SurgicalPhase.EXPLORE,SurgicalPhase.DEBRIDE,SurgicalPhase.COMPLETE,SurgicalPhase.EMERGENCY_STOP},
        SurgicalPhase.EMERGENCY_STOP:{SurgicalPhase.IDLE}, SurgicalPhase.COMPLETE:set()}
    def __init__(self, enable_auto_transitions=False):
        self._current_phase=SurgicalPhase.IDLE; self._state=SurgicalState(); self._report=WorkflowReport()
        self._phase_start_time=time.time(); self._transitions=[]; self._phase_durations={}
    @property
    def current_phase(self): return self._current_phase
    @property
    def state(self): return self._state
    def update_state(self, tissue_type=None, force_n=None, velocity=None, depth_mm=None, temperature_c=None, bleeding_level=None, tissue_confidence=None):
        if tissue_type is not None: self._state.tissue_type=tissue_type
        if force_n is not None: self._state.tool_force_n=force_n
        if velocity is not None: self._state.tool_velocity=velocity
        if depth_mm is not None: self._state.depth_mm=depth_mm
        if temperature_c is not None: self._state.temperature_c=temperature_c
        if bleeding_level is not None: self._state.bleeding_level=bleeding_level
        if tissue_confidence is not None: self._state.tissue_confidence=tissue_confidence
        self._state.phase=self._current_phase; self._state.timestamp=time.time()
        self._state.phase_duration_s=time.time()-self._phase_start_time; self._check_safety(); return self._state
    def _check_safety(self):
        from mci_world_model.sdk._force_tissue_dynamics import TISSUE_PARAMS
        self._state.safety_flags=[]; s=self._state
        if s.temperature_c>45.0: self._state.safety_flags.append("L1:温度>45C")
        p=TISSUE_PARAMS.get(s.tissue_type,{}); mf=p.get("max_force",5.0)
        if s.tool_force_n>mf*2: self._state.safety_flags.append("L2:力严重超限")
        elif s.tool_force_n>mf: self._state.safety_flags.append("L2:力超限")
        if s.tissue_confidence<0.5 and s.tissue_confidence>0: self._state.safety_flags.append("L3:置信度低")
        if s.bleeding_level>=3: self._state.safety_flags.append("L4:严重出血")
        if len(self._state.safety_flags)>=2: self._state.safety_flags.append("L5:多重违规->急停")
    def transition_to(self, target, reason="", auto=False, force=False):
        if not force and target not in self.VALID_TRANSITIONS.get(self._current_phase,set()): return False,f"非法:{self._current_phase.display}->{target.display}"
        elapsed=time.time()-self._phase_start_time
        self._phase_durations[self._current_phase.value]=self._phase_durations.get(self._current_phase.value,0)+elapsed
        cp,ct=(0,0) if force else self._run_checks()
        tr=PhaseTransition(from_phase=self._current_phase,to_phase=target,reason=reason or f"->{target.display}",auto=auto,safety_checks_passed=cp,safety_checks_total=ct,timestamp=time.time())
        self._transitions.append(tr); old=self._current_phase; self._current_phase=target; self._phase_start_time=time.time()
        if target==SurgicalPhase.EMERGENCY_STOP: self._report.emergency_stops+=1
        return True,f"{old.display}->{target.display}"
    def _run_checks(self):
        checks=[not bool(self._state.safety_flags),self._state.temperature_c<43.0,self._state.tissue_confidence>=0.5 or self._state.tissue_confidence==0,self._state.bleeding_level<3]
        return sum(checks),len(checks)
    def emergency_stop(self, reason="手动触发"): return self.transition_to(SurgicalPhase.EMERGENCY_STOP,reason=reason,force=True)[0]
    def export_report(self):
        fe=time.time()-self._phase_start_time
        self._phase_durations[self._current_phase.value]=self._phase_durations.get(self._current_phase.value,0)+fe
        total=sum(self._phase_durations.values()); pn={}
        for val,dur in self._phase_durations.items():
            try: pn[SurgicalPhase(val).display]=dur
            except ValueError: pn[f"Phase({val})"]=dur
        self._report=WorkflowReport(transitions=self._transitions,total_duration_s=total,phase_durations=pn,safety_violations=sum(1 for t in self._transitions if t.safety_checks_passed<t.safety_checks_total),emergency_stops=self._report.emergency_stops,completion_status="complete" if self._current_phase==SurgicalPhase.COMPLETE else "incomplete")
        return self._report
    def export_trajectory(self, path):
        self.export_report(); os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        traj={"version":"4.4.0","report":self._report.to_dict(),"transitions":[{"from":t.from_phase.display,"to":t.to_phase.display,"reason":t.reason} for t in self._transitions],"current_phase":self._current_phase.display,"state":self._state.to_dict()}
        with open(path,"w") as f: json.dump(traj,f,indent=2,ensure_ascii=False)
    def __repr__(self): return f"SurgicalWorkflow({self._current_phase.display})"
