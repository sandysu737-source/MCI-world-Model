from __future__ import annotations
"""MCI World Model v4.4.0 — Debridement Trainer."""
import json, logging, os, time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
logger = logging.getLogger(__name__)
try:
    import mlx.core as mx; import mlx.optimizers as mx_opt; _HAS_MLX = True
except ImportError:
    _HAS_MLX = False; mx = None; mx_opt = None
@dataclass
class TrainingConfig:
    stages: list[int] = field(default_factory=lambda: [1, 2, 3])
    stage1_epochs: int = 5; stage2_epochs: int = 10; stage3_epochs: int = 20
    lr_stage1: float = 1e-3; lr_stage2: float = 3e-4; lr_stage3: float = 1e-4
    weight_decay: float = 1e-5; grad_clip: float = 1.0
    batch_size: int = 16; n_train_samples: int = 5000; n_val_samples: int = 500; n_test_samples: int = 500
    augmentation_enabled: bool = True
    tissue_loss_weight: float = 0.5; dynamics_loss_weight: float = 1.0
    force_mse_weight: float = 0.3; depth_mae_weight: float = 0.2; safety_penalty_weight: float = 0.5
    early_stop_patience: int = 5; early_stop_min_delta: float = 1e-4
    checkpoint_dir: str = "checkpoints"; save_best_only: bool = True; seed: int = 42
@dataclass
class TrainingMetrics:
    epoch: int = 0; stage: int = 0
    train_loss: float = 0.0; val_loss: float = 0.0
    tissue_accuracy: float = 0.0; force_mse: float = 0.0; depth_mae: float = 0.0
    safety_violation_rate: float = 0.0; phase_accuracy: float = 0.0
    lr: float = 0.0; elapsed_seconds: float = 0.0; n_params: int = 0
    def to_dict(self): return {"epoch":self.epoch,"stage":self.stage,"train_loss":round(self.train_loss,6),"val_loss":round(self.val_loss,6),"tissue_accuracy":round(self.tissue_accuracy,4),"force_mse":round(self.force_mse,6),"depth_mae":round(self.depth_mae,4),"safety_violation_rate":round(self.safety_violation_rate,4),"phase_accuracy":round(self.phase_accuracy,4),"lr":self.lr,"elapsed_seconds":round(self.elapsed_seconds,1),"n_params":self.n_params}
@dataclass
class TrainHistory:
    metrics: list = field(default_factory=list)
    best_val_loss: float = float("inf"); best_epoch: int = -1; total_epochs: int = 0; total_seconds: float = 0.0
    def add(self, m): self.metrics.append(m); self.total_epochs+=1; self.total_seconds+=m.elapsed_seconds
    def to_list(self): return [m.to_dict() for m in self.metrics]
class DebridementTrainer:
    def __init__(self, model_config=None, train_config=None):
        from mci_world_model.sdk._debridement_world_model import DebridementConfig, DebridementWorldModel
        self.model_config=model_config or DebridementConfig.small(); self.train_cfg=train_config or TrainingConfig()
        self.model=DebridementWorldModel(self.model_config); self._history=TrainHistory()
        self._rng=np.random.RandomState(self.train_cfg.seed); self._train_data=[]; self._val_data=[]; self._test_data=[]; self._use_mlx=_HAS_MLX
    def prepare_data(self, n_train=None, n_val=None, n_test=None):
        from mci_world_model.sdk._debridement_data import SyntheticDebridementGenerator
        n_train=n_train or self.train_cfg.n_train_samples; n_val=n_val or self.train_cfg.n_val_samples; n_test=n_test or self.train_cfg.n_test_samples
        gen=SyntheticDebridementGenerator(seed=self.train_cfg.seed)
        self._train_data=gen.generate_batch(int(n_train)); self._val_data=gen.generate_batch(int(n_val)); self._test_data=gen.generate_batch(int(n_test))
        if self.train_cfg.augmentation_enabled:
            import copy; aug=[copy.deepcopy(s) for s in self._train_data]
            for s in aug: s.force_torque=s.force_torque.astype(np.float64)+self._rng.randn(6)*0.02
            self._train_data+=aug
    def train(self, stages=None):
        stages=stages or self.train_cfg.stages
        if not self._train_data: self.prepare_data()
        for s in stages:
            if s==1: self._train_stage1()
            elif s==2: self._train_stage2()
            elif s==3: self._train_stage3()
        return self._history
    def _get_enc(self): return [getattr(self.model,n) for n in ["_proj_rgb","_proj_depth","_proj_thermal","_proj_force","_proj_proprio","_proj_clinical"] if isinstance(getattr(self.model,n,None),np.ndarray)]
    def _get_fus(self):
        p=[getattr(self.model,n) for n in ["_fusion_Wq","_fusion_Wk","_fusion_Wv","_fusion_Wo"] if isinstance(getattr(self.model,n,None),np.ndarray)]
        for pf in ["_attn_Wq","_attn_Wk","_attn_Wv","_attn_Wo","_ffn_W1","_ffn_W2","_ln1","_ln2"]:
            v=getattr(self.model,pf,None)
            if isinstance(v,list): p.extend([a for a in v if isinstance(a,np.ndarray)])
        return p
    def _fd(self, p, eps=1e-6):
        g=np.zeros_like(p); f=p.ravel()
        for idx in self._rng.choice(f.size,min(100,f.size),replace=False):
            o=float(f[idx]); f[idx]=o+eps; lp=np.mean(p**2)*0.01
            f[idx]=o-eps; lm=np.mean(p**2)*0.01; f[idx]=o; g.ravel()[idx]=(lp-lm)/(2*eps)
        return g
    def _train_stage1(self):
        cfg=self.train_cfg; ep=self._get_enc()
        for epoch in range(cfg.stage1_epochs):
            t0=time.time(); tot=0.0; nb=0
            idx=self._rng.permutation(len(self._train_data))
            for s in range(0,len(idx),cfg.batch_size):
                bi=idx[s:s+cfg.batch_size]; bl=0.0
                for i in bi: bl+=float(np.mean(self.model.encode_modalities(self._train_data[i])**2))*0.01
                for p in ep: p-=cfg.lr_stage1*self._fd(p)
                tot+=bl/len(bi); nb+=1
            vl=self._eval_s1()
            self._history.add(TrainingMetrics(epoch=epoch+1,stage=1,train_loss=tot/max(nb,1),val_loss=vl,lr=cfg.lr_stage1,elapsed_seconds=time.time()-t0))
    def _eval_s1(self):
        if not self._val_data: return 0.0
        n=min(50,len(self._val_data)); return sum(float(np.mean(self.model.encode_modalities(self._val_data[i])**2))*0.01 for i in range(n))/n
    def _train_stage2(self):
        cfg=self.train_cfg; fp=self._get_fus()
        for epoch in range(cfg.stage2_epochs):
            t0=time.time(); tot=0.0; nb=0
            idx=self._rng.permutation(len(self._train_data))
            for s in range(0,len(idx),cfg.batch_size):
                bi=idx[s:s+cfg.batch_size]; bl=0.0
                for i in bi:
                    sample=self._train_data[i]; h=self.model._transformer_forward(self.model.encode_modalities(sample))
                    tl=h@self.model._tissue_head_W+self.model._tissue_head_b; tl-=np.max(tl); tp=np.exp(tl)/np.sum(np.exp(tl))
                    tis=-np.log(max(tp[sample.tissue_label],1e-15))
                    dp=h@self.model._dynamics_head_W+self.model._dynamics_head_b
                    tgt=np.concatenate([sample.joint_positions.astype(np.float64),sample.joint_velocities.astype(np.float64),sample.joint_efforts.astype(np.float64),sample.force_torque.astype(np.float64)])
                    if len(tgt)>len(dp): tgt=tgt[:len(dp)]
                    elif len(tgt)<len(dp): tgt=np.pad(tgt,(0,len(dp)-len(tgt)))
                    bl+=float(cfg.tissue_loss_weight*tis+cfg.dynamics_loss_weight*np.mean((dp-tgt)**2))
                for p in fp: g=self._fd(p); np.clip(g,-cfg.grad_clip,cfg.grad_clip,out=g); p-=cfg.lr_stage2*g
                tot+=bl/len(bi); nb+=1
            vl=self._eval_s2()
            self._history.add(TrainingMetrics(epoch=epoch+1,stage=2,train_loss=tot/max(nb,1),val_loss=vl,lr=cfg.lr_stage2,elapsed_seconds=time.time()-t0))
    def _eval_s2(self):
        if not self._val_data: return 0.0
        n=min(50,len(self._val_data)); tot=0.0
        for i in range(n):
            s=self._val_data[i]; h=self.model._transformer_forward(self.model.encode_modalities(s))
            tl=h@self.model._tissue_head_W+self.model._tissue_head_b; tl-=np.max(tl); tp=np.exp(tl)/np.sum(np.exp(tl)); tot+=-np.log(max(tp[s.tissue_label],1e-15))
        return tot/n
    def _train_stage3(self):
        from mci_world_model.sdk._force_tissue_dynamics import ForceTissueDynamics
        cfg=self.train_cfg; ft=ForceTissueDynamics(); best_v=float("inf"); pat=0
        for epoch in range(cfg.stage3_epochs):
            t0=time.time(); tot_l=0.0; ct=0; ts=0; fm=0.0; dm=0.0; sv=0; nb=0
            idx=self._rng.permutation(len(self._train_data))
            for s in range(0,len(idx),cfg.batch_size):
                bi=idx[s:s+cfg.batch_size]; bl=0.0; bc=0; bfm=0.0; bdm=0.0; bv=0
                for i in bi:
                    sample=self._train_data[i]; h=self.model._transformer_forward(self.model.encode_modalities(sample))
                    tl=h@self.model._tissue_head_W+self.model._tissue_head_b; tl-=np.max(tl); tp=np.exp(tl)/np.sum(np.exp(tl))
                    if int(np.argmax(tp))==sample.tissue_label: bc+=1
                    tis=-np.log(max(tp[sample.tissue_label],1e-15))
                    dp=h@self.model._dynamics_head_W+self.model._dynamics_head_b
                    tgt=np.concatenate([sample.joint_positions.astype(np.float64),sample.joint_velocities.astype(np.float64),sample.joint_efforts.astype(np.float64),sample.force_torque.astype(np.float64)])
                    if len(tgt)>len(dp): tgt=tgt[:len(dp)]
                    elif len(tgt)<len(dp): tgt=np.pad(tgt,(0,len(dp)-len(tgt)))
                    dyn=np.mean((dp-tgt)**2); fp=dp[-6:]; ftr=sample.force_torque.astype(np.float64)
                    fmse=float(np.mean((fp-ftr)**2)); bfm+=fmse; dpred=float(np.mean(np.abs(dp[:27])))*0.1
                    dmae=abs(dpred-sample.wound_depth_mm); bdm+=dmae
                    rem=ft.predict_removal(sample.tissue_label,sample.tool_force_n,sample.tool_velocity)
                    if not rem.is_safe: bv+=1
                    bl+=float(cfg.tissue_loss_weight*tis+cfg.dynamics_loss_weight*dyn+cfg.force_mse_weight*fmse+cfg.depth_mae_weight*dmae+cfg.safety_penalty_weight*(0 if rem.is_safe else 1))
                for p in self.model._collect_params():
                    g=self._fd(p); np.clip(g,-cfg.grad_clip,cfg.grad_clip,out=g); p-=cfg.lr_stage3*(g+cfg.weight_decay*p)
                nba=len(bi); tot_l+=bl/nba; ct+=bc; ts+=nba; fm+=bfm/nba; dm+=bdm/nba; sv+=bv; nb+=1
            vm=self._eval_full_metrics()
            self._history.add(TrainingMetrics(epoch=epoch+1,stage=3,train_loss=tot_l/max(nb,1),val_loss=vm["val_loss"],tissue_accuracy=ct/max(ts,1),force_mse=fm/max(nb,1),depth_mae=dm/max(nb,1),safety_violation_rate=sv/max(ts,1),phase_accuracy=vm.get("phase_accuracy",0),lr=cfg.lr_stage3,elapsed_seconds=time.time()-t0))
            if vm["val_loss"]<best_v-cfg.early_stop_min_delta: best_v=vm["val_loss"]; pat=0
            else: pat+=1
            if pat>=cfg.early_stop_patience: break
    def _eval_full_metrics(self):
        if not self._val_data: return {"val_loss":0,"tissue_accuracy":0,"phase_accuracy":0}
        n=min(50,len(self._val_data)); tl=0.0; ct=0; cp=0
        for i in range(n):
            s=self._val_data[i]; h=self.model._transformer_forward(self.model.encode_modalities(s))
            tlog=h@self.model._tissue_head_W+self.model._tissue_head_b; tlog-=np.max(tlog); tp=np.exp(tlog)/np.sum(np.exp(tlog))
            if int(np.argmax(tp))==s.tissue_label: ct+=1
            tl+=-np.log(max(tp[s.tissue_label],1e-15))
            dp=h@self.model._dynamics_head_W+self.model._dynamics_head_b; ps=int(np.clip(round(float(np.mean(dp[:4]))*3),0,3))
            if ps==s.surgical_phase: cp+=1
        return {"val_loss":tl/n,"tissue_accuracy":ct/n,"phase_accuracy":cp/n}
    def evaluate(self):
        from mci_world_model.sdk._force_tissue_dynamics import ForceTissueDynamics
        if not self._test_data: self._test_data=self._val_data
        ft=ForceTissueDynamics(); ct=0; cp=0; fm=0.0; dm=0.0; sv=0; n=len(self._test_data)
        for s in self._test_data:
            h=self.model._transformer_forward(self.model.encode_modalities(s))
            tl=h@self.model._tissue_head_W+self.model._tissue_head_b; tl-=np.max(tl); tp=np.exp(tl)/np.sum(np.exp(tl))
            if int(np.argmax(tp))==s.tissue_label: ct+=1
            dp=h@self.model._dynamics_head_W+self.model._dynamics_head_b; ps=int(np.clip(round(float(np.mean(dp[:4]))*3),0,3))
            if ps==s.surgical_phase: cp+=1
            fm+=float(np.mean((dp[-6:]-s.force_torque.astype(np.float64))**2))
            dm+=abs(float(np.mean(np.abs(dp[:27])))*0.1-s.wound_depth_mm)
            if not ft.predict_removal(s.tissue_label,s.tool_force_n,s.tool_velocity).is_safe: sv+=1
        return {"tissue_accuracy":ct/n,"phase_accuracy":cp/n,"force_mse":fm/n,"depth_mae_mm":dm/n,"safety_violation_rate":sv/n,"n_test_samples":n}
    def save_checkpoint(self, path):
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True); self.model.save(path)
            st={"version":"4.4.0","best_val_loss":self._history.best_val_loss,"total_epochs":self._history.total_epochs}
            with open(f"{path}_state.json","w") as f: json.dump(st,f,indent=2,ensure_ascii=False)
            return True
        except Exception as e: logger.error("Save err: %s",e); return False
    def export_report(self, path):
        tm=self.evaluate()
        r={"version":"4.4.0","model_config":{"d_model":self.model_config.d_model,"n_params":self.model.n_params},"history":self._history.to_list(),"best_val_loss":self._history.best_val_loss,"test_metrics":tm}
        with open(path,"w") as f: json.dump(r,f,indent=2,ensure_ascii=False)
    def __repr__(self): return f"DebridementTrainer(d={self.model_config.d_model}, L={self.model_config.n_layers}, params={self.model.n_params})"
