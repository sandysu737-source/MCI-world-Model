from __future__ import annotations
"""MCI World Model v4.4.0 — TinyTrainer."""
import json, logging, os, time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
logger = logging.getLogger(__name__)
@dataclass
class TinyTrainConfig:
    n_epochs: int = 30; lr: float = 3e-4; batch_size: int = 16; val_split: float = 0.1
    early_stop_patience: int = 5; early_stop_min_delta: float = 1e-4
    checkpoint_dir: str = "checkpoints"; seed: int = 42
@dataclass
class TinyTrainMetrics:
    epoch: int = 0; train_loss: float = 0.0; val_loss: float = 0.0; val_accuracy: float = 0.0
    lr: float = 0.0; elapsed_seconds: float = 0.0
    def to_dict(self): return {"epoch":self.epoch,"train_loss":round(self.train_loss,6),"val_loss":round(self.val_loss,6),"val_accuracy":round(self.val_accuracy,4),"lr":self.lr,"elapsed_seconds":round(self.elapsed_seconds,1)}
class TinyTrainer:
    def __init__(self, model_config=None, train_config=None):
        from mci_world_model.sdk._tiny_transformer import TinyTransformer, TinyTransformerConfig
        self.model_config=model_config or TinyTransformerConfig.micro(); self.train_cfg=train_config or TinyTrainConfig()
        self.model=TinyTransformer(self.model_config); self._history=[]; self._qa_pairs=[]; self._rng=np.random.RandomState(self.train_cfg.seed)
    def load_qa_data(self, path):
        self._qa_pairs=[]
        with open(path,encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    item=json.loads(line); q=item.get("question",""); a=item.get("answer","")
                    if q and a: self._qa_pairs.append((q,a))
                except json.JSONDecodeError: continue
        return len(self._qa_pairs)
    def train(self):
        if not self._qa_pairs: return []
        n_val=max(1,int(len(self._qa_pairs)*self.train_cfg.val_split))
        idx=self._rng.permutation(len(self._qa_pairs)); tr,va=idx[:-n_val],idx[-n_val:]; best_v=float("inf"); pat=0
        for ep in range(self.train_cfg.n_epochs):
            t0=time.time(); tl=self._train_epoch(tr); vl=self._validate(va); va_acc=self._qa_acc(va)
            self._history.append(TinyTrainMetrics(epoch=ep+1,train_loss=tl,val_loss=vl,val_accuracy=va_acc,lr=self.train_cfg.lr,elapsed_seconds=time.time()-t0))
            if vl<best_v-self.train_cfg.early_stop_min_delta: best_v=vl; pat=0
            else: pat+=1
            if pat>=self.train_cfg.early_stop_patience: break
        self.model._trained=True; return self._history
    def _train_epoch(self, indices):
        cfg=self.train_cfg; total=0.0; nb=0
        perm=self._rng.permutation(len(indices)); ap=self.model._collect_params()
        for s in range(0,len(perm),cfg.batch_size):
            bi=perm[s:s+cfg.batch_size]; bl=0.0
            for i in bi:
                q,a=self._qa_pairs[indices[i]]
                qe=self.model._forward_single(self.model._tokenize(q)); ae=self.model._forward_single(self.model._tokenize(a))
                cs=np.dot(qe,ae)/(max(np.linalg.norm(qe),1e-10)*max(np.linalg.norm(ae),1e-10)); bl+=float(-cs+1.0)
            for p in ap: p-=cfg.lr*self.model._fd(p,1e-6,self._rng)
            total+=bl/len(bi); nb+=1
        return total/max(nb,1)
    def _validate(self, indices):
        n=min(50,len(indices)); total=0.0
        for i in range(n):
            q,a=self._qa_pairs[indices[i]]
            qe=self.model._forward_single(self.model._tokenize(q)); ae=self.model._forward_single(self.model._tokenize(a))
            cs=np.dot(qe,ae)/(max(np.linalg.norm(qe),1e-10)*max(np.linalg.norm(ae),1e-10)); total+=float(-cs+1.0)
        return total/n
    def _qa_acc(self, indices):
        n=min(30,len(indices)); correct=0; ae={}
        for i in indices[:n]: _,a=self._qa_pairs[i]; ae[i]=self.model._forward_single(self.model._tokenize(a))
        for i in range(n):
            q,_=self._qa_pairs[indices[i]]; qe=self.model._forward_single(self.model._tokenize(q)); best=-2.0; bj=-1
            for j in indices[:n]:
                if j not in ae: continue
                cs=np.dot(qe,ae[j])/(max(np.linalg.norm(qe),1e-10)*max(np.linalg.norm(ae[j]),1e-10))
                if cs>best: best=cs; bj=j
            if bj==indices[i]: correct+=1
        return correct/n
    def save_checkpoint(self, path):
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True); self.model.save(path)
            st={"version":"4.4.0","history":[m.to_dict() for m in self._history],"n_qa":len(self._qa_pairs)}
            with open(f"{path}_state.json","w") as f: json.dump(st,f,indent=2,ensure_ascii=False)
            return True
        except Exception as e: logger.error("Save err: %s",e); return False
    def __repr__(self): return f"TinyTrainer(d={self.model_config.d_model}, qa={len(self._qa_pairs)})"
