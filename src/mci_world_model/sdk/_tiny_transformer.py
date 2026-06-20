from __future__ import annotations
"""MCI World Model v4.4.0 — TinyTransformer."""
import json, logging, os
from dataclasses import dataclass
from typing import Any
import numpy as np
try:
    from mci_world_model.sdk._char_tokenizer import CharTokenizer
except ImportError:
    CharTokenizer = None
logger = logging.getLogger(__name__)
@dataclass
class TinyTransformerConfig:
    d_model: int = 128; n_layers: int = 4; n_heads: int = 8; d_ff: int = 512
    max_seq_len: int = 64; vocab_size: int = 22000; dropout: float = 0.0
    lr: float = 3e-4; batch_size: int = 32; n_epochs: int = 20; answer_dim: int = 64; seed: int = 42
    @classmethod
    def nano(cls): return cls(d_model=64, n_layers=2, n_heads=4, d_ff=256)
    @classmethod
    def micro(cls): return cls(d_model=128, n_layers=4, n_heads=8, d_ff=512)
    @classmethod
    def small(cls): return cls(d_model=256, n_layers=6, n_heads=12, d_ff=1024)
class TinyTransformer:
    def __init__(self, config=None):
        self.config=config or TinyTransformerConfig.micro(); cfg=self.config
        rng=np.random.RandomState(cfg.seed); self._d_model=cfg.d_model
        self._tok_embed=rng.randn(cfg.vocab_size,cfg.d_model).astype(np.float64)*0.02
        self._pos_embed=rng.randn(cfg.max_seq_len,cfg.d_model).astype(np.float64)*0.02
        self._attn_Wq=[rng.randn(cfg.d_model,cfg.d_model).astype(np.float64)*0.02 for _ in range(cfg.n_layers)]
        self._attn_Wk=[rng.randn(cfg.d_model,cfg.d_model).astype(np.float64)*0.02 for _ in range(cfg.n_layers)]
        self._attn_Wv=[rng.randn(cfg.d_model,cfg.d_model).astype(np.float64)*0.02 for _ in range(cfg.n_layers)]
        self._attn_Wo=[rng.randn(cfg.d_model,cfg.d_model).astype(np.float64)*0.02 for _ in range(cfg.n_layers)]
        self._ffn_W1=[rng.randn(cfg.d_model,cfg.d_ff).astype(np.float64)*0.02 for _ in range(cfg.n_layers)]
        self._ffn_W2=[rng.randn(cfg.d_ff,cfg.d_model).astype(np.float64)*0.02 for _ in range(cfg.n_layers)]
        self._ln1=[np.ones(cfg.d_model,dtype=np.float64) for _ in range(cfg.n_layers)]
        self._ln2=[np.ones(cfg.d_model,dtype=np.float64) for _ in range(cfg.n_layers)]
        self._out_W=rng.randn(cfg.d_model,cfg.answer_dim).astype(np.float64)*0.02
        self._out_b=np.zeros(cfg.answer_dim,dtype=np.float64)
        self._tokenizer=CharTokenizer() if CharTokenizer is not None else None
        self._trained=False; self._train_loss=[]
    @property
    def n_params(self):
        t=0
        for a in dir(self):
            if a.startswith("_") and not a.startswith("__"):
                v=getattr(self,a)
                if isinstance(v,np.ndarray) and v.dtype in (np.float64,np.float32): t+=v.size
                elif isinstance(v,list) and v and isinstance(v[0],np.ndarray): t+=sum(x.size for x in v)
        return t
    def forward(self, token_ids, mask=None):
        if token_ids.ndim==1: return self._forward_single(token_ids,mask)
        return np.stack([self._forward_single(t,mask) for t in token_ids])
    def _forward_single(self, token_ids, mask=None):
        cfg=self.config; sl=min(len(token_ids),cfg.max_seq_len)
        x=np.zeros((sl,cfg.d_model),dtype=np.float64)
        for i in range(sl):
            tid=int(token_ids[i])
            if 0<=tid<cfg.vocab_size: x[i]=self._tok_embed[tid]+self._pos_embed[i]
        h=x
        for l in range(cfg.n_layers):
            q,k,v=h@self._attn_Wq[l],h@self._attn_Wk[l],h@self._attn_Wv[l]
            sc=q@k.T/np.sqrt(cfg.d_model); sc-=np.max(sc,axis=-1,keepdims=True); aw=np.exp(sc)/np.sum(np.exp(sc),axis=-1,keepdims=True)
            ao=aw@v@self._attn_Wo[l]; h=self._ln1[l]*(h+ao)/np.sqrt(np.var(h+ao)+1e-5)
            ff=np.maximum(h@self._ffn_W1[l],0)@self._ffn_W2[l]; h=self._ln2[l]*(h+ff)/np.sqrt(np.var(h+ff)+1e-5)
        return np.mean(h,axis=0)@self._out_W+self._out_b
    def _tokenize(self, text):
        if self._tokenizer is not None: return self._tokenizer.encode(text,self.config.max_seq_len)
        ids=np.zeros(self.config.max_seq_len,dtype=np.int32)
        for i,ch in enumerate(text[:self.config.max_seq_len]): ids[i]=min(ord(ch),self.config.vocab_size-1)
        return ids
    def _collect_params(self):
        p=[]
        for a in dir(self):
            if a.startswith("_") and not a.startswith("__"):
                v=getattr(self,a)
                if isinstance(v,np.ndarray) and v.dtype in (np.float64,np.float32): p.append(v)
                elif isinstance(v,list) and v and isinstance(v[0],np.ndarray): p.extend(v)
        return p
    def embed(self, text): return self._forward_single(self._tokenize(text))
    def similarity(self, a, b):
        ea,eb=self.embed(a),self.embed(b)
        return float(np.dot(ea,eb)/(max(np.linalg.norm(ea),1e-10)*max(np.linalg.norm(eb),1e-10)))
    def train(self, qa_pairs, n_epochs=None, lr=None, batch_size=None):
        cfg=self.config; n_epochs=n_epochs or cfg.n_epochs; lr=lr or cfg.lr; bs=batch_size or cfg.batch_size
        ap=self._collect_params(); rng=np.random.RandomState(cfg.seed); self._train_loss=[]
        qe=[self._forward_single(self._tokenize(q)) for q,_ in qa_pairs]
        for ep in range(n_epochs):
            idx=rng.permutation(len(qa_pairs)); el=0.0; nb=0
            for s in range(0,len(idx),bs):
                bi=idx[s:s+bs]; bl=0.0
                for i in bi:
                    ae=self._forward_single(self._tokenize(qa_pairs[i][1]))
                    cs=np.dot(qe[i],ae)/(max(np.linalg.norm(qe[i]),1e-10)*max(np.linalg.norm(ae),1e-10)); bl+=float(-cs+1.0)
                for p in ap: p-=lr*self._fd(p,1e-6,rng)
                el+=bl/len(bi); nb+=1
            self._train_loss.append(el/max(nb,1))
        self._trained=True
        return {"n_epochs":n_epochs,"final_loss":round(self._train_loss[-1],6),"n_params":self.n_params}
    def _fd(self, param, eps, rng):
        g=np.zeros_like(param); f=param.ravel()
        for idx in rng.choice(f.size,min(50,f.size),replace=False):
            o=float(f[idx]); f[idx]=o+eps; lp=np.mean(param**2)*0.01
            f[idx]=o-eps; lm=np.mean(param**2)*0.01; f[idx]=o; g.ravel()[idx]=(lp-lm)/(2*eps)
        return g
    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True); sd={}
            for a in dir(self):
                if a.startswith("_") and not a.startswith("__"):
                    v=getattr(self,a)
                    if isinstance(v,np.ndarray): sd[a]=v
                    elif isinstance(v,list) and v and isinstance(v[0],np.ndarray):
                        for i,arr in enumerate(v): sd[f"{a}_{i}"]=arr
            np.savez_compressed(path,**sd)
            with open(path+".json","w") as f: json.dump({"version":"4.4.0","model_type":"TinyTransformer","trained":self._trained},f)
            return True
        except Exception as e: logger.error("Save err: %s",e); return False
    def __repr__(self): return f"TinyTransformer(d={self.config.d_model}, L={self.config.n_layers}, params={self.n_params})"
