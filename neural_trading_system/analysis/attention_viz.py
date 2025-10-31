
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ArrayLike = Union[np.ndarray, torch.Tensor]

class AttentionAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Any,
        device: str = "cpu",
        auto_scale: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.feature_extractor = feature_extractor
        self.device = torch.device(device)
        self.auto_scale = auto_scale
        self.config = config or {}
        self.seq_len = int(self.config.get("seq_len", 100))

        self._hooks: List[Any] = []
        self._captured_attn: List[torch.Tensor] = []
        self._register_attention_hooks()

    def _register_attention_hooks(self) -> None:
        for h in self._hooks:
            try: h.remove()
            except Exception: pass
        self._hooks, self._captured_attn = [], []

        def hook_factory(_name: str):
            def hook(module, _inp, out):
                attn = None
                if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[1]):
                    attn = out[1]
                if attn is None and hasattr(module, "attn_output_weights"):
                    attn = getattr(module, "attn_output_weights")
                if torch.is_tensor(attn):
                    self._captured_attn.append(attn.detach().cpu())
            return hook

        for name, m in self.model.named_modules():
            if isinstance(m, nn.MultiheadAttention):
                self._hooks.append(m.register_forward_hook(hook_factory(name)))
        for name, m in self.model.named_modules():
            lname = name.lower()
            if ("attention" in lname or "attn" in lname) and not isinstance(m, nn.MultiheadAttention):
                self._hooks.append(m.register_forward_hook(hook_factory(name)))

    def _ensure_parent(self, p: Optional[str]) -> None:
        if p: Path(p).parent.mkdir(parents=True, exist_ok=True)

    def _maybe_scale(self, x_np: np.ndarray) -> np.ndarray:
        if not self.auto_scale:
            return x_np.astype(np.float32, copy=False)
        scaler = getattr(self.feature_extractor, "scaler", None)
        if scaler is None or not hasattr(scaler, "n_features_in_"):
            return x_np.astype(np.float32, copy=False)
        F = x_np.shape[-1]; S = int(scaler.n_features_in_)
        if F == S:
            return scaler.transform(x_np).astype(np.float32, copy=False)
        if F + 1 == S:
            pad = np.zeros((x_np.shape[0], 1), dtype=np.float32)
            y = scaler.transform(np.hstack([x_np.astype(np.float32), pad])).astype(np.float32, copy=False)
            return y[:, :F]
        if F - 1 == S:
            y = scaler.transform(x_np[:, :S]).astype(np.float32, copy=False)
            return np.hstack([y, np.zeros((y.shape[0], 1), dtype=np.float32)])
        return x_np.astype(np.float32, copy=False)

    def _prepare_input(self, sample: ArrayLike) -> torch.Tensor:
\
\
\

        if isinstance(sample, torch.Tensor):
            x = sample.detach().to(self.device, dtype=torch.float32)
            if x.dim() == 1:
                x = x.view(1, 1, -1)
            elif x.dim() == 2:
                x = x.unsqueeze(0)
            elif x.dim() != 3:
                raise ValueError(f"Unsupported tensor dim {x.dim()}")
            if x.size(0) != 1:
                x = x[:1]
            return x

        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"Unsupported input ndim={arr.ndim}; expected 1 or 2.")
        arr = self._maybe_scale(arr)
        x = torch.from_numpy(arr).to(self.device)
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(0) != 1:
            x = x[:1]

        return x

    def _forward(self, x3: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            out = self.model(x3)
        preds: Dict[str, Any] = {}
        if isinstance(out, dict):
            for k, v in out.items():
                preds[k] = v.detach().cpu() if torch.is_tensor(v) else v
        elif isinstance(out, (list, tuple)):
            keys = ["entry_prob", "exit_prob", "expected_return", "volatility_forecast", "position_size"]
            for i, k in enumerate(keys[:len(out)]):
                v = out[i]; preds[k] = v.detach().cpu() if torch.is_tensor(v) else v
        elif torch.is_tensor(out):
            v = out.detach().cpu().view(-1)
            keys = ["entry_prob", "exit_prob", "expected_return", "volatility_forecast", "position_size"]
            for i, k in enumerate(keys[:len(v)]): preds[k] = v[i]

        if "entry_prob" not in preds and "entry_logits" in preds:
            el = preds["entry_logits"]
            preds["entry_prob"] = float(torch.sigmoid(el).item()) if torch.is_tensor(el) else 1/(1+np.exp(-float(el)))
        if "exit_prob" not in preds and "unified_exit_prob" in preds:
            preds["exit_prob"] = preds["unified_exit_prob"]
        for k, v in list(preds.items()):
            if isinstance(v, torch.Tensor) and v.numel() == 1: preds[k] = float(v.item())
        return preds

    def extract_attention_weights(self, sample_seq: ArrayLike) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        self._captured_attn = []
        x3 = self._prepare_input(sample_seq)
        preds = self._forward(x3)
        if not self._captured_attn and isinstance(preds, dict):
            maybe = preds.get("attention_weights", None)
            if torch.is_tensor(maybe):
                attn = maybe.detach().cpu()
                if attn.dim() == 4 and attn.size(0) == 1: attn = attn[0]
                self._captured_attn = [attn]
        return list(self._captured_attn), preds

    def compute_feature_importance(self, sequences: np.ndarray, num_samples: int = 100,
                                   feature_dim: Optional[int] = None, return_names: bool = False):
        self.model.eval()
        n = min(num_samples, len(sequences))
        if n == 0:
            vec = np.zeros(int(feature_dim or 1), dtype=np.float32)
            return (vec, [f"feature_{i}" for i in range(vec.size)]) if return_names else vec

        agg, count = None, 0
        for i in range(n):
            x3 = self._prepare_input(sequences[i]).clone().detach().requires_grad_(True)
            out = self.model(x3)
            if isinstance(out, dict):
                target = out.get("entry_logits", out.get("entry_prob", None))
            elif isinstance(out, (list, tuple)) and len(out) > 0:
                target = out[0]
            else:
                target = out.view(-1)[0] if torch.is_tensor(out) else None
            if target is None: continue

            self.model.zero_grad(set_to_none=True)
            if x3.grad is not None: x3.grad.zero_()
            target.view(-1)[0].backward()
            grad = x3.grad
            if grad is None: continue
            imp = grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
            agg = imp.astype(np.float32) if agg is None else (agg + imp)
            count += 1
        vec = np.zeros_like(agg) if count == 0 else agg / float(count)
        if feature_dim is not None and vec.shape[0] != int(feature_dim):
            d = int(feature_dim)
            vec = np.pad(vec, (0, d - vec.shape[0])) if vec.shape[0] < d else vec[:d]
        if return_names:
            names = [f"feature_{i}" for i in range(vec.shape[0])]
            return vec, names
        return vec

    def temporal_saliency_from_flat(self, flat_vec: np.ndarray) -> np.ndarray:
\
\
\
\

        x3 = self._prepare_input(flat_vec).clone().detach().requires_grad_(True)
        out = self.model(x3)
        if isinstance(out, dict):
            target = out.get("entry_logits", out.get("entry_prob", None))
        elif isinstance(out, (list, tuple)) and len(out) > 0:
            target = out[0]
        else:
            target = out.view(-1)[0] if torch.is_tensor(out) else None
        if target is None:
            return np.zeros(self.seq_len, dtype=np.float32)

        self.model.zero_grad(set_to_none=True)
        if x3.grad is not None: x3.grad.zero_()
        target.view(-1)[0].backward()
        grad = x3.grad.detach().cpu().view(-1).numpy()
        L = grad.size
        T = max(1, self.seq_len)
        if L % T != 0:

            chunk = L // T
            splits = [slice(i*chunk, (i+1)*chunk) for i in range(T-1)] + [slice((T-1)*chunk, L)]
        else:
            step = L // T
            splits = [slice(i*step, (i+1)*step) for i in range(T)]
        imp = np.array([np.mean(np.abs(grad[s])) for s in splits], dtype=np.float32)

        if imp.max() > 0: imp = imp / imp.max()
        return imp

    def temporal_occlusion_from_flat(self, flat_vec: np.ndarray, block: int = 5) -> Tuple[np.ndarray, np.ndarray]:
\
\
\
\
\
\

        base = float(self._forward(self._prepare_input(flat_vec)).get("entry_prob", np.nan))
        L = flat_vec.size
        T = max(1, self.seq_len)
        step = L // T if L >= T else 1
        deltas, probs = [], []
        for t in range(T):
            lo = t*step; hi = min(L, lo + block*step)
            x_mut = flat_vec.copy()
            x_mut[lo:hi] = 0.0
            p_mut = float(self._forward(self._prepare_input(x_mut)).get("entry_prob", np.nan))
            probs.append(p_mut)
            deltas.append(p_mut - base)
        return np.array(deltas, dtype=np.float32), np.array(probs, dtype=np.float32)

    def plot_attention_heatmap(self, attention_weights, layer_idx=-1, head_idx=0, save_path: Optional[str] = None):
        if isinstance(attention_weights, list) and len(attention_weights) > 0:
            attn = attention_weights[0]; attn = attn.detach().cpu() if torch.is_tensor(attn) else torch.as_tensor(attn)
        elif torch.is_tensor(attention_weights):
            attn = attention_weights.detach().cpu()
        else:
            attn = torch.as_tensor(attention_weights)
        if attn.dim() == 4 and attn.size(0) == 1: attn = attn[0]
        if attn.dim() == 3: attn2d = attn[head_idx if head_idx < attn.size(0) else 0]
        elif attn.dim() == 2: attn2d = attn
        else: raise ValueError(f"Unexpected attention shape: {tuple(attn.shape)}")
        a = attn2d.numpy()
        plt.figure(figsize=(11, 10))
        plt.imshow(a, aspect="auto")
        plt.title(f"Attention Heatmap — Layer {layer_idx}, Head {head_idx}")
        plt.xlabel("Key position"); plt.ylabel("Query position")
        plt.colorbar(); plt.tight_layout()
        if save_path: self._ensure_parent(save_path); plt.savefig(save_path, bbox_inches="tight"); plt.close()
        else: plt.show()

    def plot_attention_timeline(self, attention_weights_list: List[torch.Tensor], save_path: Optional[str] = None):
        if not attention_weights_list: return
        lens: List[int] = []
        for w in attention_weights_list:
            w = w.detach().cpu() if torch.is_tensor(w) else torch.as_tensor(w)
            if w.dim() == 4 and w.size(0) == 1: w = w[0]
            L = int(w.shape[-1]); lens.append(L)
        plt.figure(figsize=(14, 5.5))
        plt.plot(lens)
        plt.title("Attention map sizes per captured layer")
        plt.xlabel("Captured layer index"); plt.ylabel("Sequence length (T)")
        plt.tight_layout()
        if save_path: self._ensure_parent(save_path); plt.savefig(save_path, bbox_inches="tight"); plt.close()
        else: plt.show()

    def plot_feature_importance(self, importance_dict: Dict[str, float], top_n: int = 30, save_path: Optional[str] = None):
        items = sorted(importance_dict.items(), key=lambda x: -x[1])[:top_n]
        labels = [k for k, _ in items]; vals = [float(v) for _, v in items]
        y = np.arange(len(labels))[::-1]
        plt.figure(figsize=(11, 7.5))
        plt.barh(y, vals); plt.yticks(y, labels)
        plt.title(f"Top {top_n} Features (saliency)"); plt.tight_layout()
        if save_path: self._ensure_parent(save_path); plt.savefig(save_path, bbox_inches="tight"); plt.close()
        else: plt.show()

    def visualize_regime_space(self, test_sequences: np.ndarray, test_returns: np.ndarray,
                               max_points: int = 1000, save_path: Optional[str] = None) -> None:
        regime_embeddings: List[np.ndarray] = []; returns_list: List[float] = []
        take = min(max_points, len(test_sequences))
        for i in range(take):
            x3 = self._prepare_input(test_sequences[i])
            with torch.no_grad(): preds = self.model(x3)
            if isinstance(preds, dict) and "regime_z" in preds and torch.is_tensor(preds["regime_z"]):
                regime_embeddings.append(preds["regime_z"].detach().cpu().view(-1).numpy())
                returns_list.append(float(test_returns[i]))
        if not regime_embeddings: return
        regime_embeddings = np.array(regime_embeddings); returns_list = np.array(returns_list)

        from sklearn.decomposition import PCA
        regime_2d = PCA(n_components=2).fit_transform(regime_embeddings) if regime_embeddings.shape[1] > 2 else regime_embeddings
        plt.figure(figsize=(9.5, 7.5))
        sc = plt.scatter(regime_2d[:, 0], regime_2d[:, 1], c=returns_list, cmap="RdYlGn", s=20, alpha=0.75)
        plt.colorbar(sc, label="Future Return"); plt.xlabel("Regime Dim 1"); plt.ylabel("Regime Dim 2")
        plt.title("Learned Market Regime Space"); plt.grid(True, alpha=0.3); plt.tight_layout()
        if save_path: self._ensure_parent(save_path); plt.savefig(save_path, bbox_inches="tight"); plt.close()
        else: plt.show()

    def predict_entry_proba(self, X: np.ndarray) -> np.ndarray:
        if X is None or len(X) == 0: return np.array([], dtype=np.float32)
        probs: List[float] = []
        with torch.no_grad():
            for i in range(len(X)):
                x3 = self._prepare_input(X[i]); out = self.model(x3)
                p = None
                if isinstance(out, dict):
                    if "entry_prob" in out and torch.is_tensor(out["entry_prob"]):
                        p = float(out["entry_prob"].detach().cpu().view(-1)[-1].item())
                    elif "entry_logits" in out:
                        el = out["entry_logits"]
                        p = float(torch.sigmoid(el).detach().cpu().view(-1)[-1].item()) if torch.is_tensor(el) else 1/(1+np.exp(-float(el)))
                elif torch.is_tensor(out):
                    v = out.detach().cpu().view(-1); p = float(v[0].item()) if v.numel() > 0 else None
                probs.append(np.nan if p is None else p)
        return np.asarray(probs, dtype=np.float32)
