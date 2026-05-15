#!/usr/bin/env python
"""
AdaLoRA + Bayesian Optimization NAS  ── v3  (Act-LoRA Informed Prior)
═══════════════════════════════════════════════════════════════════════
核心新增：Act-LoRA 激活範數先驗測量
───────────────────────────────────────────────────────────────────────
v3 vs v2 的核心改動：

[Phase 0]  激活範數測量（Act-LoRA 體檢）
  ┌─────────────────────────────────────────────────────────────────┐
  │  在 BO 搜索開始之前，先對 frozen 的 base model 跑一次推論，     │
  │  記錄每一層 attention 輸出的激活範數（L2 norm）。               │
  │  範數大 → 該層對 task 更重要 → 應給更多 rank 預算。            │
  └─────────────────────────────────────────────────────────────────┘

[Phase 1]  Act-LoRA Prior → BO 搜索空間的初始猜測（warm start）
  ┌─────────────────────────────────────────────────────────────────┐
  │  根據測量結果，將 12 層分為 3 個重要性等級：                    │
  │    high  (top 33% 範數) → 給較大的 rank 搜索中心               │
  │    mid                  → 中等 rank                            │
  │    low   (bot 33% 範數) → 給較小的 rank                        │
  │  用 study.enqueue_trial() 把這個「有根據的猜測」作為             │
  │  第一個 trial，讓 TPE surrogate 從好的位置開始，而非純隨機。    │
  └─────────────────────────────────────────────────────────────────┘

[Phase 2]  BO 搜索（TPE Sampler + MedianPruner）
  同 v2，但搜索空間的上下界也根據先驗測量自動收窄：
  - 若先驗顯示各層差異小（std/mean < 0.15）→ 三區間 rank 差縮小
  - 若先驗顯示各層差異大                   → 允許更大的 rank span

[Phase 3]  正式訓練
  使用 BO 找到的 best_config + per-layer rank_pattern

修復（來自 v2）：
  ✅ rank_pattern 值改為 list[int]（修正 PEFT 0.7.1 TypeError）
  ✅ lr 範圍 5e-6 ~ 8e-5
  ✅ orth_reg_weight 0.1
  ✅ 詳細參數量印出
"""

import sys, os

loralib_path = os.path.join(os.path.dirname(__file__), 'loralib')
if loralib_path in sys.path:
    sys.path.remove(loralib_path)
if 'loralib' in sys.modules:
    del sys.modules['loralib']

print("=" * 80)
print("🔧 Import Configuration  [v3 — Act-LoRA Informed Prior]")
print("=" * 80)

import logging, json, gc
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import evaluate
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_callback import TrainerControl, TrainerState

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"✅ Optuna: {optuna.__version__}")
except ImportError as e:
    print(f"❌ Optuna not found: {e}"); sys.exit(1)

try:
    from peft import get_peft_model, LoraConfig, AdaLoraConfig, TaskType
    import peft
    print(f"✅ PEFT: {peft.__version__}")
except ImportError as e:
    print(f"❌ PEFT not found: {e}"); sys.exit(1)

print("=" * 80 + "\n")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Task helpers
# ─────────────────────────────────────────────────────────────────────────────
task_to_keys = {
    "cola": ("sentence", None),       "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),"qnli": ("question", "sentence"),
    "qqp":  ("question1", "question2"),"rte":  ("sentence1", "sentence2"),
    "sst2": ("sentence", None),        "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def resolve_target_modules(model_name: str) -> list:
    n = model_name.lower()
    if "deberta-v3" in n or ("deberta" in n and "v3" in n):
        return ["query_proj", "key_proj", "value_proj", "pos_query_proj"]
    if "deberta" in n:
        return ["in_proj", "pos_proj", "pos_q_proj"]
    return ["query", "value"]

def get_num_hidden_layers(model_name: str) -> int:
    if "large" in model_name.lower():
        return 24
    try:
        return AutoConfig.from_pretrained(model_name).num_hidden_layers
    except Exception:
        return 12

def load_tokenizer_with_fallback(model_name: str):
    for use_fast in [True, False]:
        try:
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
            print(f"[Tokenizer] Loaded ({'fast' if use_fast else 'slow'})")
            return tok
        except Exception as e:
            print(f"[Tokenizer Warning] {'Fast' if use_fast else 'Slow'} failed: {e}")
    if "deberta" in model_name.lower():
        from transformers import DebertaV2Tokenizer
        tok = DebertaV2Tokenizer.from_pretrained(model_name)
        print("[Tokenizer] Loaded DebertaV2Tokenizer")
        return tok
    raise RuntimeError("Tokenizer loading failed.")

def print_param_summary(model, label: str = "Model"):
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * trainable / total if total > 0 else 0.0
    print(f"\n📐 [{label}] Parameter Summary")
    print(f"   Trainable : {trainable:>12,}  ({ratio:.4f}%)")
    print(f"   Total     : {total:>12,}")
    print(f"   Frozen    : {total - trainable:>12,}")
    return trainable, total


# ═════════════════════════════════════════════════════════════════════════════
# ★ Phase 0：Act-LoRA 激活範數測量
# ═════════════════════════════════════════════════════════════════════════════
class ActivationNormMeasurer:
    """
    Act-LoRA 的「體檢」階段。

    做法：
    1. 載入 frozen base model（不加任何 LoRA）
    2. 對少量樣本（預設 256 筆）跑 forward pass
    3. 用 hook 捕捉每一層 attention 輸出的 L2 norm
    4. 對所有樣本取平均，得到每層的「重要性分數」
    5. 根據分數自動分配 rank 建議（high/mid/low）

    回傳 LayerImportance:
      layer_norms   : list[float]  — 每層的平均激活範數（長度 = num_layers）
      layer_ranks   : list[int]    — 根據範數建議的 per-layer rank
      prior_r_bottom: int          — 底層建議 rank（供 BO 搜索空間收窄用）
      prior_r_middle: int          — 中層建議 rank
      prior_r_top   : int          — 頂層建議 rank
      diversity     : float        — 各層重要性差異程度（std/mean）
    """

    def __init__(
        self,
        model_name:      str,
        task_name:       str,
        tokenizer,
        probe_dataset,            # 用來做 forward pass 的小樣本 dataset
        device:          torch.device,
        num_layers:      int,
        n_probe_samples: int  = 256,
        r_budget_total:  int  = 64,   # 所有層的 rank 預算總和（用於比例分配）
        local_files_only: bool = False,
    ):
        self.model_name       = model_name
        self.task_name        = task_name
        self.tokenizer        = tokenizer
        self.probe_dataset    = probe_dataset
        self.device           = device
        self.num_layers       = num_layers
        self.n_probe_samples  = n_probe_samples
        self.r_budget_total   = r_budget_total
        self.local_files_only = local_files_only

    # ──────────────────────────────────────────────────────────────────────────
    def _register_hooks(self, model) -> tuple[list, list]:
        """
        對每一個 Transformer encoder layer 的輸出（LayerNorm 前）掛 hook，
        記錄 batch 內每個樣本的輸出 L2 norm（取 [CLS] token 位置）。
        """
        hooks   = []
        records = defaultdict(list)   # layer_idx → list of norms

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output 可能是 tuple（如 DeBERTa attention output）
                hidden = output[0] if isinstance(output, tuple) else output
                # hidden shape: (batch, seq_len, hidden_dim)
                # 取 [CLS] token（位置 0）的 L2 norm
                cls_norm = hidden[:, 0, :].norm(dim=-1)   # (batch,)
                records[layer_idx].extend(cls_norm.detach().cpu().tolist())
            return hook_fn

        # 嘗試各種 DeBERTa / BERT 的 layer 存取路徑
        encoder_layers = None
        for attr_path in [
            "deberta.encoder.layer",
            "bert.encoder.layer",
            "roberta.encoder.layer",
            "encoder.layer",
        ]:
            parts = attr_path.split(".")
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                if isinstance(obj, nn.ModuleList):
                    encoder_layers = obj
                    break
            except AttributeError:
                continue

        if encoder_layers is None:
            print("[ActLoRA] ⚠️ Cannot locate encoder layers; skipping measurement.")
            return hooks, records

        for idx, layer in enumerate(encoder_layers):
            h = layer.register_forward_hook(make_hook(idx))
            hooks.append(h)

        return hooks, records

    # ──────────────────────────────────────────────────────────────────────────
    def measure(self) -> dict:
        """
        執行激活範數測量，回傳 prior dict。
        """
        print("\n" + "=" * 80)
        print("🔬 Phase 0: Act-LoRA Activation Norm Measurement")
        print("=" * 80)
        print(f"  Probe samples : {self.n_probe_samples}")
        print(f"  Num layers    : {self.num_layers}")
        print(f"  Rank budget   : {self.r_budget_total} (total across all layers)")

        is_regression = self.task_name == "stsb"
        num_labels    = 1 if is_regression else (3 if self.task_name == "mnli" else 2)

        # 載入 frozen base model（純推論，不加 LoRA）
        cfg   = AutoConfig.from_pretrained(
            self.model_name, num_labels=num_labels,
            local_files_only=self.local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=cfg,
            local_files_only=self.local_files_only,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(self.device)

        # 掛 hook
        hooks, records = self._register_hooks(model)

        # 取 probe subset（從 train_dataset 前 n 筆）
        n = min(self.n_probe_samples, len(self.probe_dataset))
        probe_subset = self.probe_dataset.select(range(n))
        loader = DataLoader(
            probe_subset,
            batch_size  = 32,
            collate_fn  = default_data_collator,
            shuffle     = False,
        )

        # Forward pass
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}
                try:
                    model(**batch)
                except Exception as e:
                    print(f"[ActLoRA] Forward pass warning: {e}")

        # 移除 hook
        for h in hooks:
            h.remove()

        # 計算每層平均範數
        layer_norms = []
        for i in range(self.num_layers):
            norms_i = records.get(i, [0.0])
            layer_norms.append(float(np.mean(norms_i)))

        # 清理
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # ── 根據範數分配 per-layer rank ───────────────────────────────────────
        norms_arr = np.array(layer_norms)
        if norms_arr.max() > 0:
            norm_ratio = norms_arr / norms_arr.sum()   # 歸一化為比例
        else:
            norm_ratio = np.ones(self.num_layers) / self.num_layers

        # 按比例分配 rank（最少 2，最多 16）
        raw_ranks = norm_ratio * self.r_budget_total
        layer_ranks = np.clip(np.round(raw_ranks).astype(int), 2, 16).tolist()

        # 三區間統計（底/中/頂）→ 供 BO warm start 用
        boundary_low  = self.num_layers // 3
        boundary_high = (self.num_layers * 2) // 3
        prior_r_bottom = int(round(np.mean(layer_ranks[:boundary_low])))
        prior_r_middle = int(round(np.mean(layer_ranks[boundary_low:boundary_high])))
        prior_r_top    = int(round(np.mean(layer_ranks[boundary_high:])))

        # 多樣性指標：std/mean，衡量各層重要性差異
        diversity = float(norms_arr.std() / norms_arr.mean()) if norms_arr.mean() > 0 else 0.0

        # ── 印出測量結果 ──────────────────────────────────────────────────────
        print(f"\n  Layer-wise Activation Norms (L2, CLS token):")
        for i, (n_val, r_val) in enumerate(zip(layer_norms, layer_ranks)):
            zone = "bot" if i < boundary_low else ("mid" if i < boundary_high else "top")
            bar  = "█" * int(n_val / max(layer_norms) * 20) if max(layer_norms) > 0 else ""
            print(f"    Layer {i:02d} [{zone}] norm={n_val:7.3f}  rank={r_val:2d}  {bar}")

        print(f"\n  Prior Summary:")
        print(f"    r_bottom (layers 0~{boundary_low-1})       : {prior_r_bottom}")
        print(f"    r_middle (layers {boundary_low}~{boundary_high-1})      : {prior_r_middle}")
        print(f"    r_top    (layers {boundary_high}~{self.num_layers-1})     : {prior_r_top}")
        print(f"    Diversity (std/mean)      : {diversity:.4f}")
        if diversity < 0.15:
            print(f"    → Low diversity: layers are similarly important; "
                  f"BO rank span will be narrowed.")
        else:
            print(f"    → High diversity: layers differ significantly; "
                  f"BO rank span will be wider.")
        print("=" * 80)

        return {
            "layer_norms":    layer_norms,
            "layer_ranks":    layer_ranks,
            "prior_r_bottom": prior_r_bottom,
            "prior_r_middle": prior_r_middle,
            "prior_r_top":    prior_r_top,
            "diversity":      diversity,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Rank Pattern builder（list 版，修正 PEFT 0.7.1 TypeError）
# ─────────────────────────────────────────────────────────────────────────────
def build_rank_pattern_from_peft_model(
    peft_model, r_bottom: int, r_middle: int, r_top: int, num_layers: int
) -> dict:
    """
    PEFT 0.7.1 的 resize_state_dict_by_rank_pattern 執行 sum(rank_idx)，
    rank_idx 必須是可迭代物件，因此這裡用 [int] 而非 int。
    """
    rank_pattern  = {}
    boundary_low  = num_layers // 3
    boundary_high = (num_layers * 2) // 3

    for name, module in peft_model.named_modules():
        if hasattr(module, 'lora_E') or hasattr(module, 'lora_A'):
            for layer_idx in range(num_layers):
                for pat in [
                    f"layer.{layer_idx}.",
                    f"layers.{layer_idx}.",
                    f".{layer_idx}.attention",
                ]:
                    if pat in name:
                        if layer_idx < boundary_low:
                            rank_pattern[name] = [r_bottom]
                        elif layer_idx < boundary_high:
                            rank_pattern[name] = [r_middle]
                        else:
                            rank_pattern[name] = [r_top]
                        break

    if not rank_pattern:
        print("[rank_pattern] ⚠️ No adapter modules matched; using global target_r.")
    else:
        counts: dict[int, int] = {}
        for v in rank_pattern.values():
            key = v[0] if isinstance(v, list) else v
            counts[key] = counts.get(key, 0) + 1
        print(
            f"[rank_pattern] ✅ {len(rank_pattern)} entries — "
            f"bot={r_bottom}×{counts.get(r_bottom,0)}, "
            f"mid={r_middle}×{counts.get(r_middle,0)}, "
            f"top={r_top}×{counts.get(r_top,0)}"
        )
    return rank_pattern


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────
class PruningException(Exception):
    pass

class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial, metric_key: str = "eval_accuracy"):
        self.trial      = trial
        self.metric_key = metric_key
        self._step      = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control
        value = metrics.get(self.metric_key)
        if value is None:
            return control
        self.trial.report(value, step=self._step)
        self._step += 1
        if self.trial.should_prune():
            print(f"\n[Optuna] Trial {self.trial.number} pruned "
                  f"at step={self._step}, {self.metric_key}={value:.4f}")
            control.should_training_stop = True
            raise PruningException(f"Pruned at step {self._step}")
        return control

class AdaLoraCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        tgt = model
        for attr in ["base_model", "model"]:
            if hasattr(tgt, attr):
                cand = getattr(tgt, attr)
                tgt  = cand.base_model if hasattr(cand, "base_model") else cand
                break
        if hasattr(tgt, "update_and_allocate"):
            try:
                tgt.update_and_allocate(state.global_step)
                if state.global_step % 100 == 0:
                    total_r = sum(
                        (m.r.get('default', 0) if isinstance(m.r, dict) else m.r)
                        for m in tgt.modules() if hasattr(m, 'r')
                    )
                    print(f"\n[AdaLoRA] Step {state.global_step}: Total Rank = {total_r}")
            except Exception as e:
                if state.global_step % 200 == 0:
                    print(f"\n[AdaLoRA Warning] {e}")
        return control

class AdaLoraTrainer(Trainer):
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        tgt  = model
        if hasattr(tgt, "module"):       tgt = tgt.module
        if hasattr(tgt, "base_model"):   tgt = tgt.base_model
        if hasattr(tgt, "model") and hasattr(tgt.model, "base_model"):
            tgt = tgt.model.base_model
        if hasattr(tgt, "update_and_allocate"):
            try:
                tgt.update_and_allocate(self.state.global_step)
            except Exception as e:
                if self.state.global_step % 100 == 0:
                    print(f"\n[AdaLoRA Warning] {e}")
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        kw = {"num_items_in_batch": num_items_in_batch} if num_items_in_batch is not None else {}
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kw)
        tgt = model
        if hasattr(tgt, "base_model"): tgt = tgt.base_model
        if hasattr(tgt, "model") and hasattr(tgt.model, "base_model"):
            tgt = tgt.model.base_model
        if hasattr(tgt, "get_orth_regu_loss"):
            try:
                loss = loss + tgt.get_orth_regu_loss()
            except Exception:
                pass
        return (loss, outputs) if return_outputs else loss


# ═════════════════════════════════════════════════════════════════════════════
# ★ Phase 1 + 2：Act-LoRA Informed BO Search
# ═════════════════════════════════════════════════════════════════════════════
class BONASSearch:
    """
    v3 新增：
      - __init__ 接受 act_prior（Phase 0 測量結果）
      - _compute_search_bounds()：根據先驗自動設定 rank 搜索上下界
      - search() 開頭用 study.enqueue_trial() 注入先驗猜測（warm start）
      - _suggest_params() 中 r_bottom/r_middle/r_top 的範圍依 prior 動態調整
    """

    def __init__(
        self,
        model_name:      str,
        task_name:       str,
        tokenizer,
        train_dataset,
        eval_dataset,
        device:          torch.device,
        act_prior:       dict,          # ★ Phase 0 的測量結果
        use_adalora:     bool = True,
        n_trials:        int  = 20,
        n_startup:       int  = 5,
        eval_steps:      int  = 800,
        local_files_only: bool = False,
    ):
        self.model_name       = model_name
        self.task_name        = task_name
        self.tokenizer        = tokenizer
        self.train_dataset    = train_dataset
        self.eval_dataset     = eval_dataset
        self.device           = device
        self.act_prior        = act_prior   # ★
        self.use_adalora      = use_adalora
        self.n_trials         = n_trials
        self.n_startup        = n_startup
        self.eval_steps       = eval_steps
        self.local_files_only = local_files_only
        self.target_modules   = resolve_target_modules(model_name)
        self.num_layers       = get_num_hidden_layers(model_name)

        self.metric   = evaluate.load("glue", self.task_name)
        self.history: list[dict] = []
        self.best_score:        float = -float('inf')
        self.best_config:       dict  = {}
        self.best_params:       int   = 0
        self.best_rank_pattern: dict  = {}

        # ── 根據先驗計算搜索邊界 ─────────────────────────────────────────────
        self.bounds = self._compute_search_bounds()

        print(f"[TargetModules] {self.target_modules}")
        print(f"[NumLayers]     {self.num_layers}")
        print(f"[BO Config]     n_trials={n_trials}, n_startup={n_startup}, eval_steps={eval_steps}")
        print(f"[SearchBounds]  r_bottom={self.bounds['r_bot_lo']}~{self.bounds['r_bot_hi']}, "
              f"r_middle={self.bounds['r_mid_lo']}~{self.bounds['r_mid_hi']}, "
              f"r_top={self.bounds['r_top_lo']}~{self.bounds['r_top_hi']}")

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 根據 Act-LoRA 先驗自動收窄搜索空間
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_search_bounds(self) -> dict:
        """
        利用 act_prior 的 prior_r_bottom/middle/top 作為搜索中心，
        根據 diversity 決定搜索半徑：
          - diversity 高（各層差異大）→ 半徑 ±4，允許更大探索
          - diversity 低（各層差異小）→ 半徑 ±2，集中搜索
        這樣 BO 就不需要從整個空間盲目搜索，而是在有根據的區域附近探索。
        """
        p          = self.act_prior
        diversity  = p.get("diversity", 0.3)
        radius     = 4 if diversity >= 0.15 else 2

        def clamp(lo, hi, min_val=2, max_val=16):
            return max(min_val, lo), min(max_val, hi)

        r_bot = p.get("prior_r_bottom", 4)
        r_mid = p.get("prior_r_middle", 6)
        r_top = p.get("prior_r_top",    8)

        r_bot_lo, r_bot_hi = clamp(r_bot - radius, r_bot + radius)
        r_mid_lo, r_mid_hi = clamp(r_mid - radius, r_mid + radius)
        r_top_lo, r_top_hi = clamp(r_top - radius, r_top + radius)

        # 確保三區間有意義的順序（bottom ≤ middle ≤ top，至少差 1）
        r_mid_lo = max(r_mid_lo, r_bot_lo)
        r_top_lo = max(r_top_lo, r_mid_lo)

        return {
            "r_bot_lo": r_bot_lo, "r_bot_hi": r_bot_hi,
            "r_mid_lo": r_mid_lo, "r_mid_hi": r_mid_hi,
            "r_top_lo": r_top_lo, "r_top_hi": r_top_hi,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 搜索空間定義（動態邊界）
    # ──────────────────────────────────────────────────────────────────────────
    def _suggest_params(self, trial: optuna.Trial) -> dict:
        b = self.bounds
        p = self.act_prior

        r_bottom = trial.suggest_int("target_r_bottom", b["r_bot_lo"], b["r_bot_hi"])
        r_middle = trial.suggest_int("target_r_middle", b["r_mid_lo"], b["r_mid_hi"])
        r_top    = trial.suggest_int("target_r_top",    b["r_top_lo"], b["r_top_hi"])

        min_lora_r = max(r_bottom, r_middle, r_top) + 2
        lora_r     = trial.suggest_int("lora_r", min_lora_r, max(min_lora_r, 16))
        lora_alpha = trial.suggest_int("lora_alpha", 8, 32)
        lr         = trial.suggest_float("learning_rate", 5e-6, 8e-5, log=True)
        tinit_r    = trial.suggest_float("tinit_ratio",  0.10, 0.30)
        tfinal_r   = trial.suggest_float("tfinal_ratio", 0.65, 0.85)

        avg_r = int(round((r_bottom + r_middle + r_top) / 3))
        return {
            "lora_r": lora_r, "lora_alpha": lora_alpha,
            "learning_rate": lr, "avg_target_r": avg_r,
            "r_bottom": r_bottom, "r_middle": r_middle, "r_top": r_top,
            "tinit_ratio": tinit_r, "tfinal_ratio": tfinal_r,
            "debug_info": (
                f"r={lora_r}, target(bot/mid/top)={r_bottom}/{r_middle}/{r_top}, "
                f"alpha={lora_alpha}, lr={lr:.2e}"
            ),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 單次 Trial 評估
    # ──────────────────────────────────────────────────────────────────────────
    def _objective(self, trial: optuna.Trial) -> float:
        config = self._suggest_params(trial)
        print(f"\n🔬 Trial {trial.number + 1}/{self.n_trials} | {config['debug_info']}")

        is_regression = self.task_name == "stsb"
        num_labels    = 1 if is_regression else (3 if self.task_name == "mnli" else 2)

        try:
            model_cfg = AutoConfig.from_pretrained(
                self.model_name, num_labels=num_labels,
                finetuning_task=self.task_name,
                local_files_only=self.local_files_only,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, config=model_cfg,
                local_files_only=self.local_files_only,
            )

            tinit  = int(self.eval_steps * config["tinit_ratio"])
            tfinal = int(self.eval_steps * config["tfinal_ratio"])

            if self.use_adalora:
                peft_cfg = AdaLoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    lora_alpha=config["lora_alpha"],
                    lora_dropout=0.1,
                    target_modules=self.target_modules,
                    init_r=config["lora_r"],
                    target_r=config["avg_target_r"],
                    total_step=self.eval_steps,
                    tinit=tinit, tfinal=tfinal,
                    deltaT=10, orth_reg_weight=0.1,
                )
            else:
                peft_cfg = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=config["lora_r"], lora_alpha=config["lora_alpha"],
                    lora_dropout=0.1, target_modules=self.target_modules,
                )

            model = get_peft_model(model, peft_cfg)

            rank_pattern = build_rank_pattern_from_peft_model(
                model, config["r_bottom"], config["r_middle"],
                config["r_top"], self.num_layers,
            )
            if rank_pattern:
                try:
                    model.peft_config["default"].rank_pattern = rank_pattern
                except Exception:
                    pass

            for name, param in model.named_parameters():
                if "classifier" in name or "score" in name:
                    param.requires_grad = True

            trainable_params, total_params = print_param_summary(
                model, f"Trial {trial.number + 1}"
            )
            model.to(self.device)

            metric_key = {
                "stsb": "eval_pearson",
                "cola": "eval_matthews_correlation",
            }.get(self.task_name, "eval_accuracy")

            training_args = TrainingArguments(
                output_dir="./bo_temp",
                num_train_epochs=1,
                max_steps=self.eval_steps,
                per_device_train_batch_size=16,
                gradient_accumulation_steps=2,
                per_device_eval_batch_size=64,
                learning_rate=config["learning_rate"],
                logging_steps=100,
                save_strategy="no",
                evaluation_strategy="steps",
                eval_steps=max(100, self.eval_steps // 4),
                report_to="none",
                no_cuda=(self.device.type == "cpu"),
                disable_tqdm=True,
                lr_scheduler_type="cosine",
                warmup_ratio=0.06,
                weight_decay=0.01,
                dataloader_num_workers=0,
            )

            def compute_metrics(p: EvalPrediction):
                preds = (np.squeeze(p.predictions) if is_regression
                         else np.argmax(p.predictions, axis=1))
                return self.metric.compute(predictions=preds, references=p.label_ids)

            callbacks = [OptunaPruningCallback(trial, metric_key=metric_key)]
            if self.use_adalora:
                callbacks.append(AdaLoraCallback())

            TrainerClass = AdaLoraTrainer if self.use_adalora else Trainer
            trainer = TrainerClass(
                model=model, args=training_args,
                train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer, data_collator=default_data_collator,
                compute_metrics=compute_metrics, callbacks=callbacks,
            )
            trainer.train()

            try:
                eval_results = trainer.evaluate()
                accuracy = eval_results.get(
                    "eval_pearson"               if self.task_name == "stsb"
                    else "eval_matthews_correlation" if self.task_name == "cola"
                    else "eval_accuracy", 0.0
                )
            except Exception:
                accuracy = 0.0

            penalty = (trainable_params / 100_000) * 0.01
            fitness = accuracy - penalty

            print(f"   ✅ Acc={accuracy:.4f}, Params={trainable_params:,}, "
                  f"Penalty={penalty:.4f}, Fitness={fitness:.4f}")

            self.history.append({
                "trial": trial.number + 1, "fitness": fitness,
                "accuracy": accuracy,
                "trainable_params": trainable_params, "total_params": total_params,
                "config": {k: v for k, v in config.items() if k != "debug_info"},
                "rank_pattern": {
                    str(k): (v[0] if isinstance(v, list) else v)
                    for k, v in rank_pattern.items()
                },
            })

            if fitness > self.best_score:
                self.best_score        = fitness
                self.best_config       = config
                self.best_params       = trainable_params
                self.best_rank_pattern = rank_pattern
                print(f"   🏆 New best! Fitness={fitness:.4f}")

            del model, trainer
            torch.cuda.empty_cache(); gc.collect()
            return fitness

        except PruningException:
            raise optuna.exceptions.TrialPruned()
        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            import traceback; traceback.print_exc()
            return -1.0

    # ──────────────────────────────────────────────────────────────────────────
    # ★ 主搜索迴圈（含 warm start）
    # ──────────────────────────────────────────────────────────────────────────
    def search(self):
        print("\n" + "=" * 80)
        print("🤖 Phase 1+2: Act-LoRA Informed BO Search  [v3]")
        print("=" * 80)

        sampler = TPESampler(
            n_startup_trials=self.n_startup,
            multivariate=True,
            seed=42,
        )
        pruner = MedianPruner(
            n_startup_trials=max(3, self.n_startup),
            n_warmup_steps=2,
        )
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )

        # ★ warm start：把 Act-LoRA 先驗猜測注入為第一個 trial
        p        = self.act_prior
        prior_lr = 3e-5   # Act-LoRA 不測 lr，用固定合理值

        prior_r_bot = p["prior_r_bottom"]
        prior_r_mid = p["prior_r_middle"]
        prior_r_top = p["prior_r_top"]
        min_r       = max(prior_r_bot, prior_r_mid, prior_r_top) + 2
        prior_init_r = min(max(min_r, 10), 16)
        prior_alpha  = max(prior_init_r, 16)   # alpha ≥ init_r 是慣例

        warm_start_params = {
            "target_r_bottom": max(self.bounds["r_bot_lo"],
                                   min(prior_r_bot, self.bounds["r_bot_hi"])),
            "target_r_middle": max(self.bounds["r_mid_lo"],
                                   min(prior_r_mid, self.bounds["r_mid_hi"])),
            "target_r_top":    max(self.bounds["r_top_lo"],
                                   min(prior_r_top, self.bounds["r_top_hi"])),
            "lora_r":          prior_init_r,
            "lora_alpha":      prior_alpha,
            "learning_rate":   prior_lr,
            "tinit_ratio":     0.20,
            "tfinal_ratio":    0.75,
        }
        study.enqueue_trial(warm_start_params)
        print(f"  [Warm Start] Act-LoRA prior enqueued as Trial #1:")
        print(f"    r_bottom={warm_start_params['target_r_bottom']}, "
              f"r_middle={warm_start_params['target_r_middle']}, "
              f"r_top={warm_start_params['target_r_top']}, "
              f"init_r={prior_init_r}, alpha={prior_alpha}")
        print("=" * 80)

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        print("\n" + "=" * 80)
        print("📊 BO Search Summary  [v3]")
        print("=" * 80)
        print(f"  Completed: {len(completed)} | Pruned: {len(pruned)} | Failed: {len(failed)}")
        if completed:
            print(f"  Best Fitness: {study.best_value:.4f}")
            print(f"  Best Trial : #{study.best_trial.number + 1}")
            print(f"  Best Params: {study.best_trial.params}")

        if not self.best_config and completed:
            p2 = study.best_trial.params
            rb, rm, rt = (p2.get("target_r_bottom", 4),
                          p2.get("target_r_middle", 6),
                          p2.get("target_r_top", 8))
            self.best_config = {
                "lora_r": p2.get("lora_r", 12), "lora_alpha": p2.get("lora_alpha", 16),
                "learning_rate": p2.get("learning_rate", 3e-5),
                "avg_target_r": int(round((rb + rm + rt) / 3)),
                "r_bottom": rb, "r_middle": rm, "r_top": rt,
                "tinit_ratio": p2.get("tinit_ratio", 0.2),
                "tfinal_ratio": p2.get("tfinal_ratio", 0.75),
                "debug_info": str(p2),
            }

        return self.best_config, self.best_score, self.best_params, self.best_rank_pattern


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="AdaLoRA + Act-LoRA Prior + BO NAS  v3")
    parser.add_argument("--model_name",        type=str,  default="bert-base-uncased")
    parser.add_argument("--task_name",         type=str,  default="sst2")
    parser.add_argument("--use_adalora",       action="store_true", default=False)
    parser.add_argument("--use_lora",          action="store_true", default=False)
    parser.add_argument("--n_trials",          type=int,  default=20)
    parser.add_argument("--n_startup",         type=int,  default=5)
    parser.add_argument("--eval_steps",        type=int,  default=800,
        help="搜索階段每個 trial 最大步數（建議 800~1200）")
    parser.add_argument("--n_probe_samples",   type=int,  default=256,
        help="Act-LoRA 體檢用的樣本數（256 已足夠，過多浪費時間）")
    parser.add_argument("--r_budget_total",    type=int,  default=64,
        help="Act-LoRA 分配 rank 的總預算（控制整體參數量上限）")
    parser.add_argument("--use_gpu",           action="store_true", default=False)
    parser.add_argument("--output_dir",        type=str,  default="./bo_nas_results_v3")
    parser.add_argument("--full_train_epochs", type=int,  default=3)
    parser.add_argument("--seed",              type=int,  default=42)
    parser.add_argument("--local_files_only",  action="store_true", default=False)
    args = parser.parse_args()

    if args.use_lora:
        args.use_adalora = False

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    target_modules = resolve_target_modules(args.model_name)
    num_layers     = get_num_hidden_layers(args.model_name)

    print(f"[Device]  {device}")
    print(f"[Modules] {target_modules}")

    # ── 載入資料集與 tokenizer ────────────────────────────────────────────────
    print("\n📂 Loading dataset...")
    raw_datasets = load_dataset("glue", args.task_name)
    tokenizer    = load_tokenizer_with_fallback(args.model_name)

    s1_key, s2_key = task_to_keys[args.task_name]
    def preprocess(examples):
        tup = ((examples[s1_key],) if s2_key is None
               else (examples[s1_key], examples[s2_key]))
        return tokenizer(*tup, truncation=True, padding="max_length", max_length=128)

    raw_datasets = raw_datasets.map(preprocess, batched=True)
    val_key      = "validation_matched" if args.task_name == "mnli" else "validation"

    train_subset = raw_datasets["train"].select(range(min(30000, len(raw_datasets["train"]))))
    eval_subset  = raw_datasets[val_key].select(range(min(5000,  len(raw_datasets[val_key]))))
    print(f"Train subset: {len(train_subset)}, Eval subset: {len(eval_subset)}")

    # ════════════════════════════════════════════════════════════════════════
    # Phase 0：Act-LoRA 激活範數測量
    # ════════════════════════════════════════════════════════════════════════
    measurer = ActivationNormMeasurer(
        model_name       = args.model_name,
        task_name        = args.task_name,
        tokenizer        = tokenizer,
        probe_dataset    = train_subset,
        device           = device,
        num_layers       = num_layers,
        n_probe_samples  = args.n_probe_samples,
        r_budget_total   = args.r_budget_total,
        local_files_only = args.local_files_only,
    )
    act_prior = measurer.measure()

    # 儲存先驗測量結果
    with open(os.path.join(args.output_dir, "act_prior.json"), "w") as f:
        json.dump(act_prior, f, indent=2)
    print(f"\n💾 Act-LoRA prior saved to {args.output_dir}/act_prior.json")

    # ════════════════════════════════════════════════════════════════════════
    # Phase 1+2：BO 搜索
    # ════════════════════════════════════════════════════════════════════════
    searcher = BONASSearch(
        model_name       = args.model_name,
        task_name        = args.task_name,
        tokenizer        = tokenizer,
        train_dataset    = train_subset,
        eval_dataset     = eval_subset,
        device           = device,
        act_prior        = act_prior,
        use_adalora      = args.use_adalora,
        n_trials         = args.n_trials,
        n_startup        = args.n_startup,
        eval_steps       = args.eval_steps,
        local_files_only = args.local_files_only,
    )
    best_config, best_score, best_search_params, best_rank_pattern = searcher.search()

    # 儲存搜索結果
    with open(os.path.join(args.output_dir, "search_results.json"), "w") as f:
        json.dump({
            "best_config":       best_config,
            "best_score":        best_score,
            "best_params_count": best_search_params,
            "best_rank_pattern": {
                str(k): (v[0] if isinstance(v, list) else v)
                for k, v in best_rank_pattern.items()
            },
            "act_prior":  act_prior,
            "history":    searcher.history,
            "args":       vars(args),
        }, f, indent=2, default=str)
    print(f"\n💾 Search results saved to {args.output_dir}/search_results.json")

    # ════════════════════════════════════════════════════════════════════════
    # Phase 3：正式訓練
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("🚀 Phase 3: Full Training with Best Configuration  [v3]")
    print("=" * 80)

    is_regression = args.task_name == "stsb"
    num_labels    = 1 if is_regression else (3 if args.task_name == "mnli" else 2)

    cfg   = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels,
                                       local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=cfg, local_files_only=args.local_files_only
    )

    train_batch_size = 16
    steps_per_epoch  = len(raw_datasets["train"]) // (train_batch_size * 2)
    total_step       = steps_per_epoch * args.full_train_epochs
    print(f"Full training steps : {total_step} ({args.full_train_epochs} epochs)")
    print(f"Best rank pattern   : {len(best_rank_pattern)} entries")

    if args.use_adalora:
        tinit_full  = max(200, min(int(total_step * best_config["tinit_ratio"]), total_step // 3))
        tfinal_full = max(tinit_full + 100,
                         min(int(total_step * best_config["tfinal_ratio"]), total_step - 100))
        print(f"AdaLoRA tinit={tinit_full}, tfinal={tfinal_full}")
        peft_config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            lora_alpha=best_config["lora_alpha"], lora_dropout=0.1,
            target_modules=target_modules,
            init_r=best_config["lora_r"], target_r=best_config["avg_target_r"],
            total_step=total_step, tinit=tinit_full, tfinal=tfinal_full,
            deltaT=10, orth_reg_weight=0.1,
        )
        callbacks    = [AdaLoraCallback()]
        TrainerClass = AdaLoraTrainer
    else:
        avg_r = best_config.get("avg_target_r", best_config["lora_r"])
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=avg_r,
            lora_alpha=best_config["lora_alpha"], lora_dropout=0.1,
            target_modules=target_modules,
        )
        callbacks    = []
        TrainerClass = Trainer

    model = get_peft_model(model, peft_config)

    final_rank_pattern = build_rank_pattern_from_peft_model(
        model,
        r_bottom=best_config["r_bottom"], r_middle=best_config["r_middle"],
        r_top=best_config["r_top"], num_layers=num_layers,
    )
    if final_rank_pattern:
        try:
            model.peft_config["default"].rank_pattern = final_rank_pattern
            print(f"rank_pattern injected: {len(final_rank_pattern)} entries")
        except Exception as e:
            print(f"rank_pattern injection failed (non-critical): {e}")

    for name, param in model.named_parameters():
        if "classifier" in name or "score" in name:
            param.requires_grad = True

    full_trainable, full_total = print_param_summary(model, "Full Training (before)")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.full_train_epochs,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=64,
        learning_rate=best_config["learning_rate"],
        logging_steps=500,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",
        no_cuda=(device.type == "cpu"),
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        load_best_model_at_end=True,
        metric_for_best_model=("accuracy" if args.task_name not in ["stsb", "cola"] else None),
        weight_decay=0.01,
        dataloader_num_workers=0,
    )

    metric_full = evaluate.load("glue", args.task_name)
    def compute_metrics_full(p: EvalPrediction):
        preds = (p.predictions.flatten() if is_regression
                 else np.argmax(p.predictions, axis=1))
        return metric_full.compute(predictions=preds, references=p.label_ids)

    trainer = TrainerClass(
        model=model, args=training_args,
        train_dataset=raw_datasets["train"], eval_dataset=raw_datasets[val_key],
        tokenizer=tokenizer, data_collator=default_data_collator,
        compute_metrics=compute_metrics_full, callbacks=callbacks,
    )
    trainer.train()
    eval_results = trainer.evaluate()

    post_trainable, post_total = print_param_summary(model, "Full Training (after AdaLoRA pruning)")
    print(f"\n🎯 Final Evaluation: {eval_results}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── 最終報告 ──────────────────────────────────────────────────────────────
    baseline_params = 294912
    baseline_acc    = 0.927
    final_acc = eval_results.get(
        "eval_accuracy",
        eval_results.get("eval_pearson",
        eval_results.get("eval_matthews_correlation", 0))
    )

    print("\n" + "=" * 80)
    print("📊 Final Benchmark Report  [v3 — Act-LoRA Informed BO]")
    print("=" * 80)
    print(f"Best Config         : {best_config['debug_info']}")
    print(f"BO Trials           : {args.n_trials} (startup={args.n_startup})")
    print(f"Act-LoRA Prior      : bot={act_prior['prior_r_bottom']}, "
          f"mid={act_prior['prior_r_middle']}, top={act_prior['prior_r_top']}, "
          f"diversity={act_prior['diversity']:.3f}")
    print()
    print(f"✅ Final Accuracy   : {final_acc:.4f}")
    print(f"ℹ️ Baseline Accuracy: {baseline_acc:.4f}  "
          f"(diff: {(final_acc - baseline_acc)*100:+.2f}%)")
    print()
    print(f"📐 Parameter Summary:")
    print(f"   Trainable (before): {full_trainable:>12,}  "
          f"({100.0*full_trainable/full_total:.4f}%)")
    print(f"   Trainable (after) : {post_trainable:>12,}  (AdaLoRA 動態剪枝後)")
    print(f"   Total             : {post_total:>12,}")
    print(f"   BO Search best    : {best_search_params:>12,}  (搜索階段最佳 trial)")
    print(f"   Baseline          : {baseline_params:>12,}  (Standard AdaLoRA)")
    print()
    param_diff = post_trainable - baseline_params
    if param_diff < 0:
        eff = final_acc / (post_trainable / 1000)
        print(f"🏆 Param Saving : {abs(param_diff):,} "
              f"({abs(param_diff)/baseline_params*100:.2f}%) LESS than baseline")
        print(f"💡 Efficiency   : {eff:.6f} (Acc / 1K trainable params)")
    else:
        print(f"⚠️ Param vs Baseline: +{param_diff:,} "
              f"({param_diff/baseline_params*100:.2f}%) MORE")
    print("=" * 80)


if __name__ == "__main__":
    main()