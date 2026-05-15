#!/usr/bin/env python
"""
Bayesian Optimization (Optuna) Layer-wise Architecture Search  ── v2
主要修改（vs v1）：
  1. learning_rate 搜索範圍縮小至 5e-6 ~ 8e-5（DeBERTa-v3 官方建議區間）
  2. eval_steps 搜索階段建議用 800（預設），讓 surrogate model 見到更收斂的準確度
  3. rank 上限收斂：r_bottom(2~6), r_middle(4~8), r_top(6~12)，避免 init_r 爆大
  4. orth_reg_weight 0.5 → 0.1（正交正則化不能太強，否則短訓練期間壓縮有效 rank）
  5. 正式訓練階段加入詳細參數量印出（trainable / total / 佔比）
  6. 搜索階段每個 trial 完成後也印出參數量摘要
  7. warmup_ratio 搜索階段 0.1 → 0.06（步數少時 warmup 不能太長）
  8. lora_alpha 範圍調整為 8~32，alpha/r 比值更健康
"""

import sys
import os

loralib_path = os.path.join(os.path.dirname(__file__), 'loralib')
if loralib_path in sys.path:
    sys.path.remove(loralib_path)
if 'loralib' in sys.modules:
    del sys.modules['loralib']

print("=" * 80)
print("🔧 Import Configuration")
print("=" * 80)

import logging
import json
import torch
import numpy as np
import evaluate
import gc
from typing import Optional
from datetime import datetime

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
    print(f"✅ Using Optuna: {optuna.__version__}")
except ImportError as e:
    print(f"❌ Failed to import Optuna: {e}")
    sys.exit(1)

try:
    from peft import get_peft_model, LoraConfig, AdaLoraConfig, TaskType
    import peft
    print(f"✅ Using PEFT from: {peft.__file__}")
    print(f"   Version: {peft.__version__}")
    USE_PEFT = True
except ImportError as e:
    print(f"❌ Failed to import PEFT: {e}")
    sys.exit(1)

print("=" * 80 + "\n")

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Task / Model helpers
# ──────────────────────────────────────────────────────────────────────────────
task_to_keys = {
    "cola":  ("sentence", None),
    "mnli":  ("premise", "hypothesis"),
    "mrpc":  ("sentence1", "sentence2"),
    "qnli":  ("question", "sentence"),
    "qqp":   ("question1", "question2"),
    "rte":   ("sentence1", "sentence2"),
    "sst2":  ("sentence", None),
    "stsb":  ("sentence1", "sentence2"),
    "wnli":  ("sentence1", "sentence2"),
}


def resolve_target_modules(model_name: str):
    name = model_name.lower()
    if "deberta-v3" in name or ("deberta" in name and "v3" in name):
        return ["query_proj", "key_proj", "value_proj", "pos_query_proj"]
    if "deberta" in name:
        return ["in_proj", "pos_proj", "pos_q_proj"]
    return ["query", "value"]


def get_num_hidden_layers(model_name: str) -> int:
    name = model_name.lower()
    if "large" in name:
        return 24
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        return cfg.num_hidden_layers
    except Exception:
        return 12


def load_tokenizer_with_fallback(model_name: str):
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print("[Tokenizer] Loaded fast tokenizer")
        return tok
    except Exception as e:
        print(f"[Tokenizer Warning] Fast failed: {e}, falling back to slow...")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("[Tokenizer] Loaded slow tokenizer")
        return tok
    except Exception as slow_err:
        if "deberta" in model_name.lower():
            from transformers import DebertaV2Tokenizer
            tok = DebertaV2Tokenizer.from_pretrained(model_name)
            print("[Tokenizer] Loaded DebertaV2Tokenizer")
            return tok
        raise RuntimeError("Tokenizer loading failed.") from slow_err


# ──────────────────────────────────────────────────────────────────────────────
# 參數量統計工具（新增）
# ──────────────────────────────────────────────────────────────────────────────
def print_param_summary(model, label: str = "Model"):
    """
    印出模型的 trainable / total / 佔比。
    同時回傳 (trainable_params, total_params)。
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Rank Pattern helpers
# ──────────────────────────────────────────────────────────────────────────────
def build_rank_pattern_from_peft_model(
    peft_model, r_bottom: int, r_middle: int, r_top: int, num_layers: int
) -> dict:
    rank_pattern = {}
    boundary_low  = num_layers // 3
    boundary_high = (num_layers * 2) // 3

    for name, module in peft_model.named_modules():
        if hasattr(module, 'lora_E') or hasattr(module, 'lora_A'):
            for layer_idx in range(num_layers):
                for pat in [f"layer.{layer_idx}.", f"layers.{layer_idx}.", f".{layer_idx}.attention"]:
                    if pat in name:
                        if layer_idx < boundary_low:
                            # ✅ PEFT 0.7.1 的 resize_state_dict_by_rank_pattern
                            # 內部執行 sum(rank_idx)，rank_idx 必須是可迭代物件
                            # 若傳入 int 會報 "TypeError: 'int' object is not iterable"
                            rank_pattern[name] = [r_bottom]
                        elif layer_idx < boundary_high:
                            rank_pattern[name] = [r_middle]
                        else:
                            rank_pattern[name] = [r_top]
                        break

    if not rank_pattern:
        print("[rank_pattern] ⚠️ No adapter modules matched; using global target_r.")
    else:
        # counts 用 v[0] 取出實際 rank 值
        counts = {}
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


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────
class PruningException(Exception):
    pass


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial, metric_key: str = "eval_accuracy"):
        self.trial      = trial
        self.metric_key = metric_key
        self._step      = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl,
                    metrics=None, **kwargs):
        if metrics is None:
            return control
        value = metrics.get(self.metric_key, None)
        if value is None:
            return control

        self.trial.report(value, step=self._step)
        self._step += 1

        if self.trial.should_prune():
            print(f"\n[Optuna] Trial {self.trial.number} pruned at step={self._step}, "
                  f"{self.metric_key}={value:.4f}")
            control.should_training_stop = True
            raise PruningException(f"Trial pruned at step {self._step}")

        return control


class AdaLoraCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        target_model = model
        for attr in ["base_model", "model"]:
            if hasattr(target_model, attr):
                candidate = getattr(target_model, attr)
                if hasattr(candidate, "base_model"):
                    target_model = candidate.base_model
                    break
                target_model = candidate
        if hasattr(target_model, "update_and_allocate"):
            try:
                target_model.update_and_allocate(state.global_step)
                if state.global_step % 100 == 0:
                    total_rank = sum(
                        (m.r.get('default', 0) if isinstance(m.r, dict) else m.r)
                        for m in target_model.modules() if hasattr(m, 'r')
                    )
                    print(f"\n[AdaLoRA] Step {state.global_step}: Total Rank = {total_rank}")
            except Exception as e:
                if state.global_step % 200 == 0:
                    print(f"\n[AdaLoRA Warning] rank update failed: {e}")
        return control


class AdaLoraTrainer(Trainer):
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if self.state.global_step > 0:
            target_model = model
            if hasattr(target_model, "module"):
                target_model = target_model.module
            if hasattr(target_model, "base_model"):
                target_model = target_model.base_model
            if hasattr(target_model, "model") and hasattr(target_model.model, "base_model"):
                target_model = target_model.model.base_model
            if hasattr(target_model, "update_and_allocate"):
                try:
                    target_model.update_and_allocate(self.state.global_step)
                except Exception as e:
                    if self.state.global_step % 100 == 0:
                        print(f"\n[AdaLoRA Warning] Update failed: {e}")
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if num_items_in_batch is not None:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                                  num_items_in_batch=num_items_in_batch)
        else:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        try:
            target_model = model
            if hasattr(target_model, "base_model"):
                target_model = target_model.base_model
            if hasattr(target_model, "model") and hasattr(target_model.model, "base_model"):
                target_model = target_model.model.base_model
            if hasattr(target_model, "get_orth_regu_loss"):
                loss = loss + target_model.get_orth_regu_loss()
        except Exception:
            pass
        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────────────────────────────────────
# Bayesian Optimization NAS
# ──────────────────────────────────────────────────────────────────────────────
class BONASSearch:
    """
    修改重點（v2）：
      - learning_rate 改為 5e-6 ~ 8e-5（DeBERTa-v3 實驗驗證區間）
      - rank 上限收斂，避免 init_r 爆大
      - orth_reg_weight 0.5 → 0.1
      - 每個 trial 印出參數量
      - warmup_ratio 搜索階段改為 0.06
    """

    def __init__(
        self,
        model_name:     str,
        task_name:      str,
        tokenizer,
        train_dataset,
        eval_dataset,
        device:         torch.device,
        use_adalora:    bool = True,
        n_trials:       int  = 20,
        n_startup:      int  = 5,
        eval_steps:     int  = 800,    # ✅ v2: 預設改為 800（讓 surrogate 學到更收斂的分數）
        local_files_only: bool = False,
    ):
        self.model_name       = model_name
        self.task_name        = task_name
        self.tokenizer        = tokenizer
        self.train_dataset    = train_dataset
        self.eval_dataset     = eval_dataset
        self.device           = device
        self.use_adalora      = use_adalora
        self.n_trials         = n_trials
        self.n_startup        = n_startup
        self.eval_steps       = eval_steps
        self.local_files_only = local_files_only
        self.target_modules   = resolve_target_modules(model_name)
        self.num_layers       = get_num_hidden_layers(model_name)

        self.metric = evaluate.load("glue", self.task_name)
        self.history: list[dict] = []

        self.best_score:        float = -float('inf')
        self.best_config:       dict  = {}
        self.best_params:       int   = 0
        self.best_rank_pattern: dict  = {}

        print(f"[TargetModules] model={self.model_name}, target_modules={self.target_modules}")
        print(f"[NumLayers]     Detected {self.num_layers} hidden layers")
        print(f"[BO Config]     n_trials={n_trials}, n_startup={n_startup}, "
              f"eval_steps={eval_steps}")

    # ──────────────────────────────────────────────────────────────────────────
    # 搜索空間定義（v2 修改）
    # ──────────────────────────────────────────────────────────────────────────
    def _suggest_params(self, trial: optuna.Trial) -> dict:
        # ✅ v2: rank 上限收斂，避免 init_r 爆大
        r_bottom = trial.suggest_int("target_r_bottom", 2, 6)    # v1: 2~8
        r_middle = trial.suggest_int("target_r_middle", 4, 8)    # v1: 4~10
        r_top    = trial.suggest_int("target_r_top",    6, 12)   # v1: 6~14

        min_lora_r = max(r_bottom, r_middle, r_top) + 2
        lora_r     = trial.suggest_int("lora_r", min_lora_r, max(min_lora_r, 16))  # v1 上限 20

        # ✅ v2: alpha 範圍擴大至 32，讓 alpha/r 比值更靈活
        lora_alpha = trial.suggest_int("lora_alpha", 8, 32)      # v1: 8~24

        # ✅ v2: lr 範圍縮小 — DeBERTa-v3 對高 lr 敏感，3e-4 容易爆掉
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 8e-5, log=True)  # v1: 5e-5~3e-4

        tinit_ratio  = trial.suggest_float("tinit_ratio",  0.10, 0.30)   # v1: 0.15~0.35
        tfinal_ratio = trial.suggest_float("tfinal_ratio", 0.65, 0.85)

        avg_target_r = int(round((r_bottom + r_middle + r_top) / 3))

        return {
            "lora_r":        lora_r,
            "lora_alpha":    lora_alpha,
            "learning_rate": learning_rate,
            "avg_target_r":  avg_target_r,
            "r_bottom":      r_bottom,
            "r_middle":      r_middle,
            "r_top":         r_top,
            "tinit_ratio":   tinit_ratio,
            "tfinal_ratio":  tfinal_ratio,
            "debug_info": (
                f"r={lora_r}, target(bot/mid/top)={r_bottom}/{r_middle}/{r_top}, "
                f"alpha={lora_alpha}"
            ),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 單次 Trial 評估
    # ──────────────────────────────────────────────────────────────────────────
    def _objective(self, trial: optuna.Trial) -> float:
        config = self._suggest_params(trial)
        print(
            f"\n🔬 Trial {trial.number + 1}/{self.n_trials} | "
            f"{config['debug_info']}, lr={config['learning_rate']:.2e}"
        )

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

            tinit_steps  = int(self.eval_steps * config["tinit_ratio"])
            tfinal_steps = int(self.eval_steps * config["tfinal_ratio"])

            if self.use_adalora:
                peft_cfg = AdaLoraConfig(
                    task_type       = TaskType.SEQ_CLS,
                    lora_alpha      = config["lora_alpha"],
                    lora_dropout    = 0.1,
                    target_modules  = self.target_modules,
                    init_r          = config["lora_r"],
                    target_r        = config["avg_target_r"],
                    total_step      = self.eval_steps,
                    tinit           = tinit_steps,
                    tfinal          = tfinal_steps,
                    deltaT          = 10,
                    orth_reg_weight = 0.1,   # ✅ v2: 0.5 → 0.1（正交正則化不能太強）
                )
            else:
                peft_cfg = LoraConfig(
                    task_type      = TaskType.SEQ_CLS,
                    r              = config["lora_r"],
                    lora_alpha     = config["lora_alpha"],
                    lora_dropout   = 0.1,
                    target_modules = self.target_modules,
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

            # ✅ v2: 每個 trial 印出參數量
            trainable_params, total_params = print_param_summary(model, f"Trial {trial.number + 1}")

            model.to(self.device)

            metric_key = {
                "stsb": "eval_pearson",
                "cola": "eval_matthews_correlation",
            }.get(self.task_name, "eval_accuracy")

            training_args = TrainingArguments(
                output_dir                  = "./bo_temp",
                num_train_epochs            = 1,
                max_steps                   = self.eval_steps,
                per_device_train_batch_size = 16,
                gradient_accumulation_steps = 2,
                per_device_eval_batch_size  = 64,     # v1: 32 → 64，加快 eval
                learning_rate               = config["learning_rate"],
                logging_steps               = 100,
                save_strategy               = "no",
                evaluation_strategy         = "steps",
                eval_steps                  = max(100, self.eval_steps // 4),
                report_to                   = "none",
                no_cuda                     = (self.device.type == "cpu"),
                disable_tqdm                = True,
                lr_scheduler_type           = "cosine",
                warmup_ratio                = 0.06,   # ✅ v2: 0.1 → 0.06（步數少時 warmup 不能太長）
                dataloader_num_workers      = 0,
                weight_decay                = 0.01,   # ✅ v2: 新增正則化（原版無）
            )

            def compute_metrics(p: EvalPrediction):
                preds = np.squeeze(p.predictions) if is_regression else np.argmax(p.predictions, axis=1)
                return self.metric.compute(predictions=preds, references=p.label_ids)

            callbacks = [OptunaPruningCallback(trial, metric_key=metric_key)]
            if self.use_adalora:
                callbacks.append(AdaLoraCallback())

            TrainerClass = AdaLoraTrainer if self.use_adalora else Trainer
            trainer = TrainerClass(
                model           = model,
                args            = training_args,
                train_dataset   = self.train_dataset,
                eval_dataset    = self.eval_dataset,
                tokenizer       = self.tokenizer,
                data_collator   = default_data_collator,
                compute_metrics = compute_metrics,
                callbacks       = callbacks,
            )

            trainer.train()

            try:
                eval_results = trainer.evaluate()
                accuracy = eval_results.get(
                    "eval_pearson" if self.task_name == "stsb"
                    else "eval_matthews_correlation" if self.task_name == "cola"
                    else "eval_accuracy",
                    0.0
                )
            except Exception:
                accuracy = 0.0

            num_params = trainable_params   # 已在上方計算

            penalty = (num_params / 100_000) * 0.01
            fitness = accuracy - penalty

            print(
                f"   ✅ Score={accuracy:.4f}, TrainableParams={num_params:,}, "
                f"Penalty={penalty:.4f}, Fitness={fitness:.4f}"
            )

            self.history.append({
                "trial":            trial.number + 1,
                "fitness":          fitness,
                "accuracy":         accuracy,
                "trainable_params": num_params,
                "total_params":     total_params,
                "config":           {k: v for k, v in config.items() if k != "debug_info"},
                "rank_pattern":     {str(k): (v[0] if isinstance(v, list) else v) for k, v in rank_pattern.items()},
            })

            if fitness > self.best_score:
                self.best_score        = fitness
                self.best_config       = config
                self.best_params       = num_params
                self.best_rank_pattern = rank_pattern
                print(f"   🏆 New best! Fitness={fitness:.4f}")

            del model, trainer
            torch.cuda.empty_cache()
            gc.collect()

            return fitness

        except PruningException:
            raise optuna.exceptions.TrialPruned()
        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            return -1.0

    # ──────────────────────────────────────────────────────────────────────────
    # 主搜索迴圈
    # ──────────────────────────────────────────────────────────────────────────
    def search(self):
        print("\n" + "=" * 80)
        print("🤖 Bayesian Optimization (Optuna TPE) Layer-wise NAS  [v2]")
        print("=" * 80)
        print(f"n_trials={self.n_trials}, n_startup={self.n_startup}, "
              f"eval_steps={self.eval_steps}")
        print(f"Target Modules: {self.target_modules}")
        print(f"Num Layers: {self.num_layers}")
        print(f"Sampler: TPESampler (multivariate), Pruner: MedianPruner")
        print("=" * 80)

        sampler = TPESampler(
            n_startup_trials = self.n_startup,
            multivariate     = True,
            seed             = 42,
        )
        pruner = MedianPruner(
            n_startup_trials = max(3, self.n_startup),
            n_warmup_steps   = 2,   # ✅ v2: 1 → 2，避免太早剪枝（eval_steps 變大後更需要）
        )

        study = optuna.create_study(
            direction = "maximize",
            sampler   = sampler,
            pruner    = pruner,
        )

        study.optimize(
            self._objective,
            n_trials          = self.n_trials,
            show_progress_bar = False,
        )

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        print("\n" + "=" * 80)
        print("📊 BO Search Summary")
        print("=" * 80)
        print(f"  Completed : {len(completed)}")
        print(f"  Pruned    : {len(pruned)}")
        print(f"  Failed    : {len(failed)}")
        if study.best_value is not None:
            print(f"  Best Fitness (Optuna): {study.best_value:.4f}")
            print(f"  Best Trial : #{study.best_trial.number + 1}")
            print(f"  Best Params: {study.best_trial.params}")

        if not self.best_config and len(completed) > 0:
            p = study.best_trial.params
            r_b = p.get("target_r_bottom", 4)
            r_m = p.get("target_r_middle", 6)
            r_t = p.get("target_r_top", 8)
            self.best_config = {
                "lora_r":        p.get("lora_r", 12),
                "lora_alpha":    p.get("lora_alpha", 16),
                "learning_rate": p.get("learning_rate", 3e-5),
                "avg_target_r":  int(round((r_b + r_m + r_t) / 3)),
                "r_bottom": r_b, "r_middle": r_m, "r_top": r_t,
                "tinit_ratio":  p.get("tinit_ratio", 0.2),
                "tfinal_ratio": p.get("tfinal_ratio", 0.75),
                "debug_info":   str(p),
            }

        return (
            self.best_config,
            self.best_score,
            self.best_params,
            self.best_rank_pattern,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="AdaLoRA + Bayesian Optimization NAS  v2")
    parser.add_argument("--model_name",         type=str,  default="bert-base-uncased")
    parser.add_argument("--task_name",          type=str,  default="sst2")
    parser.add_argument("--use_adalora",        action="store_true", default=False)
    parser.add_argument("--use_lora",           action="store_true", default=False)
    parser.add_argument(
        "--n_trials", type=int, default=20,
        help="Optuna 總 trial 數（建議 15~40）"
    )
    parser.add_argument(
        "--n_startup", type=int, default=5,
        help="TPE warm-up 隨機 trial 數（建議 n_trials 的 20~30%%）"
    )
    # ✅ v2: 預設 eval_steps 改為 800
    parser.add_argument(
        "--eval_steps", type=int, default=800,
        help="搜索階段每個 trial 的最大訓練步數（建議 800~1200，太少會讓 surrogate 學到未收斂的分數）"
    )
    parser.add_argument("--use_gpu",            action="store_true", default=False)
    parser.add_argument("--output_dir",         type=str,  default="./bo_nas_results")
    parser.add_argument("--full_train_epochs",  type=int,  default=3)
    parser.add_argument("--seed",               type=int,  default=42)
    parser.add_argument("--local_files_only",   action="store_true", default=False)
    args = parser.parse_args()

    if args.use_lora:
        args.use_adalora = False

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    target_modules = resolve_target_modules(args.model_name)
    num_layers     = get_num_hidden_layers(args.model_name)

    print(f"[TargetModules] model={args.model_name}, target_modules={target_modules}")
    print(f"[Device]        Using: {device}")

    print("\n📂 Loading dataset...")
    raw_datasets = load_dataset("glue", args.task_name)
    tokenizer    = load_tokenizer_with_fallback(args.model_name)

    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    def preprocess(examples):
        args_tuple = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        return tokenizer(*args_tuple, truncation=True, padding="max_length", max_length=128)

    raw_datasets = raw_datasets.map(preprocess, batched=True)
    val_key = "validation_matched" if args.task_name == "mnli" else "validation"

    # ✅ v2: train subset 稍微增大，讓每個 trial 見到更多樣本
    train_subset = raw_datasets["train"].select(range(min(30000, len(raw_datasets["train"]))))
    eval_subset  = raw_datasets[val_key].select(range(min(5000, len(raw_datasets[val_key]))))
    print(f"Train subset: {len(train_subset)}, Eval subset: {len(eval_subset)}")

    # ── BO NAS 搜索 ───────────────────────────────────────────────────────────
    searcher = BONASSearch(
        model_name       = args.model_name,
        task_name        = args.task_name,
        tokenizer        = tokenizer,
        train_dataset    = train_subset,
        eval_dataset     = eval_subset,
        device           = device,
        use_adalora      = args.use_adalora,
        n_trials         = args.n_trials,
        n_startup        = args.n_startup,
        eval_steps       = args.eval_steps,
        local_files_only = args.local_files_only,
    )

    best_config, best_score, best_search_params, best_rank_pattern = searcher.search()

    serializable_rank_pattern = {str(k): (v[0] if isinstance(v, list) else v) for k, v in best_rank_pattern.items()}
    with open(os.path.join(args.output_dir, "search_results.json"), "w") as f:
        json.dump(
            {
                "best_config":       best_config,
                "best_score":        best_score,
                "best_params_count": best_search_params,
                "best_rank_pattern": serializable_rank_pattern,
                "history":           searcher.history,
                "args":              vars(args),
            },
            f, indent=2, default=str,
        )
    print(f"\n💾 Search results saved to {args.output_dir}/search_results.json")

    # ── 正式訓練 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🚀 Full Training with Best Configuration  [v2]")
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
    print(f"Full training steps: {total_step} ({args.full_train_epochs} epochs)")
    print(f"Best rank pattern entries: {len(best_rank_pattern)}")

    if args.use_adalora:
        tinit_full  = max(200, min(int(total_step * best_config["tinit_ratio"]),  total_step // 3))
        tfinal_full = max(tinit_full + 100,
                          min(int(total_step * best_config["tfinal_ratio"]), total_step - 100))
        print(f"[Full Train] AdaLoRA: tinit={tinit_full}, tfinal={tfinal_full}")
        peft_config = AdaLoraConfig(
            task_type       = TaskType.SEQ_CLS,
            lora_alpha      = best_config["lora_alpha"],
            lora_dropout    = 0.1,
            target_modules  = target_modules,
            init_r          = best_config["lora_r"],
            target_r        = best_config["avg_target_r"],
            total_step      = total_step,
            tinit           = tinit_full,
            tfinal          = tfinal_full,
            deltaT          = 10,
            orth_reg_weight = 0.1,   # ✅ v2: 同搜索階段保持一致
        )
        callbacks    = [AdaLoraCallback()]
        TrainerClass = AdaLoraTrainer
    else:
        avg_r = best_config.get("avg_target_r", best_config["lora_r"])
        print(f"[Full Train] LoRA (fixed), avg_r={avg_r}")
        peft_config = LoraConfig(
            task_type      = TaskType.SEQ_CLS,
            r              = avg_r,
            lora_alpha     = best_config["lora_alpha"],
            lora_dropout   = 0.1,
            target_modules = target_modules,
        )
        callbacks    = []
        TrainerClass = Trainer

    model = get_peft_model(model, peft_config)

    final_rank_pattern = build_rank_pattern_from_peft_model(
        model,
        r_bottom  = best_config["r_bottom"],
        r_middle  = best_config["r_middle"],
        r_top     = best_config["r_top"],
        num_layers= num_layers,
    )
    if final_rank_pattern:
        try:
            model.peft_config["default"].rank_pattern = final_rank_pattern
            print(f"[Full Train] rank_pattern injected: {len(final_rank_pattern)} entries")
        except Exception as e:
            print(f"[Full Train] rank_pattern injection failed (non-critical): {e}")

    for name, param in model.named_parameters():
        if "classifier" in name or "score" in name:
            param.requires_grad = True

    # ✅ v2: 正式訓練前印詳細參數量
    full_trainable, full_total = print_param_summary(model, "Full Training")

    model.to(device)

    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.full_train_epochs,
        per_device_train_batch_size = train_batch_size,
        gradient_accumulation_steps = 2,
        per_device_eval_batch_size  = 64,
        learning_rate               = best_config["learning_rate"],
        logging_steps               = 500,
        save_strategy               = "epoch",
        evaluation_strategy         = "epoch",
        report_to                   = "none",
        no_cuda                     = (device.type == "cpu"),
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.06,   # ✅ v2: 與搜索階段一致
        load_best_model_at_end      = True,
        metric_for_best_model       = "accuracy" if args.task_name not in ["stsb", "cola"] else None,
        weight_decay                = 0.01,   # ✅ v2: 新增
        dataloader_num_workers      = 0,
    )

    metric_full = evaluate.load("glue", args.task_name)

    def compute_metrics_full(p: EvalPrediction):
        preds = p.predictions.flatten() if is_regression else np.argmax(p.predictions, axis=1)
        return metric_full.compute(predictions=preds, references=p.label_ids)

    trainer = TrainerClass(
        model           = model,
        args            = training_args,
        train_dataset   = raw_datasets["train"],
        eval_dataset    = raw_datasets[val_key],
        tokenizer       = tokenizer,
        data_collator   = default_data_collator,
        compute_metrics = compute_metrics_full,
        callbacks       = callbacks,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    # ✅ v2: 訓練完成後再次印出（AdaLoRA 動態剪枝後參數量可能改變）
    post_trainable, post_total = print_param_summary(model, "After Full Training")

    print(f"\n🎯 Final Evaluation Results: {eval_results}")

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
    print("📊 Final Benchmark Report  [v2]")
    print("=" * 80)
    print(f"Best Configuration  : {best_config['debug_info']}")
    print(f"BO Trials Run       : {args.n_trials} (startup={args.n_startup})")
    print(f"✅ Final Accuracy   : {final_acc:.4f}")
    print(f"ℹ️ Baseline Accuracy: {baseline_acc:.4f}  (diff: {(final_acc - baseline_acc)*100:+.2f}%)")
    print()
    print(f"📐 Parameter Summary (Full Training):")
    print(f"   Trainable (before): {full_trainable:>12,}  ({100.0*full_trainable/full_total:.4f}%)")
    print(f"   Trainable (after) : {post_trainable:>12,}  (AdaLoRA 動態剪枝後)")
    print(f"   Total             : {post_total:>12,}")
    print(f"   BO Search best    : {best_search_params:>12,}  (搜索階段最佳 trial)")
    print(f"   Baseline          : {baseline_params:>12,}  (Standard AdaLoRA)")
    print()

    param_diff = post_trainable - baseline_params
    if param_diff < 0:
        efficiency = final_acc / (post_trainable / 1000)
        print(f"🏆 Parameter Saving : {abs(param_diff):,} ({abs(param_diff)/baseline_params*100:.2f}%) LESS than baseline")
        print(f"💡 Efficiency Score : {efficiency:.6f} (Acc per 1K trainable params)")
    else:
        print(f"⚠️ Parameter vs Baseline: +{param_diff:,} ({param_diff/baseline_params*100:.2f}%) MORE")
    print("=" * 80)


if __name__ == "__main__":
    main()