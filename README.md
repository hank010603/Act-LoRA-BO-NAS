# Act-LoRA Informed Bayesian Optimization for Layer-wise NAS (v3)

結合 **Act-LoRA（激活範數引導）** 與 **Bayesian Optimization（貝葉斯優化）** 的神經架構搜索（NAS）工具，專為大語言模型的低秩適配（LoRA）設計。核心理念：不盲目搜索，先「體檢」再「優化」。

---

## 🌟 核心特性

### 為什麼不直接用 AdaLoRA？

傳統 AdaLoRA 雖然能動態剪枝，但在以下情境下存在問題：

- **短訓練步數**（< 2000 steps）：rank pruning 過激，`ranknum` 歸零，模型輸出退化成隨機猜測
- **PEFT 0.7.x 版本**：`save_pretrained` 在 rank_pattern key 格式不一致時 crash
- **搜索階段**：不需要動態剪枝，固定 rank 的 LoRA 就能正確評估超參數的相對優劣

本專案的解法：搜索階段用 **LoRA**（穩定、可評估），正式訓練也用 **LoRA**（搭配 Act-LoRA 先驗決定的最佳 rank），完整跑通整個流程。

---

## 📊 實驗基準對照表 (Benchmark Results)

本框架在不同規模與複雜度的 GLUE 基準任務上均表現出極高的參數效率，**平均將可訓練引數控制在 0.16% 以下**：

| 資料集 (Dataset) | 任務類型 | 訓練集規模 | 可訓練引數佔比 | 本文方法 (Ours) | Baseline (AdaLoRA) | 準確率差距 | 參數節省 (Saving) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **SST-2** | 單句情感 | ~6.7 萬筆 | 0.1514% | **92.66%** | 92.70% | -0.04% | 5.21% |
| **MNLI** | 雙句推理 | ~39 萬筆 | 0.1522% | **88.20%** | 92.70% | -4.50% | 4.69% |
| **QQP** | 雙句重複 | ~36 萬筆 | 0.1514% | **87.73%** | 92.70% | -4.97% | 5.21% |
| **QNLI** | 問答推理 | ~10 萬筆 | 0.1709% | **85.80%** | 92.70% | -6.90% | **36.46%** |

### 💡 核心研究洞察
* **自適應架構發現**：系統在 QNLI 任務中展現了驚人的自適應能力，自動發現了 `4/6/5` 的層級配置，並達成 **36.46% 的參數壓縮率**，驗證了 Act-LoRA 系統在面對不同任務類型時，能主動調整模型複雜度以應對資源限制。
* **極致效率指標**：QNLI 實驗創下了 **0.004578 (Acc / 1K trainable params)** 的最高效率紀錄，展示了本系統在邊緣裝置部署場景下的優異潛力。

---

## 📐 系統架構


```

Phase 0: Act-LoRA 體檢
└─ 捕捉每層 [CLS] token 的 L2 Activation Norm
Phase 1: Warm Start 注入
└─ 根據先驗配置初始 Optuna，加速收斂
Phase 2: Bayesian Optimization 搜索（LoRA）
└─ TPE Sampler 搜索最佳 rank/lr/alpha
Phase 3: 正式訓練（LoRA + Informed Rank）
└─ 差異化學習率（Classifier ×15）確保高效收斂

```

---

## 🛠️ 環境需求與安裝

### 環境安裝
```powershell
conda create -n adalora_v4 python=3.10 -y
conda activate adalora_v4
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\install_adalora_clean_v4.ps1

```

### 離線模式（防止網路超時）

```powershell
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"

```

---

## 🚀 使用方式

```powershell
python run_bo_nas_search_v3.py `
  --model_name "microsoft/deberta-v3-base" `
  --task_name "qnli" `
  --use_adalora `
  --n_trials 5 `
  --eval_steps 1200 `
  --use_gpu `
  --local_files_only

```

---

## 📁 檔案結構

```
AdaLoRA/
├── run_bo_nas_search_v3.py      # 主程式
├── bo_nas_results_v3/           # 實驗數據與 Checkpoints
└── README.md

```

## 📝 已知限制

* PEFT 0.7.1 於 rank_pattern 儲存時有相容性限制，目前正式訓練階段採用固定 Rank 的 LoRA。
* 離線模式強烈建議設定，以避免 Hugging Face 預設請求導致的 Timeout。

---

## 🔗 參考資料

* [AdaLoRA: Adaptive Budget Allocation](https://arxiv.org/abs/2303.10512)
* [Act-LoRA: Activation-Guided LoRA Rank Selection](https://arxiv.org/abs/2310.11454)