# Act-LoRA Informed Bayesian Optimization for Layer-wise NAS (v3)

這是一個結合了 **Act-LoRA (激活範數引導)** 與 **Bayesian Optimization (貝葉斯優化)** 的神經架構搜索 (NAS) 工具，專為大語言模型（如 DeBERTa、BERT）的低秩適配 (LoRA/AdaLoRA) 設計。

## 🌟 核心特性

本專案 v3 版本引入了 **"Informed Prior" (具備先驗知識的搜索)** 機制，顯著提升了 NAS 在處理 PEFT (Parameter-Efficient Fine-Tuning) 時的效率：

1.  **Phase 0: Act-LoRA 體檢階段**
    在正式搜索前，對 Frozen Base Model 進行少量樣本推論，利用 `Forward Hook` 捕捉每一層 Attention 輸出的 **L2 Activation Norms**。範數越高代表該層對任務越關鍵。
2.  **Phase 1: 智能熱啟動 (Warm Start)**
    將測量到的激活範數轉換為建議的 Rank 分佈，並透過 `study.enqueue_trial()` 注入 Optuna，讓貝葉斯優化 (TPE) 從「高品質起點」開始，而非純隨機碰撞。
3.  **Phase 2: 層級差異化搜索 (Layer-wise Search)**
    根據 Phase 0 的結果將模型分為高、中、低重要性區域，並自動調整每個區域的搜索半徑。這讓模型能自動學習到哪些層需要更多的 Rank 預算。
4.  **Phase 3: AdaLoRA 動態剪枝整合**
    結合 AdaLoRA 在訓練過程中的奇異值剪枝，實現從「宏觀層級分配」到「微觀權重剪枝」的全自動優化。

## 📊 運算邏輯示意



1. **預測量 (Pre-measurement)**: 獲取各層的激活值，計算其 L2 範數作為顯著性指標。
2. **優先級分配**: 根據顯著性將層劃分為 Bottom, Middle, Top 三類重要度等級。
3. **BO 優化**: 使用 Optuna (TPE) 尋找最優的 $r$、$\alpha$ 與 $learning\_rate$。
4. **適應度函數**: $Fitness = Accuracy - \lambda \cdot \text{Trainable Params}$。

## 🚀 快速開始

### 環境需求
- Python 3.8+
- PyTorch, Transformers, PEFT, Optuna, Evaluate

### 執行搜索與訓練
```bash
python run_bo_nas_search_v3.py \
    --model_name "microsoft/deberta-v3-base" \
    --task_name "sst2" \
    --use_adalora \
    --n_trials 20 \
    --eval_steps 800 \
    --use_gpu

## 📚 支援任務與資料集

本專案支援 GLUE Benchmark (NLU) 以及常見的生成式任務 (NLG)：

### NLU (自然語言理解) - GLUE Benchmark
- **CoLA**: 語言合法性判定
- **SST-2**: 情感分析
- **MRPC/QQP**: 語義相似度
- **MNLI/QNLI/RTE**: 自然語言推理 (NLI)
- **STS-B**: 語義相關性評分 (回歸任務)

### NLG (自然語言生成) & QA
- **SQuAD v2.0**: 問答任務
- **XSum / CNN/DailyMail**: 摘要生成 (需配合 BART/T5 等模型)

## ⚙️ 關鍵超參數說明

在執行 `run_bo_nas_search_v3.py` 時，主要調整以下參數：

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--model_name` | `bert-base-uncased` | 預訓練模型名稱（支援 DeBERTaV3, RoBERTa 等） |
| `--eval_steps` | `800` | 搜索階段每個 Trial 訓練的步數 |
| `--n_probe_samples`| `256` | Act-LoRA 體檢（激活範數測量）所用的樣本數 |
| `--r_budget_total` | `64` | Act-LoRA 初始分配的總 Rank 預算 |
| `--n_trials` | `20` | 貝葉斯優化總共嘗試的次數 |