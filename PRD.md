PRD: 潛伏攻擊對比實驗 MVP
文檔版本: v1.0
目標受眾: Coding Agent
預計開發時間: 4-6 小時

一、項目目標
1.1 核心目的
創建一個最小可行性實驗，通過實證數據證明：

BlockDFL 的弱點：當驗證委員會被共謀控制，系統性選擇最差的 global update 時，模型訓練會停滯甚至退化
我們方法的優勢：通過 audit 機制總是選擇最優 global update，即使在相同攻擊場景下也能正常收斂

1.2 使用場景
此實驗結果將用於學術論文的實驗章節，需要生成可直接引用的對比圖表和定量數據。
1.3 非目標（明確不做的事）

❌ 不實現區塊鏈相關功能（共識、區塊、鏈）
❌ 不實現完整的聯邦學習系統（客戶端-服務器架構）
❌ 不實現網絡通信
❌ 不實現 Stake 機制、角色分配
❌ 不需要考慮實際部署、性能優化
❌ 不需要支持分佈式運行


二、概念模型
2.1 真實 BlockDFL 的流程（僅供理解，不需實現）
真實系統中：
1. 50個參與者被分配為：Update Providers, Aggregators, Verifiers
2. Aggregators 從多個 local updates 中聚合出 global update
3. 多個 Aggregators 產生多個候選 global updates
4. Verifiers 投票選擇一個 global update
5. 選中的 global update 被所有人應用

攻擊場景：
- 當 >2/3 Verifiers 被共謀控制
- 他們會投票選擇質量最差的 global update
- 導致模型訓練受到破壞
2.2 我們要模擬的簡化版本
簡化模型：
1. 維護兩個獨立訓練的模型：
   - Model_BlockDFL（模擬被攻擊的系統）
   - Model_Ours（模擬有 audit 保護的系統）

2. 每個訓練步驟：
   a) 生成 4 個候選 updates
      - 每個 update 基於不同的數據子集訓練
      - 模擬 4 個 aggregators 各自產生的 global update
   
   b) 評估每個 update 的質量
      - 使用測試集準確度作為質量指標
   
   c) 選擇策略：
      - Epoch 1-100: 兩個模型都選最好的 update（正常訓練）
      - Epoch 101+:  
        * BlockDFL: 選最差的 update（模擬被攻擊）
        * Ours: 選最好的 update（audit 機制檢測並糾正）
   
   d) 各自應用選中的 update

3. 記錄每個 epoch 後兩個模型的測試準確度
2.3 為什麼這個簡化是有效的

保留了核心機制：多個候選中選擇一個的決策過程
體現了關鍵差異：選擇最差 vs 選擇最優的對比
避免了無關複雜度：區塊鏈、網絡通信等不影響此對比


三、數據規格
3.1 數據集要求
Dataset 1: MNIST

來源: torchvision.datasets.MNIST
訓練集: 60,000 樣本
測試集: 10,000 樣本
類別數: 10（數字 0-9）
預處理: 標準化（mean=0.1307, std=0.3081）

Dataset 2: CIFAR-10

來源: torchvision.datasets.CIFAR10
訓練集: 50,000 樣本
測試集: 10,000 樣本
類別數: 10（飛機、汽車等）
預處理: 標準化（RGB 各通道）

3.2 數據分割規格
目的: 將訓練集分成 4 個子集，模擬 4 個 aggregators 各自擁有的數據
分割方法: Non-IID Dirichlet 分佈

參數: α = 0.5（控制不均勻程度）
原理: 每個子集包含所有類別，但比例不同
效果: 每個子集訓練出的 update 質量會有差異

具體要求:
輸入: 完整訓練集 D = {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}
輸出: 4個子集 D₁, D₂, D₃, D₄

約束條件:
1. D₁ ∪ D₂ ∪ D₃ ∪ D₄ = D（無遺漏）
2. Dᵢ ∩ Dⱼ = ∅, ∀i≠j（無重疊）
3. 每個 Dᵢ 的類別分佈不同（non-IID）
4. 使用 Dirichlet(α=0.5) 控制分佈
參考實現邏輯（不要求實際編碼，只是說明算法）:
For each class c in {0,1,...,9}:
    1. 獲取該類別的所有樣本索引 Ic
    2. 從 Dirichlet(α, α, α, α) 採樣得到分配比例 [p1, p2, p3, p4]
    3. 按比例將 Ic 分配給 4 個子集
3.3 測試集使用規格

測試集不分割: 完整的測試集用於評估所有模型
使用時機:

每個 epoch 結束後評估當前模型準確度
評估每個候選 update 的質量（應用 update 後計算準確度）




四、模型規格
4.1 MNIST 模型架構
必須使用與 BlockDFL 論文完全相同的架構（確保可比性）:
層結構:
1. Conv2D: 1 → 32 channels, kernel=3×3, stride=1
2. ReLU
3. Conv2D: 32 → 64 channels, kernel=3×3, stride=1
4. ReLU
5. MaxPool2D: kernel=2×2
6. Dropout: p=0.25
7. Flatten
8. Linear: 9216 → 128
9. ReLU
10. Dropout: p=0.5
11. Linear: 128 → 10
12. LogSoftmax

總參數量: 1,662,752（與論文一致）
4.2 CIFAR-10 模型架構
必須使用與 BlockDFL 論文相同的 CIFARNET:
架構描述:
64C3×3 - 64C3×3 - MaxPool2 - Dropout(0.1) 
- 128C3×3 - 128C3×3 - AvgPool2 
- 256C3×3 - 256C3×3 - AvgPool8 
- Dropout(0.5) - FC256 - FC10

其中 "64C3×3" 表示 64 個 3×3 卷積核

總參數量: 1,149,770（與論文一致）
4.3 模型初始化規格

兩個模型必須從相同的初始權重開始
使用 PyTorch 默認初始化即可
需要在訓練開始前設置相同的 random seed


五、訓練流程規格
5.1 超參數設定
yamlMNIST:
  total_epochs: 300  # 統一為 300 輪 (符合 README)
  attack_start_epoch: 100  # 第 100 epoch 開始攻擊
  local_training_epochs: 1  # 每個 update 的訓練輪數
  batch_size: 32
  learning_rate: 0.01
  optimizer: SGD
  lr_decay: 0.99 per epoch

CIFAR-10:
  total_epochs: 300
  attack_start_epoch: 100  # 統一為 100 開始攻擊
  local_training_epochs: 1
  batch_size: 32
  learning_rate: 0.01
  optimizer: SGD
  lr_decay: 0.99 per epoch
```

### 5.2 單個 Epoch 的詳細流程
```
For epoch in [1, 2, ..., total_epochs]:
    
    Step 1: 生成 4 個候選 updates
    --------------------------------
    For aggregator_id in [0, 1, 2, 3]:
        a) 創建當前模型的副本 temp_model
        b) 在數據子集 D[aggregator_id] 上訓練 temp_model
           - 訓練 1 個完整的 epoch
           - 使用 batch_size=32
           - 使用 SGD 優化器，lr = 0.01 × 0.99^epoch
        c) 計算 update = temp_model.parameters - current_model.parameters
        d) 評估 update 質量 = accuracy(current_model + update, test_set)
        e) 保存 (update, quality_score)
    
    結果: updates = [(Δw₁, s₁), (Δw₂, s₂), (Δw₃, s₃), (Δw₄, s₄)]
    
    
    Step 2: 選擇 update（關鍵差異）
    --------------------------------
    If epoch < attack_start_epoch:
        # 攻擊前：兩者都選最好的（正常訓練）
        best_idx = argmax([s₁, s₂, s₃, s₄])
        blockdfl_chosen_update = updates[best_idx]
        ours_chosen_update = updates[best_idx]
    Else:
        # 攻擊後：BlockDFL 選最差，我們的選最好
        worst_idx = argmin([s₁, s₂, s₃, s₄])
        best_idx = argmax([s₁, s₂, s₃, s₄])
        blockdfl_chosen_update = updates[worst_idx]
        ours_chosen_update = updates[best_idx]
    
    
    Step 3: 應用 updates
    -------------------
    blockdfl_model.parameters += blockdfl_chosen_update
    ours_model.parameters += ours_chosen_update
    
    
    Step 4: 記錄當前性能
    -------------------
    blockdfl_accuracy = evaluate(blockdfl_model, test_set)
    ours_accuracy = evaluate(ours_model, test_set)
    
    log(epoch, blockdfl_accuracy, ours_accuracy)
```

### 5.3 Update 質量評估方法
```
定義: 一個 update 的質量 = 應用該 update 後模型的測試準確度

計算步驟:
1. 創建當前模型的副本 eval_model
2. 應用 update: eval_model.parameters += update
3. 在完整測試集上評估: accuracy = correct / total
4. 返回 accuracy 作為質量分數

注意: 
- 評估時使用完整測試集（10,000 樣本）
- 不使用訓練集評估（避免過擬合偏差）
- 評估是臨時的，不影響主模型

六、輸出規格
6.1 數據輸出
主要輸出文件: JSON 格式
json// mnist_results.json
{
  "dataset": "MNIST",
  "total_epochs": 200,
  "attack_start_epoch": 100,
  "results": [
    {
      "epoch": 1,
      "blockdfl_accuracy": 0.1823,
      "ours_accuracy": 0.1823,
      "selected_update_quality": {
        "blockdfl": 0.1823,
        "ours": 0.1823
      },
      "all_update_qualities": [0.1823, 0.1654, 0.1897, 0.1723]
    },
    {
      "epoch": 2,
      "blockdfl_accuracy": 0.4521,
      "ours_accuracy": 0.4521,
      ...
    },
    ...
    {
      "epoch": 100,
      "blockdfl_accuracy": 0.9812,
      "ours_accuracy": 0.9812,
      ...
    },
    {
      "epoch": 101,
      "blockdfl_accuracy": 0.9798,  // 開始下降
      "ours_accuracy": 0.9824,       // 繼續提升
      ...
    },
    ...
  ],
  "final_results": {
    "blockdfl_final_accuracy": 0.8521,
    "ours_final_accuracy": 0.9923,
    "gap": 0.1402
  }
}
```

### 6.2 可視化輸出

**圖表 1: 訓練曲線對比圖** (`mnist_convergence.png`, `cifar10_convergence.png`)
```
規格:
- 尺寸: 10 inch × 6 inch
- DPI: 300（用於論文印刷）
- 格式: PNG（也可輸出 PDF）

內容:
- X軸: Epoch (1 to 200/300)
- Y軸: Test Accuracy (0 to 1.0)
- 曲線1: BlockDFL（紅色虛線）
- 曲線2: Ours（藍色實線）
- 垂直虛線標記攻擊開始點（灰色）
- 標註文字: "Attack Starts" 在垂直線旁

視覺效果要求:
- 清晰展示攻擊後的分叉（BlockDFL 下降，Ours 持續上升）
- 圖例放在右下角
- 網格線（淺灰色，alpha=0.3）
- 標題: "{Dataset}: Convergence Under Latent Collusion Attack"
```

**圖表 2: 關鍵指標對比** (`comparison_summary.png`)
```
橫向柱狀圖，對比:
- 攻擊前最終準確度（epoch 100/150）
- 攻擊後最終準確度（epoch 200/300）
- 準確度下降幅度

每個指標兩根柱子（BlockDFL vs Ours）
顏色編碼一致（紅 vs 藍）
```

### 6.3 終端輸出規格

訓練過程中的實時輸出：
```
==================================================
Running MNIST Experiment
==================================================

Epoch 1/200
  Generating 4 candidate updates... Done
  Update qualities: [0.1823, 0.1654, 0.1897, 0.1723]
  BlockDFL selected: Update 2 (best, score=0.1897)
  Ours selected:     Update 2 (best, score=0.1897)
  → BlockDFL accuracy: 18.23%
  → Ours accuracy:     18.23%

Epoch 2/200
  ...

...

Epoch 100/200
  Generating 4 candidate updates... Done
  Update qualities: [0.9812, 0.9765, 0.9834, 0.9801]
  BlockDFL selected: Update 2 (best, score=0.9834)
  Ours selected:     Update 2 (best, score=0.9834)
  → BlockDFL accuracy: 98.12%
  → Ours accuracy:     98.12%

⚠️  ATTACK STARTS (Epoch 101)

Epoch 101/200
  Generating 4 candidate updates... Done
  Update qualities: [0.9823, 0.9654, 0.9841, 0.9712]
  BlockDFL selected: Update 1 (WORST, score=0.9654) ⚠️
  Ours selected:     Update 2 (best, score=0.9841)
  → BlockDFL accuracy: 97.98% ⬇️
  → Ours accuracy:     98.24% ⬆️

...

Epoch 200/200
  ...
  → BlockDFL accuracy: 85.21% 
  → Ours accuracy:     99.23%

==================================================
Experiment Complete
==================================================

Final Test Accuracy:
  BlockDFL:  85.21%
  Ours:      99.23%
  Gap:       +14.02 pp

Results saved to: ./results/mnist_results.json
Figure saved to:  ./results/mnist_convergence.png
```

---

## 七、技術約束與要求

### 7.1 必須使用的技術

- **語言**: Python 3.8+
- **包管理器**: uv (必須使用 uv 進行依賴管理)
- **深度學習框架**: PyTorch 2.0+
- **數據加載**: torchvision.datasets
- **可視化**: matplotlib 3.5+
- **數據處理**: numpy 1.21+

### 7.2 禁止使用的技術/庫

- ❌ TensorFlow / JAX（保持一致性）
- ❌ Flower / PySyft / FedML（不需要 FL 框架）
- ❌ 任何區塊鏈相關庫
- ❌ 分佈式訓練框架（Horovod, DeepSpeed等）
- ❌ Ray, Dask（不需要分佈式）

### 7.3 代碼結構要求
```
project/
├── data/
│   ├── __init__.py
│   ├── loader.py          # 統一的數據加載和分割邏輯
│   └── splits/            # （可選）保存數據分割結果
├── models/
│   ├── __init__.py
│   ├── mnist_net.py       # MNIST 模型定義
│   └── cifar10_net.py     # CIFAR-10 模型定義
├── simulator.py           # 核心模擬邏輯
├── visualize.py           # 可視化腳本
├── main.py                # 主入口
├── config.yaml            # 配置文件
├── pyproject.toml         # uv 依賴定義
├── uv.lock                # uv 鎖定文件
└── results/               # 輸出目錄
    ├── mnist_results.json
    ├── mnist_convergence.png
    ├── cifar10_results.json
    └── cifar10_convergence.png
7.4 運行環境要求

內存: 至少 8GB RAM
GPU: 可選但推薦（加速訓練）

如果有 GPU: 使用 CUDA
如果無 GPU: 使用 CPU（會慢但可行）


磁盤: 至少 2GB 空閒空間（數據集 + 結果）
運行時間預估:

MNIST (GPU): ~30 分鐘
MNIST (CPU): ~3 小時
CIFAR-10 (GPU): ~45 分鐘
CIFAR-10 (CPU): ~4 小時


7.5 項目初始化 (uv)
```bash
# 安裝 uv (如果未安裝)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 初始化項目
uv init

# 添加依賴
uv add torch torchvision numpy matplotlib pyyaml
```

7.6 可重現性要求
python# 所有隨機操作必須使用固定 seed
SEED = 42

要求設置:
- torch.manual_seed(SEED)
- torch.cuda.manual_seed(SEED)
- numpy.random.seed(SEED)
- random.seed(SEED)

確保:
- 數據分割可重現
- 模型初始化可重現
- 訓練過程可重現
```

---

## 八、驗收標準

### 8.1 功能驗收

- [ ] 成功運行 MNIST 實驗（200 epochs）
- [ ] 成功運行 CIFAR-10 實驗（300 epochs）
- [ ] 生成所有要求的 JSON 輸出文件
- [ ] 生成所有要求的 PNG 圖表
- [ ] 終端輸出符合規格
- [ ] 可以通過 `python main.py --dataset mnist` 一鍵運行

### 8.2 正確性驗收

**關鍵預期結果**（必須滿足）:
```
MNIST:
✓ Epoch 1-100: 兩條曲線重合，準確度從 ~10% → ~98%
✓ Epoch 101:   出現明顯分叉
✓ Epoch 101-200: 
  - BlockDFL 曲線下降或停滯（最終 80-88%）
  - Ours 曲線持續上升（最終 99%+）
✓ 最終差距 > 10 個百分點

CIFAR-10:
✓ Epoch 1-150: 兩條曲線重合，準確度從 ~10% → ~75%
✓ Epoch 151:   出現明顯分叉
✓ Epoch 151-300:
  - BlockDFL 曲線下降或停滯（最終 60-70%）
  - Ours 曲線持續上升（最終 85%+）
✓ 最終差距 > 15 個百分點
```

**如果結果不符合預期**:
- 檢查數據分割是否 non-IID
- 檢查 update 選擇邏輯是否正確（最差 vs 最優）
- 檢查模型是否從相同初始化開始
- 檢查 random seed 是否固定

### 8.3 代碼質量驗收

- [ ] 有清晰的函數註釋
- [ ] 關鍵步驟有行內註釋
- [ ] 沒有硬編碼的魔術數字（都在 config.yaml）
- [ ] 可以在 CPU 和 GPU 上運行
- [ ] 錯誤處理（例如數據集下載失敗）

---

## 九、參考資料

### 9.1 BlockDFL 論文關鍵信息

**論文標題**: "BlockDFL: A Blockchain-based Fully Decentralized Peer-to-Peer Federated Learning Framework"

**模型架構來源**: 
- MNIST: Section 5.1, 表格 1
- CIFAR-10: Section 5.1, 腳註 1

**實驗設置**:
- MNIST: 200 rounds, learning rate 0.01, decay 0.99
- CIFAR-10: 300 rounds, learning rate 0.01, decay 0.99

### 9.2 Non-IID 數據分割參考

參考實現邏輯（概念理解）:
```
使用 Dirichlet 分佈是標準做法，參考論文：
"Measuring the Effects of Non-Identical Data Distribution for 
Federated Visual Classification" (2019)

α 參數含義:
- α = 0.1: 極度 non-IID（每個客戶端只有 1-2 類為主）
- α = 0.5: 中度 non-IID（我們使用這個）
- α = 1.0: 輕度 non-IID
- α → ∞: 接近 IID
```

### 9.3 Dirichlet 分佈的直觀理解

不需要深入數學，只需理解效果：
```
假設 10 個類別，4 個客戶端，α = 0.5:

客戶端 1 可能得到: [40%, 30%, 10%, 5%, 5%, 3%, 3%, 2%, 1%, 1%]
客戶端 2 可能得到: [5%, 10%, 35%, 25%, 10%, 5%, 5%, 3%, 1%, 1%]
客戶端 3 可能得到: [10%, 5%, 5%, 8%, 40%, 20%, 7%, 3%, 1%, 1%]
客戶端 4 可能得到: [5%, 3%, 2%, 2%, 5%, 10%, 35%, 25%, 10%, 3%]

每個客戶端都有所有類別，但主導類別不同
```

---

## 十、成功指標

### 10.1 主要目標（必須達成）

- ✅ 視覺上清晰展示攻擊後的性能分叉
- ✅ 定量證明 BlockDFL 在攻擊下性能顯著下降
- ✅ 定量證明我們的方法在攻擊下保持性能

### 10.2 次要目標（錦上添花）

- 🎯 生成可直接用於論文的高質量圖表
- 🎯 實驗可在合理時間內完成（MNIST < 30分鐘）
- 🎯 代碼清晰易讀，方便未來擴展

### 10.3 交付檢查清單

提交前確認：
- [ ] `python main.py --dataset mnist` 可以無錯誤運行
- [ ] `python main.py --dataset cifar10` 可以無錯誤運行
- [ ] `results/` 目錄包含 4 個文件（2 JSON + 2 PNG）
- [ ] PNG 圖片清晰、可讀、符合學術標準
- [ ] JSON 數據完整、格式正確
- [ ] README.md 說明如何運行（包括環境設置）

---

## 十一、常見問題預判

### Q1: 為什麼不直接選擇 4 個完全不同的模型作為 updates？
**A**: 我們需要模擬的是「同一個全局模型的不同局部訓練結果」，而不是「4 個獨立的模型」。在真實 FL 中，所有 aggregator 都從相同的全局模型開始，只是訓練數據不同。

### Q2: 為什麼要在測試集上評估 update 質量，而不是驗證集？
**A**: 為了簡化實驗。在真實系統中確實應該用驗證集，但我們的重點是展示選擇策略的差異，不是優化驗證方法。使用測試集不影響對比結果。

### Q3: 4 個 updates 的質量差異會很明顯嗎？
**A**: 會的。由於 non-IID 分割（α=0.5），每個子集的類別分佈不同，導致訓練出的 update 在測試集上的表現有 2-5% 的差異。這足以產生明顯的累積效果。

### Q4: 如果攻擊前兩個模型的準確度就不相同怎麼辦？
**A**: 這不應該發生，因為：
1. 兩個模型從相同權重初始化
2. 攻擊前選擇相同的 update
3. 使用固定 random seed

如果出現差異，說明實現有 bug。

### Q5: CIFAR-10 的訓練時間太長，可以減少 epochs 嗎？
**A**: 不建議。300 epochs 是為了讓攻擊效果充分顯現。如果時間緊迫，可以：
- 先只運行 MNIST（更快）
- 使用 GPU 加速
- 或者臨時改為 150 total epochs + 75 attack start（但需要在論文中說明）

---

## 十二、開發建議

### 推薦開發順序

1. **Phase 1: 數據準備**（30 分鐘）
   - 實現數據加載
   - 實現 Dirichlet 分割
   - 驗證分割結果（可視化每個子集的類別分佈）

2. **Phase 2: 模型定義**（20 分鐘）
   - 定義 MNIST 模型
   - 定義 CIFAR-10 模型
   - 驗證參數量是否正確

3. **Phase 3: 核心訓練邏輯**（2 小時）
   - 實現單個 update 的生成和評估
   - 實現 update 選擇邏輯
   - 實現主訓練循環

4. **Phase 4: 測試和調試**（1 小時）
   - 運行 10 個 epochs 測試
   - 檢查輸出是否符合預期
   - 修復 bugs

5. **Phase 5: 完整運行**（1-2 小時）
   - 運行完整的 MNIST 實驗
   - 運行完整的 CIFAR-10 實驗

6. **Phase 6: 可視化**（30 分鐘）
   - 生成圖表
   - 調整圖表樣式使其專業

### Debug 技巧

如果結果不符合預期：
```
檢查清單：
□ 兩個模型的初始權重是否完全相同？
   → 打印第一層權重的前 10 個值對比

□ 數據分割是否 non-IID？
   → 打印每個子集的類別分佈統計

□ Update 選擇是否正確？
   → 在每個 epoch 打印：selected_idx, qualities

□ Update 應用是否正確？
   → 驗證 model.parameters 確實改變了

□ Random seed 是否固定？
   → 運行兩次，檢查結果是否完全相同