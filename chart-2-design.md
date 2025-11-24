Chart 7-2A（時間序列 Line Chart）
X軸：訓練輪次 (RR
R) 或時間（單位：天）


範圍：0 到 100 輪（或 0 到 5000 輪，取決於你的訓練時長設定）
含義：代表聯邦學習進行的總進度

Y軸：Stake 總值（USD 或虛擬代幣）

範圍：0 到 10,000（相對值，或實際 USD）
標度：線性或對數都可，建議線性便於直觀對比
含義：各節點類型累積的Stake

Z軸（3條不同的曲線）：

藍色曲線 — BlockDFL 中的誠實節點 Stake

公式：ShonestBlockDFL(r)=S0+r×rewardhonestS_{honest}^{BlockDFL}(r) = S_0 + r \times \text{reward}_{honest}
ShonestBlockDFL​(r)=S0​+r×rewardhonest​
特徵：緩慢線性增長，因為獎勵被平均分配
注：假設誠實節點穩定參與，每輪獲得固定獎勵


紅色曲線 — BlockDFL 中的惡意節點 Stake

公式：SmaliciousBlockDFL(r)=S0+r×rewardmalicious+strategic_starvationS_{malicious}^{BlockDFL}(r) = S_0 + r \times \text{reward}_{malicious} + \text{strategic\_starvation}
SmaliciousBlockDFL​(r)=S0​+r×rewardmalicious​+strategic_starvation
特徵：超線性增長，因為惡意節點控制委員會後：

（a）獨佔大部分獎勵
（b）通過 "Strategic Starvation" 機制排斥誠實節點的獎勵


在某個臨界點 r∗r^*
r∗ 後，紅線
超越藍線，代表惡意Stake達到多數
紅色區域應該用陰影標示，表示「系統被接管」


綠色曲線 — 你的方法中的惡意節點 Stake

公式：SmaliciousOurs(r)=S0+r×rewardhonest until r=rattackS_{malicious}^{Ours}(r) = S_0 + r \times \text{reward}_{honest} \text{ until } r = r_{attack}
SmaliciousOurs​(r)=S0​+r×rewardhonest​ until r=rattack​
特徵：平緩增長直到攻擊瞬間，然後：

在 r=rattackr = r_{attack}
r=rattack​ 時，Stake
瞬間歸零（Slashing）


視覺上呈現為一條陡直的垂直下降，宛如懸崖
可選：在歸零後上翹一點，表示罰沒後的最小Stake