# 多臂吃角子老虎 (Multi-Armed Bandit) 模擬
## 六種探索–利用策略比較

本專案透過模擬比較六種常見的 Multi-Armed Bandit 演算法，觀察在有限預算下，不同探索策略如何影響整體報酬與後悔值（regret）。

目標是理解：

如何在「探索 (exploration)」與「利用 (exploitation)」之間取得最佳平衡。

---

# 問題說明

本問題為三個選擇的決策情境（3-armed bandit）。

每次選擇一個選項（bandit），會得到：

reward = 1（成功）
reward = 0（失敗）

三個 bandit 的真實期望報酬為：

| Bandit | 真實平均報酬 |
|--------|-------------|
| A | 0.8 |
| B | 0.7 |
| C | 0.5 |

總嘗試次數：

T = 10000

模擬次數：

runs = 300

目標：

最大化總報酬 (total reward)

同時學習最佳策略。

---

# 六種演算法介紹

本專案比較以下六種方法：

---

## 1. A/B Testing

傳統統計方法。

流程：

前 2000 次：
平均分配給 A 與 B
各 1000 次
不測試 C

後 8000 次：
選擇平均報酬較高的選項

缺點：

不會動態調整策略
可能浪費資源在較差選項

---

## 2. Optimistic Initial Values（樂觀初始值）

將所有選項的初始估計值設為較高數值（例如 1）。

效果：

演算法會主動嘗試所有選項
逐漸收斂到最佳選項

優點：

簡單
不需要額外隨機探索

---

## 3. ε-Greedy（ε = 0.1）

策略：

90% 機率：
選擇目前平均報酬最高的選項

10% 機率：
隨機探索

特性：

持續探索
容易實作
常作為基準方法

---

## 4. Softmax（Boltzmann Exploration）

選擇機率：

P(a) ∝ exp(Q(a)/τ)

τ（temperature）控制探索程度：

τ 越小：
越偏向選擇最佳選項

τ 越大：
越隨機探索

優點：

探索過程較平滑
不會突然停止探索

---

## 5. UCB（Upper Confidence Bound）

選擇依據：

平均報酬 + 不確定性

公式概念：

Q(a) + c √( log(t) / N(a) )

特性：

偏好：
高平均報酬
資料較少的選項

具有理論保證
後悔值成長速度較慢

---

## 6. Thompson Sampling

貝氏方法（Bayesian approach）。

每個選項使用機率分布描述不確定性。

流程：

從每個選項的機率分布中抽樣
選擇抽樣結果最大的選項

優點：

自動平衡探索與利用
實務中表現通常最佳

---

# 如何執行

安裝套件：

pip install numpy matplotlib

執行程式：

python main.py

總模擬次數為：

10000 × 300 × 6
= 18,000,000 次迭代

---

# 如何解讀圖

程式會產生一張圖：

X 軸：

Round Index（第幾次選擇）

Y 軸：

Cumulative Regret（累積後悔值）
採用 log scale（對數尺度）

後悔值 (regret) 代表：

與最佳策略相比損失了多少報酬。

最佳策略：

每次都選擇 Bandit A

累積 regret 越低：

表示演算法越有效率。

常見結果：

Thompson Sampling
UCB

通常最低 regret。

A/B testing

通常較高 regret。

原因：

早期浪費太多次數在次佳選項。

---

# 理論背景

Multi-Armed Bandit 問題核心挑戰：

是否應該：

探索新選項？
還是利用已知最佳選項？

探索太多：

浪費資源

探索太少：

可能錯過最佳選項

理想策略：

用最少探索成本
找到最佳行動

常見 regret 成長速度：

隨機策略：

regret ∝ T

有效策略：

regret ∝ log(T)

代表：

長期效率更高。

UCB 與 Thompson Sampling
通常具有最佳理論性質。

---

# 參數說明

可調整參數：

T = 10000

總決策次數

runs = 300

模擬重複次數
越高結果越穩定

epsilon = 0.1

ε-greedy 的探索比例

tau = 0.2

softmax 的隨機程度

c = 2

UCB 的探索強度

參數影響：

探索越多：

學習較安全
但短期報酬較低

探索越少：

收斂較快
但可能選錯選項

---

# 重點結論

探索是必要的
但過度探索會降低總報酬

有效演算法應：

快速識別最佳選項
同時避免過多嘗試次佳選項

通常表現最佳：

UCB
Thompson Sampling