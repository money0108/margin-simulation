# 遠期契約保證金模擬平台

## 概述

本平台用於模擬遠期契約的保證金計算，支援逐日回放、對沖折減計算、ETF look-through 以及追繳狀態追蹤。

## 制度口徑一句話摘要

> **本制度以固定槓桿計算分邊 Base IM，並以 Base IM 判定大小邊；對沖折減僅適用於小邊，依三產業桶與 3M 加權累積報酬率決定折減率（50% 或 20%）；0050/0056 ETF 採 look-through，成份股完全對沖部分可 100% 減收；維持保證金為當日 IM 的 70%，跌破維持保證金時需追繳回補至當日 IM（100%）。**

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 執行 Streamlit 應用

```bash
cd "P:\Claude Example2\遠期_新\保證金模擬平台"
streamlit run app.py
```

### 3. 執行測試

```bash
pytest tests/ -v
```

## 專案結構

```
保證金模擬平台/
├── app.py                 # Streamlit 主程式
├── requirements.txt       # 依賴套件
├── README.md             # 本文件
├── config/
│   └── settings.yaml     # 設定檔（路徑、槓桿、折減規則）
├── core/
│   ├── __init__.py
│   ├── data_loader.py    # 資料載入與驗證
│   ├── instruments.py    # 標的分類與 ETF look-through
│   ├── buckets.py        # 產業桶分類與 3M 報酬計算
│   ├── margin.py         # 保證金計算核心
│   ├── engine.py         # 回測引擎
│   └── reporting.py      # 報表與稽核包
└── tests/
    ├── test_margin_rules.py      # 保證金規則測試
    ├── test_etf_lookthrough.py   # ETF look-through 測試
    └── test_bucket_reduction.py  # 產業桶折減測試
```

## 資料來源

| 資料 | 路徑 | 格式 |
|------|------|------|
| 股價歷史數據 | `P:\Claude Example2\遠期_新\Data\stock_price_last2y.csv` | CSV |
| 公司名稱對照 | `P:\Claude Example2\遠期_新\Data\台股代號公司全名對照.csv` | CSV |
| 產業分類 | `P:\Claude Example2\遠期_新\產業對照\上市櫃個股產業分類.xlsx` | XLSX |
| 槓桿倍數表 | `P:\Claude Example2\遠期_新\產業對照\期貨槓桿倍數_完整表.xlsx` | XLSX |
| ETF 成分股權重 | `P:\Claude Example2\遠期_新\產業對照\ETF成份股權重.xlsx` | XLSX |
| 示範部位 | `P:\Claude Example2\遠期_新\模擬部位\模擬1\模擬1l部位.xlsx` | XLSX |

## 槓桿倍數規則

| 標的類型 | 槓桿倍數 |
|---------|---------|
| 股票期貨標的 | 5x |
| 0050/0056 成份股 | 4x |
| 其他股票 | 3x |
| 0050/0056 ETF | 7x |

## 對沖折減規則

| 對沖類型 | 條件 | 折減率 |
|---------|------|--------|
| 同桶對沖 | 3M 報酬差 ≥ 10% | 50% |
| 同桶對沖 | 3M 報酬差 < 10% | 20% |
| 跨桶對沖 | - | 20% |
| ETF 完全對沖 | 成分股反向曝險 | 100% |

## 輸入格式

### 部位上傳（Excel）

| 欄位 | 說明 | 範例 |
|------|------|------|
| 代號 | 證券代碼 | 2330 |
| 買進張數 | 多方張數 | 50 |
| 賣出張數 | 空方張數 | -100 |

### 標準格式（可選）

| 欄位 | 說明 | 必要 |
|------|------|------|
| trade_date | 交易日期 (YYYY-MM-DD) | 否 |
| code | 證券代碼 | 是 |
| side | LONG/SHORT | 是 |
| qty | 股數 | 是 |
| instrument | STK/ETF | 否 |
| entry_price | 進場價格 | 否 |

## 輸出欄位

| 欄位 | 說明 |
|------|------|
| date | 日期 |
| Long_MV | 多方市值 |
| Short_MV | 空方市值 |
| Base_IM_long | 多方基礎 IM |
| Base_IM_short | 空方基礎 IM |
| IM_big | 大邊 IM |
| IM_small_before | 小邊 IM（折減前） |
| reduction_etf_100 | ETF 100% 折減 |
| reduction_same_bucket | 同桶折減 |
| reduction_cross_bucket | 跨桶折減 |
| IM_today | 當日 IM |
| MM_today | 維持保證金 (70%) |
| Equity | 權益 |
| margin_call_flag | 追繳旗標 |
| Required_Deposit | 追繳金額 |
| Gross_Lev | Gross 槓桿（有折減）|
| Raw_Lev | 無折減槓桿 |

## 稽核包內容

下載的稽核包 ZIP 包含：

```
audit_package_YYYYMMDD_HHMMSS.zip
├── final_timeseries.csv      # 完整時序資料
├── margin_call_events.csv    # 追繳事件
├── assumptions.md            # 假設與保守口徑說明
├── verification.json         # 驗證結果
├── inputs_snapshot/
│   └── positions.csv         # 原始部位快照
└── calc_steps/
    ├── YYYYMMDD_summary.csv          # 首/中/末日摘要
    ├── YYYYMMDD_bucket_hedge.csv     # 產業桶對沖明細
    └── YYYYMMDD_reduction_breakdown.csv  # 折減分解
```

## 關鍵函式

```python
# 載入股價
from core.data_loader import load_prices
prices = load_prices("path/to/prices.csv")

# 載入部位
from core.data_loader import load_positions
positions = load_positions("path/to/positions.xlsx")

# 執行回測
from core.engine import BacktestEngine
from core.data_loader import DataLoader

loader = DataLoader()
engine = BacktestEngine(loader)
results = engine.run(
    positions=positions,
    start_date=pd.Timestamp("2024-01-01"),
    end_date=pd.Timestamp("2024-03-31"),
    equity=100_000_000
)

# 取得時序資料
timeseries = results.timeseries_df

# 驗證結果
from core.reporting import verify
verification = verify(results)
```

## 單元測試說明

### test_margin_rules.py
- 同桶 3M 報酬差 ≥ 10% → 小邊折減 50%
- 大邊不折減
- 追繳觸發條件
- 槓桿分類正確性

### test_etf_lookthrough.py
- ETF 正確拆解至成分股
- 完全對沖 100% 減收
- 部分對沖處理
- ETF 成分股使用 7x 槓桿

### test_bucket_reduction.py
- 跨桶對沖 20% 折減
- 折減只作用於小邊
- 同桶低報酬差 20% 折減
- 產業桶分類正確
- Base IM 加權折減

## 注意事項

1. **缺碼處理**：若股票缺少產業分類或槓桿資訊，以保守口徑處理（3x 槓桿、非金電桶），並在稽核包中記錄。

2. **日期對齊**：價格資料以交易日對齊，缺值採前值填補。

3. **張數轉股數**：預設 1 張 = 1,000 股。

4. **金額精度**：金額四捨五入至元（可在設定檔調整）。

5. **ETF 代碼**：支援 `0050`/`50`、`0056`/`56` 格式，系統會自動標準化。

## 錯誤處理

- 檔案不存在：顯示明確路徑與修復建議
- 欄位缺漏：列出缺少的必要欄位
- 資料異常：在報告中揭露缺值處理比率

## 版本資訊

- 版本：v1.0.0
- 更新日期：2026-02-01
- 制度版本：MARGIN_POLICY v1.6.1

## 聯絡資訊

如有問題，請聯繫系統管理員。
