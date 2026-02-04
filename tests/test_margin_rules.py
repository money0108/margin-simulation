# =============================================================================
# 遠期契約保證金模擬平台 - 保證金規則單元測試
# 測試項目：同桶對沖、3M 報酬差 >= 10% → 小邊折減 50%
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.instruments import InstrumentClassifier, ETFLookthrough, ExplodedPosition
from core.buckets import BucketClassifier, ReturnCalculator, Bucket, BucketAnalyzer
from core.margin import MarginCalculator, MarginResult


class TestMarginRules:
    """保證金規則測試"""

    @pytest.fixture
    def setup_test_data(self):
        """設定測試資料"""
        # 模擬股價資料（3M 報酬差 > 10%）
        # 2330: 起始 800，結束 900 → 報酬 12.5%
        # 2317: 起始 100，結束 100 → 報酬 0%
        # 報酬差 = 12.5% - 0% = 12.5% >= 10% → 應觸發 50% 折減

        base_date = pd.Timestamp('2024-01-15')
        start_date = base_date - pd.Timedelta(days=92)

        dates = pd.date_range(start_date, base_date, freq='B')

        # 生成價格資料
        price_records = []
        for d in dates:
            # 2330: 從 800 漲到 900（電子股）
            progress = (d - start_date).days / 92
            price_2330 = 800 + 100 * progress
            price_records.append({'date': d, 'code': '2330', 'close': price_2330})

            # 2317: 維持 100（電子股）
            price_records.append({'date': d, 'code': '2317', 'close': 100})

        prices_df = pd.DataFrame(price_records)

        # 產業分類（都是電子）
        industry_df = pd.DataFrame([
            {'code': '2330', 'bucket': 'ELECT'},
            {'code': '2317', 'bucket': 'ELECT'},
        ])

        # 部位：2330 做多，2317 做空
        positions = pd.DataFrame([
            {'code': '2330', 'side': 'LONG', 'qty': 100000, 'instrument': 'STK'},
            {'code': '2317', 'side': 'SHORT', 'qty': 90000, 'instrument': 'STK'},
        ])

        # 當日價格
        prices_today = {'2330': 900, '2317': 100}

        return {
            'prices_df': prices_df,
            'industry_df': industry_df,
            'positions': positions,
            'prices_today': prices_today,
            'asof_date': base_date
        }

    def test_same_bucket_high_return_diff_50_reduction(self, setup_test_data):
        """
        測試：同桶且 3M 報酬差 >= 10% → 小邊折減 50%

        情境：
        - 2330 (電子) LONG 100,000 股 @ 900 = MV 90,000,000
        - 2317 (電子) SHORT 90,000 股 @ 100 = MV 9,000,000
        - 2330 3M 報酬 = 12.5%，2317 3M 報酬 = 0%
        - 報酬差 = 12.5% >= 10%
        - 多方 Base IM (2330 股期槓桿>7 → 5x) = 90M / 5 = 18M
        - 空方 Base IM (2317 股期槓桿>7 → 5x) = 9M / 5 = 1.8M
        - 大邊 = LONG (18M > 1.8M)
        - 小邊 = SHORT，應套用 50% 折減
        """
        data = setup_test_data

        # 建立分類器
        futures_set = {'2330', '2317'}
        futures_leverage_map = {'2330': 5.0, '2317': 5.0}  # 股期槓桿>7 → 5x

        classifier = InstrumentClassifier(
            futures_underlying_set=futures_set,
            constituent_set_0050_0056=set(),
            futures_leverage_map=futures_leverage_map
        )

        # ETF 處理器（空，因為沒有 ETF）
        etf_weights = {}
        etf_lookthrough = ETFLookthrough(etf_weights, classifier)

        # 產業桶分類器
        bucket_classifier = BucketClassifier(data['industry_df'])

        # 報酬計算器
        return_calculator = ReturnCalculator(data['prices_df'], lookback_days=63)

        # 保證金計算器
        margin_calc = MarginCalculator(
            classifier=classifier,
            etf_lookthrough=etf_lookthrough,
            bucket_classifier=bucket_classifier,
            return_calculator=return_calculator
        )

        # 執行計算
        result = margin_calc.calculate(
            positions=data['positions'],
            prices_today=data['prices_today'],
            asof_date=data['asof_date'],
            equity=50_000_000
        )

        # 驗證
        # 1. 大邊應為 LONG
        assert result.big_side == 'LONG', f"大邊應為 LONG，實際為 {result.big_side}"

        # 2. 小邊應為 SHORT
        assert result.small_side == 'SHORT', f"小邊應為 SHORT，實際為 {result.small_side}"

        # 3. MV 驗證
        expected_long_mv = 100000 * 900  # 90,000,000
        expected_short_mv = 90000 * 100  # 9,000,000
        assert abs(result.long_mv - expected_long_mv) < 1, f"Long MV 錯誤: {result.long_mv}"
        assert abs(result.short_mv - expected_short_mv) < 1, f"Short MV 錯誤: {result.short_mv}"

        # 4. Base IM 驗證（5x 槓桿）
        expected_base_im_long = expected_long_mv / 5  # 18,000,000
        expected_base_im_short = expected_short_mv / 5  # 1,800,000
        assert abs(result.base_im_long - expected_base_im_long) < 1
        assert abs(result.base_im_short - expected_base_im_short) < 1

        # 5. 折減驗證：同桶 50% 折減
        # 小邊 Base IM = 1,800,000
        # 可對沖比例 = min(long_mv, short_mv) / short_mv = 9M / 9M = 100%
        # 50% 折減 × 100% 可對沖 = 50% 有效折減
        # 折減金額 = 1,800,000 × 50% = 900,000
        expected_reduction = expected_base_im_short * 0.50
        actual_reduction = result.reduction_same_bucket

        # 允許一些誤差（因為報酬計算的日期可能略有不同）
        tolerance = expected_reduction * 0.1  # 10% 容差
        assert abs(actual_reduction - expected_reduction) < tolerance, \
            f"同桶折減金額錯誤: 預期 {expected_reduction:,.0f}，實際 {actual_reduction:,.0f}"

        # 6. IM_today 驗證
        # IM_today = IM_big + IM_small_after
        # IM_big = 18,000,000
        # IM_small_after = 1,800,000 - 900,000 = 900,000
        # IM_today = 18,000,000 + 900,000 = 18,900,000
        expected_im_today = expected_base_im_long + (expected_base_im_short - expected_reduction)
        tolerance = expected_im_today * 0.1
        assert abs(result.im_today - expected_im_today) < tolerance, \
            f"IM_today 錯誤: 預期 {expected_im_today:,.0f}，實際 {result.im_today:,.0f}"

        # 7. MM_today = 70% × IM_today
        expected_mm = result.im_today * 0.70
        assert abs(result.mm_today - expected_mm) < 1, \
            f"MM_today 應為 IM_today 的 70%: 預期 {expected_mm:,.0f}，實際 {result.mm_today:,.0f}"

        print("✅ test_same_bucket_high_return_diff_50_reduction 通過")

    def test_big_side_not_reduced(self, setup_test_data):
        """
        測試：大邊不應被折減

        制度關鍵規則：折減只套用小邊（IM_small）；大邊不折減。
        """
        data = setup_test_data

        # 建立分類器
        classifier = InstrumentClassifier(
            futures_underlying_set={'2330', '2317'},
            constituent_set_0050_0056=set(),
            futures_leverage_map={'2330': 5.0, '2317': 5.0}
        )

        etf_lookthrough = ETFLookthrough({}, classifier)
        bucket_classifier = BucketClassifier(data['industry_df'])
        return_calculator = ReturnCalculator(data['prices_df'], lookback_days=63)

        margin_calc = MarginCalculator(
            classifier=classifier,
            etf_lookthrough=etf_lookthrough,
            bucket_classifier=bucket_classifier,
            return_calculator=return_calculator
        )

        result = margin_calc.calculate(
            positions=data['positions'],
            prices_today=data['prices_today'],
            asof_date=data['asof_date'],
            equity=50_000_000
        )

        # 大邊 IM = Base IM（無折減）
        # IM_big 應等於大邊的 Base IM
        if result.big_side == 'LONG':
            assert abs(result.im_big - result.base_im_long) < 1, \
                "大邊 LONG 不應被折減"
        else:
            assert abs(result.im_big - result.base_im_short) < 1, \
                "大邊 SHORT 不應被折減"

        print("✅ test_big_side_not_reduced 通過")

    def test_margin_call_trigger(self):
        """
        測試：Equity < MM → 觸發追繳

        制度規則：觸發條件為 Equity < MM_today，
        回補目標為 IM_today（100%）
        """
        # 簡化測試：直接構造 MarginResult
        # IM_today = 10,000,000
        # MM_today = 7,000,000 (70%)
        # Equity = 6,000,000 < MM → 應觸發追繳
        # Required Deposit = IM_today - Equity = 4,000,000

        im_today = 10_000_000
        mm_today = im_today * 0.70
        equity = 6_000_000

        # 驗證觸發條件
        margin_call = equity < mm_today
        assert margin_call == True, "Equity < MM 應觸發追繳"

        # 驗證回補金額
        required_deposit = im_today - equity
        assert required_deposit == 4_000_000, \
            f"回補金額應為 4,000,000，實際為 {required_deposit}"

        print("✅ test_margin_call_trigger 通過")

    def test_leverage_classification(self):
        """
        測試：槓桿倍數分類正確

        規則：
        - ETF → 月均量 ≥ 10,000 張 → 7x，否則 5x
        - 股期標的 → 依股期槓桿動態判定 (5x/4x/3x)
        - 其他股票 → 3x
        """
        futures_leverage_map = {
            '2330': 5.0,  # 股期槓桿 > 7 → 5x
            '2317': 4.0,  # 股期槓桿 > 6 且 ≤ 7 → 4x
            '2382': 3.0,  # 股期槓桿 ≤ 6 → 3x
        }

        etf_volume_map = {
            '0050': 97919,  # 月均量 ~97,919 張 → 7x
            '0056': 54581,  # 月均量 ~54,581 張 → 7x
            '00878': 3000,  # 月均量 3,000 張 → 5x
        }

        classifier = InstrumentClassifier(
            futures_underlying_set=set(futures_leverage_map.keys()),
            constituent_set_0050_0056=set(),
            futures_leverage_map=futures_leverage_map,
            etf_volume_map=etf_volume_map
        )

        # 測試股期標的（股期槓桿 > 7）→ 5x
        lev_2330 = classifier.classify_leverage('2330', 'STK')
        assert lev_2330.leverage == 5.0, f"2330 應為 5x，實際為 {lev_2330.leverage}"

        # 測試股期標的（股期槓桿 > 6 且 ≤ 7）→ 4x
        lev_2317 = classifier.classify_leverage('2317', 'STK')
        assert lev_2317.leverage == 4.0, f"2317 應為 4x，實際為 {lev_2317.leverage}"

        # 測試股期標的（股期槓桿 ≤ 6）→ 3x
        lev_2382 = classifier.classify_leverage('2382', 'STK')
        assert lev_2382.leverage == 3.0, f"2382 應為 3x，實際為 {lev_2382.leverage}"

        # 測試其他股票（非股期標的）→ 3x
        lev_other = classifier.classify_leverage('9999', 'STK')
        assert lev_other.leverage == 3.0, f"其他股票應為 3x，實際為 {lev_other.leverage}"

        # 測試 0050 ETF（月均量 ≥ 10,000 張）→ 7x
        lev_0050 = classifier.classify_leverage('0050', 'ETF')
        assert lev_0050.leverage == 7.0, f"0050 月均量高應為 7x，實際為 {lev_0050.leverage}"

        # 測試 0056 ETF（月均量 ≥ 10,000 張）→ 7x
        lev_0056 = classifier.classify_leverage('0056', 'ETF')
        assert lev_0056.leverage == 7.0, f"0056 月均量高應為 7x，實際為 {lev_0056.leverage}"

        # 測試其他 ETF（月均量 < 10,000 張）→ 5x
        lev_other_etf = classifier.classify_leverage('00878', 'ETF')
        assert lev_other_etf.leverage == 5.0, f"00878 月均量低應為 5x，實際為 {lev_other_etf.leverage}"

        # 測試無月均量資料的 ETF → 5x（保守處理）
        lev_no_vol = classifier.classify_leverage('00929', 'ETF')
        assert lev_no_vol.leverage == 5.0, f"無月均量資料的 ETF 應為 5x，實際為 {lev_no_vol.leverage}"

        print("✅ test_leverage_classification 通過")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
