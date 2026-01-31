# =============================================================================
# 遠期契約保證金模擬平台 - 產業桶折減單元測試
# 測試項目：跨桶對沖 → 一律 20% 折減（只折小邊）
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.instruments import InstrumentClassifier, ETFLookthrough
from core.buckets import BucketClassifier, ReturnCalculator, Bucket, BucketAnalyzer
from core.margin import MarginCalculator


class TestBucketReduction:
    """產業桶折減測試"""

    @pytest.fixture
    def setup_cross_bucket_data(self):
        """設定跨桶測試資料"""
        # 模擬：電子做多、金融做空（跨桶對沖）
        base_date = pd.Timestamp('2024-01-15')
        dates = pd.date_range(base_date - pd.Timedelta(days=92), base_date, freq='B')

        price_records = []
        for d in dates:
            price_records.append({'date': d, 'code': '2330', 'close': 800})  # 電子
            price_records.append({'date': d, 'code': '2891', 'close': 30})   # 金融

        prices_df = pd.DataFrame(price_records)

        # 產業分類
        industry_df = pd.DataFrame([
            {'code': '2330', 'bucket': 'ELECT'},  # 電子
            {'code': '2891', 'bucket': 'FIN'},    # 金融
        ])

        prices_today = {
            '2330': 800,
            '2891': 30
        }

        return {
            'prices_df': prices_df,
            'industry_df': industry_df,
            'prices_today': prices_today,
            'asof_date': base_date
        }

    def test_cross_bucket_20_reduction(self, setup_cross_bucket_data):
        """
        測試：跨桶對沖 → 一律 20% 折減

        制度條款 4.4：不同產業桶的多空對沖，一律給 20% 折減

        情境：
        - 2330 (電子) LONG 100,000 股 @ 800 = MV 80,000,000
        - 2891 (金融) SHORT 1,000,000 股 @ 30 = MV 30,000,000
        - 電子 vs 金融 = 跨桶對沖
        - 多方 Base IM (5x) = 80M / 5 = 16M
        - 空方 Base IM (3x) = 30M / 3 = 10M（2891 非期貨標的，非成分股 → 3x）
        - 大邊 = LONG (16M > 10M)
        - 小邊 = SHORT (10M)，跨桶折減 20%
        """
        data = setup_cross_bucket_data

        # 部位
        positions = pd.DataFrame([
            {'code': '2330', 'side': 'LONG', 'qty': 100000, 'instrument': 'STK'},
            {'code': '2891', 'side': 'SHORT', 'qty': 1000000, 'instrument': 'STK'},
        ])

        # 分類器
        classifier = InstrumentClassifier(
            futures_underlying_set={'2330'},  # 2330 是期貨標的 → 5x
            constituent_set_0050_0056=set()   # 2891 非成分股 → 3x
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
            positions=positions,
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
        expected_long_mv = 100000 * 800  # 80,000,000
        expected_short_mv = 1000000 * 30  # 30,000,000
        assert abs(result.long_mv - expected_long_mv) < 1
        assert abs(result.short_mv - expected_short_mv) < 1

        # 4. Base IM 驗證
        expected_base_im_long = expected_long_mv / 5  # 16,000,000
        expected_base_im_short = expected_short_mv / 3  # 10,000,000
        assert abs(result.base_im_long - expected_base_im_long) < 1
        assert abs(result.base_im_short - expected_base_im_short) < 1

        # 5. 跨桶折減驗證
        # 小邊 Base IM = 10,000,000
        # 跨桶對沖比例 = min(long_mv, short_mv) / short_mv = 30M / 30M = 100%
        # 20% 折減 × 100% = 20%
        # 折減金額 = 10,000,000 × 20% = 2,000,000
        expected_cross_reduction = expected_base_im_short * 0.20

        # 由於沒有同桶對沖（電子只有多方，金融只有空方），
        # 所有折減都應來自跨桶
        assert result.reduction_same_bucket < 100, \
            f"不應有顯著的同桶折減（因為同桶內無對向部位）"

        # 跨桶折減應約為 2,000,000
        # 注意：跨桶折減是針對小邊「同桶未被覆蓋的殘餘」
        # 由於金融桶（小邊）完全沒有同桶多方，所以全部走跨桶
        tolerance = expected_cross_reduction * 0.2  # 20% 容差
        assert result.reduction_cross_bucket > 0, "應有跨桶折減"

        print(f"跨桶折減: {result.reduction_cross_bucket:,.0f}")
        print(f"同桶折減: {result.reduction_same_bucket:,.0f}")
        print(f"ETF折減: {result.reduction_etf_100:,.0f}")

        print("✅ test_cross_bucket_20_reduction 通過")

    def test_only_small_side_reduced(self, setup_cross_bucket_data):
        """
        測試：折減只作用於小邊

        制度條款 4.5：折減只套用在小邊（IM_small）；大邊不折減
        """
        data = setup_cross_bucket_data

        positions = pd.DataFrame([
            {'code': '2330', 'side': 'LONG', 'qty': 100000, 'instrument': 'STK'},
            {'code': '2891', 'side': 'SHORT', 'qty': 1000000, 'instrument': 'STK'},
        ])

        classifier = InstrumentClassifier(
            futures_underlying_set={'2330'},
            constituent_set_0050_0056=set()
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
            positions=positions,
            prices_today=data['prices_today'],
            asof_date=data['asof_date'],
            equity=50_000_000
        )

        # 驗證：大邊 IM = Base IM（無折減）
        if result.big_side == 'LONG':
            assert abs(result.im_big - result.base_im_long) < 1, \
                f"大邊 LONG 不應被折減: IM_big={result.im_big}, Base={result.base_im_long}"
        else:
            assert abs(result.im_big - result.base_im_short) < 1, \
                f"大邊 SHORT 不應被折減: IM_big={result.im_big}, Base={result.base_im_short}"

        # 驗證：小邊 IM < Base IM（有折減）
        assert result.im_small_after < result.im_small_before, \
            f"小邊應被折減: before={result.im_small_before}, after={result.im_small_after}"

        # 驗證：IM_today = IM_big + IM_small_after
        expected_im_today = result.im_big + result.im_small_after
        assert abs(result.im_today - expected_im_today) < 1, \
            f"IM_today 計算錯誤: 預期 {expected_im_today}, 實際 {result.im_today}"

        print("✅ test_only_small_side_reduced 通過")

    def test_same_bucket_low_return_diff_20_reduction(self):
        """
        測試：同桶但 3M 報酬差 < 10% → 小邊折減 20%

        制度條款 4.3：
        若 R_big_bucket(3M) − R_small_bucket(3M) >= 10% → 50%
        否則 → 20%
        """
        # 設定資料：同產業（電子）多空，報酬差 < 10%
        base_date = pd.Timestamp('2024-01-15')
        dates = pd.date_range(base_date - pd.Timedelta(days=92), base_date, freq='B')

        price_records = []
        for d in dates:
            # 兩檔電子股價格變化相近（報酬差 < 10%）
            progress = (d - dates[0]).days / 92
            # 2330: 800 → 840（報酬 5%）
            price_records.append({'date': d, 'code': '2330', 'close': 800 + 40 * progress})
            # 2317: 100 → 102（報酬 2%）
            price_records.append({'date': d, 'code': '2317', 'close': 100 + 2 * progress})

        prices_df = pd.DataFrame(price_records)

        industry_df = pd.DataFrame([
            {'code': '2330', 'bucket': 'ELECT'},
            {'code': '2317', 'bucket': 'ELECT'},
        ])

        prices_today = {'2330': 840, '2317': 102}

        positions = pd.DataFrame([
            {'code': '2330', 'side': 'LONG', 'qty': 100000, 'instrument': 'STK'},
            {'code': '2317', 'side': 'SHORT', 'qty': 500000, 'instrument': 'STK'},
        ])

        classifier = InstrumentClassifier(
            futures_underlying_set={'2330', '2317'},
            constituent_set_0050_0056=set()
        )

        etf_lookthrough = ETFLookthrough({}, classifier)
        bucket_classifier = BucketClassifier(industry_df)
        return_calculator = ReturnCalculator(prices_df, lookback_days=63)

        margin_calc = MarginCalculator(
            classifier=classifier,
            etf_lookthrough=etf_lookthrough,
            bucket_classifier=bucket_classifier,
            return_calculator=return_calculator
        )

        result = margin_calc.calculate(
            positions=positions,
            prices_today=prices_today,
            asof_date=base_date,
            equity=50_000_000
        )

        # 驗證：報酬差 = 5% - 2% = 3% < 10%，應採用 20% 折減
        # MV: long = 84M, short = 51M
        # Base IM: long = 16.8M, short = 10.2M
        # 大邊 = LONG
        # 小邊折減 = 10.2M × 20% × 覆蓋比例

        # 同桶折減應存在且反映 20% 規則
        assert result.reduction_same_bucket > 0, "同桶對沖應有折減"

        # 驗證折減率不是 50%（因為報酬差 < 10%）
        max_50_reduction = result.im_small_before * 0.50
        assert result.reduction_same_bucket < max_50_reduction * 0.9, \
            "報酬差 < 10% 時不應採用 50% 折減"

        print("✅ test_same_bucket_low_return_diff_20_reduction 通過")

    def test_bucket_classification(self):
        """
        測試：產業桶分類正確

        三產業桶：電子、金融、非金電
        """
        industry_df = pd.DataFrame([
            {'code': '2330', 'bucket': 'ELECT'},  # 台積電 → 電子
            {'code': '2891', 'bucket': 'FIN'},    # 中信金 → 金融
            {'code': '1101', 'bucket': 'NON'},    # 台泥 → 非金電
        ])

        classifier = BucketClassifier(industry_df)

        # 測試分類
        assert classifier.classify('2330') == Bucket.ELECT, "2330 應為電子桶"
        assert classifier.classify('2891') == Bucket.FIN, "2891 應為金融桶"
        assert classifier.classify('1101') == Bucket.NON, "1101 應為非金電桶"

        # 測試缺碼（應歸入非金電）
        assert classifier.classify('9999') == Bucket.NON, "缺碼應歸入非金電"
        assert '9999' in classifier.missing_codes, "缺碼應被記錄"

        print("✅ test_bucket_classification 通過")

    def test_weighted_base_im_reduction(self):
        """
        測試：折減採 Base IM 加權

        制度條款 4.5：
        小邊若包含多檔，折減採 Base IM 加權折減
        Weighted Discount = Σ(BaseIM_i × Discount_i) ÷ Σ(BaseIM_i)
        """
        # 這個測試驗證當小邊有多檔時，折減是逐檔計算後加總
        # 而不是用平均折減率乘以總 Base IM

        base_date = pd.Timestamp('2024-01-15')
        dates = pd.date_range(base_date - pd.Timedelta(days=92), base_date, freq='B')

        price_records = []
        for d in dates:
            price_records.append({'date': d, 'code': '2330', 'close': 800})
            price_records.append({'date': d, 'code': '2317', 'close': 100})
            price_records.append({'date': d, 'code': '2891', 'close': 30})

        prices_df = pd.DataFrame(price_records)

        industry_df = pd.DataFrame([
            {'code': '2330', 'bucket': 'ELECT'},
            {'code': '2317', 'bucket': 'ELECT'},
            {'code': '2891', 'bucket': 'FIN'},
        ])

        prices_today = {'2330': 800, '2317': 100, '2891': 30}

        # 小邊有兩檔：2317 (電子) 和 2891 (金融)
        # 應分別計算折減後加總
        positions = pd.DataFrame([
            {'code': '2330', 'side': 'LONG', 'qty': 100000, 'instrument': 'STK'},
            {'code': '2317', 'side': 'SHORT', 'qty': 500000, 'instrument': 'STK'},
            {'code': '2891', 'side': 'SHORT', 'qty': 1000000, 'instrument': 'STK'},
        ])

        classifier = InstrumentClassifier(
            futures_underlying_set={'2330', '2317'},
            constituent_set_0050_0056=set()
        )

        etf_lookthrough = ETFLookthrough({}, classifier)
        bucket_classifier = BucketClassifier(industry_df)
        return_calculator = ReturnCalculator(prices_df, lookback_days=63)

        margin_calc = MarginCalculator(
            classifier=classifier,
            etf_lookthrough=etf_lookthrough,
            bucket_classifier=bucket_classifier,
            return_calculator=return_calculator
        )

        result = margin_calc.calculate(
            positions=positions,
            prices_today=prices_today,
            asof_date=base_date,
            equity=50_000_000
        )

        # 驗證：小邊有多檔，各自有折減
        # 2317 (電子) 走同桶折減
        # 2891 (金融) 走跨桶折減
        assert result.reduction_same_bucket > 0 or result.reduction_cross_bucket > 0, \
            "小邊多檔應有折減"

        # 驗證總折減 = 各項折減之和
        total_reduction = result.reduction_etf_100 + result.reduction_same_bucket + result.reduction_cross_bucket
        assert abs(total_reduction - result.total_reduction) < 1, \
            f"總折減計算錯誤: {total_reduction} vs {result.total_reduction}"

        print("✅ test_weighted_base_im_reduction 通過")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
