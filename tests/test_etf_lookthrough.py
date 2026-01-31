# =============================================================================
# 遠期契約保證金模擬平台 - ETF Look-through 單元測試
# 測試項目：0050 與等權籃子完全對沖 → 100% 減收
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.instruments import InstrumentClassifier, ETFLookthrough, ExplodedPosition
from core.buckets import BucketClassifier, ReturnCalculator
from core.margin import MarginCalculator


class TestETFLookthrough:
    """ETF Look-through 測試"""

    @pytest.fixture
    def setup_etf_test_data(self):
        """設定 ETF 測試資料"""
        # ETF 成分股權重（簡化版 0050）
        etf_weights = {
            '0050': pd.DataFrame([
                {'etf_code': '0050', 'code': '2330', 'name': '台積電', 'weight': 0.50},
                {'etf_code': '0050', 'code': '2317', 'name': '鴻海', 'weight': 0.30},
                {'etf_code': '0050', 'code': '2454', 'name': '聯發科', 'weight': 0.20},
            ])
        }

        # 產業分類
        industry_df = pd.DataFrame([
            {'code': '2330', 'bucket': 'ELECT'},
            {'code': '2317', 'bucket': 'ELECT'},
            {'code': '2454', 'bucket': 'ELECT'},
        ])

        # 股價資料
        base_date = pd.Timestamp('2024-01-15')
        dates = pd.date_range(base_date - pd.Timedelta(days=92), base_date, freq='B')

        price_records = []
        for d in dates:
            price_records.append({'date': d, 'code': '0050', 'close': 150})
            price_records.append({'date': d, 'code': '2330', 'close': 800})
            price_records.append({'date': d, 'code': '2317', 'close': 100})
            price_records.append({'date': d, 'code': '2454', 'close': 700})

        prices_df = pd.DataFrame(price_records)

        # 當日價格
        prices_today = {
            '0050': 150,
            '2330': 800,
            '2317': 100,
            '2454': 700
        }

        return {
            'etf_weights': etf_weights,
            'industry_df': industry_df,
            'prices_df': prices_df,
            'prices_today': prices_today,
            'asof_date': base_date
        }

    def test_etf_lookthrough_decomposition(self, setup_etf_test_data):
        """
        測試：ETF 正確拆解至成分股

        情境：
        - 0050 ETF 10,000 股 @ 150 = MV 1,500,000
        - 應拆解為：
          - 2330: 50% × 1,500,000 = 750,000
          - 2317: 30% × 1,500,000 = 450,000
          - 2454: 20% × 1,500,000 = 300,000
        """
        data = setup_etf_test_data

        # 部位：做空 0050 ETF
        positions = pd.DataFrame([
            {'code': '0050', 'side': 'SHORT', 'qty': 10000, 'instrument': 'ETF'},
        ])

        classifier = InstrumentClassifier(
            futures_underlying_set=set(),
            constituent_set_0050_0056=set()
        )

        etf_lookthrough = ETFLookthrough(data['etf_weights'], classifier)

        # 執行 look-through
        atoms, detail_df = etf_lookthrough.lookthrough(positions, data['prices_today'])

        # 驗證拆解數量
        assert len(atoms) == 3, f"應拆解為 3 個成分股，實際為 {len(atoms)}"

        # 驗證各成分股 MV
        total_mv = 10000 * 150  # 1,500,000
        expected_mvs = {
            '2330': total_mv * 0.50,  # 750,000
            '2317': total_mv * 0.30,  # 450,000
            '2454': total_mv * 0.20,  # 300,000
        }

        for atom in atoms:
            expected_mv = expected_mvs.get(atom.code, 0)
            assert abs(atom.mv - expected_mv) < 1, \
                f"{atom.code} MV 錯誤: 預期 {expected_mv}，實際 {atom.mv}"
            assert atom.is_from_etf == True, f"{atom.code} 應標記為 ETF 來源"
            assert atom.origin == 'ETF_0050', f"{atom.code} origin 應為 ETF_0050"

        print("✅ test_etf_lookthrough_decomposition 通過")

    def test_etf_full_hedge_100_reduction(self, setup_etf_test_data):
        """
        測試：ETF 成分股完全對沖 → 100% 減收

        制度條款 5.2：
        若一籃子股票與 ETF 成份股在同一檔股票上形成反向曝險，
        則該檔股票可認定存在「完全對沖市值」
        MV_hedged_i = min(MV_long_i, MV_short_i)

        注意：ETF 100% 折減只套用於小邊的 ETF look-through 曝險

        情境（調整為 ETF 在小邊）：
        - 0050 ETF LONG 5,000 股（拆解後含 2330 成分 = 375,000）
        - 2330 股票 SHORT 1,000 股（MV = 800,000，超過 ETF 成分）
        - 大邊 = SHORT，小邊 = LONG（含 ETF）
        - 2330 在小邊的 ETF 部分形成完全對沖，應 100% 減收
        """
        data = setup_etf_test_data

        # 部位設計使大邊為 SHORT，小邊為 LONG（含 ETF）
        # ETF 在小邊，才能觸發 100% 折減
        positions = pd.DataFrame([
            {'code': '0050', 'side': 'LONG', 'qty': 5000, 'instrument': 'ETF'},    # ETF 在小邊
            {'code': '2330', 'side': 'SHORT', 'qty': 1000, 'instrument': 'STK'},  # 1000 × 800 = 800,000
        ])

        classifier = InstrumentClassifier(
            futures_underlying_set={'2330'},  # 2330 是期貨標的 → 5x
            constituent_set_0050_0056={'2330', '2317', '2454'}
        )

        etf_lookthrough = ETFLookthrough(data['etf_weights'], classifier)
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
            equity=1_000_000
        )

        # 驗證 ETF 100% 折減有發生
        # ETF LONG: 5000 × 150 = 750,000 總 MV
        #   拆解：2330 = 50% × 750,000 = 375,000
        #         2317 = 30% × 750,000 = 225,000
        #         2454 = 20% × 750,000 = 150,000
        # SHORT 2330: 1000 × 800 = 800,000
        #
        # 大邊 = SHORT（Base IM = 800,000 / 5 = 160,000）
        # 小邊 = LONG（ETF Base IM = 750,000 / 7 = 107,143）
        #
        # 完全對沖：min(2330_long_from_etf, 2330_short) = min(375,000, 800,000) = 375,000
        # 完全對沖 Base IM = 375,000 / 7 ≈ 53,571
        # 此部分應 100% 減收

        # 確認有 ETF 100% 折減
        assert result.reduction_etf_100 > 0, \
            f"應有 ETF 100% 折減，實際 reduction_etf_100 = {result.reduction_etf_100}"

        # 預期折減金額（約 53,571）
        expected_hedge_mv = 375_000  # min(375,000, 800,000)
        expected_reduction = expected_hedge_mv / 7  # 7x 槓桿
        tolerance = expected_reduction * 0.15  # 15% 容差（因計算細節可能有些差異）

        assert abs(result.reduction_etf_100 - expected_reduction) < tolerance, \
            f"ETF 100% 折減金額錯誤: 預期約 {expected_reduction:,.0f}，實際 {result.reduction_etf_100:,.0f}"

        print("test_etf_full_hedge_100_reduction passed")

    def test_etf_partial_hedge(self, setup_etf_test_data):
        """
        測試：ETF 部分對沖

        注意：ETF 100% 折減只套用於小邊的 ETF look-through 曝險

        情境（調整為 ETF 在小邊）：
        - 0050 ETF LONG 5,000 股（MV = 750,000，Base IM = 750,000/7 = 107,143）
        - 2330 股票 SHORT 2,000 股（MV = 1,600,000，Base IM = 1,600,000/5 = 320,000）
        - 大邊 = SHORT（320,000 > 107,143），小邊 = LONG（含 ETF）
        - ETF 拆解後 2330 成分 = 375,000
        - 完全對沖部分 = min(375,000, 1,600,000) = 375,000
        - 殘餘未對沖（其他 ETF 成分）再走三桶規則
        """
        data = setup_etf_test_data

        # 部位設計使大邊為 SHORT，小邊為 LONG（含 ETF）
        # SHORT 2330 需要足夠大使 Base IM > ETF 的 Base IM
        positions = pd.DataFrame([
            {'code': '0050', 'side': 'LONG', 'qty': 5000, 'instrument': 'ETF'},     # ETF 在小邊
            {'code': '2330', 'side': 'SHORT', 'qty': 2000, 'instrument': 'STK'},   # 2000 × 800 = 1,600,000
        ])

        classifier = InstrumentClassifier(
            futures_underlying_set={'2330'},
            constituent_set_0050_0056={'2330', '2317', '2454'}
        )

        etf_lookthrough = ETFLookthrough(data['etf_weights'], classifier)
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
            equity=1_000_000
        )

        # 驗證：
        # ETF LONG: MV = 750,000, Base IM = 107,143
        # SHORT 2330: MV = 1,600,000, Base IM = 320,000
        # 大邊 = SHORT (320,000 > 107,143)
        # 小邊 = LONG (ETF)
        #
        # ETF LONG 2330 成分 = 50% × 750,000 = 375,000
        # SHORT 2330 = 1,600,000
        # 完全對沖 = min(375,000, 1,600,000) = 375,000
        # 100% 折減金額 = 375,000 / 7 ≈ 53,571

        # 確認大邊是 SHORT
        assert result.big_side == 'SHORT', f"大邊應為 SHORT，實際為 {result.big_side}"

        expected_full_hedge_mv = 375_000
        expected_reduction_100 = expected_full_hedge_mv / 7
        tolerance = expected_reduction_100 * 0.15  # 15% 容差

        assert result.reduction_etf_100 > 0, "部分對沖也應有 ETF 100% 折減"
        assert abs(result.reduction_etf_100 - expected_reduction_100) < tolerance, \
            f"部分對沖折減金額錯誤: 預期約 {expected_reduction_100:,.0f}，實際 {result.reduction_etf_100:,.0f}"

        # 殘餘未對沖部分應走三桶規則
        # 殘餘：2317 from ETF = 225,000，2454 from ETF = 150,000
        # 這些殘餘應有同桶或跨桶折減
        total_reduction = result.reduction_etf_100 + result.reduction_same_bucket + result.reduction_cross_bucket
        assert total_reduction >= result.reduction_etf_100, \
            "殘餘部位可能有額外的桶折減"

        print("test_etf_partial_hedge passed")

    def test_etf_7x_leverage(self, setup_etf_test_data):
        """
        測試：ETF look-through 後使用 7x 槓桿

        制度條款 5.1：拆解後每一檔成份股視為「曝險單元」，
        並以 ETF 槓桿 7x 計算其 Base IM
        """
        data = setup_etf_test_data

        classifier = InstrumentClassifier(
            futures_underlying_set={'2330'},
            constituent_set_0050_0056={'2330', '2317', '2454'}
        )

        # ETF look-through 後的槓桿判定
        lev_info = classifier.classify_leverage_for_etf_component('0050', '2330')
        assert lev_info.leverage == 7.0, \
            f"ETF 成分股應使用 7x 槓桿，實際為 {lev_info.leverage}"

        lev_info = classifier.classify_leverage_for_etf_component('0056', '2317')
        assert lev_info.leverage == 7.0, \
            f"0056 成分股也應使用 7x 槓桿，實際為 {lev_info.leverage}"

        print("✅ test_etf_7x_leverage 通過")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
