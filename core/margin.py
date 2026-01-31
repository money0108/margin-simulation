# =============================================================================
# 遠期契約保證金模擬平台 - 保證金計算模組
# 功能：Base IM/大小邊/折減/IM_today/MM_today/追繳判定
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .instruments import InstrumentClassifier, ETFLookthrough, ExplodedPosition, LeverageInfo
from .buckets import BucketClassifier, ReturnCalculator, BucketAnalyzer, Bucket, BucketMetrics


@dataclass
class MarginResult:
    """保證金計算結果"""
    # 核心指標
    asof_date: pd.Timestamp
    im_today: float  # 當日 IM
    mm_today: float  # 維持保證金（追繳後可能重置）
    equity: float  # 權益（追繳後可能重置）
    margin_call: bool  # 是否觸發追繳
    required_deposit: float  # 追繳金額

    # 大小邊資訊
    big_side: str  # LONG / SHORT
    small_side: str
    long_mv: float
    short_mv: float
    base_im_long: float
    base_im_short: float
    im_big: float
    im_small_before: float  # 折減前
    im_small_after: float  # 折減後

    # 折減明細
    reduction_etf_100: float  # ETF 完全對沖 100% 減收
    reduction_same_bucket: float  # 同桶折減
    reduction_cross_bucket: float  # 跨桶折減
    total_reduction: float  # 總折減金額

    # 槓桿指標
    gross_leverage: float  # (Long MV + Short MV) / IM_today（有折減）
    raw_leverage: float  # (Long MV + Short MV) / (Base_IM_long + Base_IM_short)（無折減）

    # 追繳判定用（顯示追繳前狀態）- 有預設值
    equity_before_call: float = 0.0  # 追繳前權益
    mm_at_call: float = 0.0  # 追繳判定時的 MM

    # 明細表格
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    side_totals_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    bucket_hedge_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    small_side_detail_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    atom_detail_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    reduction_breakdown_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    hedge_pairing_df: pd.DataFrame = field(default_factory=pd.DataFrame)  # 多空配對明細


class MarginCalculator:
    """
    保證金計算器

    核心制度規則：
    1. Base IM = MV ÷ Leverage（逐檔計算，不可用平均槓桿）
    2. 以分邊 Base IM 判定大小邊
    3. 折減只套用小邊
    4. 同桶折減：3M 報酬差 >= 10% → 50%，否則 20%
    5. 跨桶折減：一律 20%
    6. ETF 完全對沖：100% 減收
    7. IM_today = IM_big + IM_small_after_discount
    8. MM_today = 0.70 × IM_today
    9. 追繳：Equity < MM → 回補至 IM_today
    """

    def __init__(self,
                 classifier: InstrumentClassifier,
                 etf_lookthrough: ETFLookthrough,
                 bucket_classifier: BucketClassifier,
                 return_calculator: ReturnCalculator,
                 mm_ratio: float = 0.70):
        """
        初始化保證金計算器

        Args:
            classifier: 標的分類器
            etf_lookthrough: ETF look-through 處理器
            bucket_classifier: 產業桶分類器
            return_calculator: 3M 報酬計算器
            mm_ratio: 維持保證金比例（預設 70%）
        """
        self.classifier = classifier
        self.etf_lookthrough = etf_lookthrough
        self.bucket_classifier = bucket_classifier
        self.return_calculator = return_calculator
        self.mm_ratio = mm_ratio

        # 產業桶分析器
        self.bucket_analyzer = BucketAnalyzer(bucket_classifier, return_calculator)

    def _build_hedge_pairing(self,
                             atoms_df: pd.DataFrame,
                             big_side: str,
                             small_side: str,
                             reduction_rates: Dict) -> pd.DataFrame:
        """
        建立多空配對明細表

        顯示每個標的的多空配對狀況與減收明細
        """
        # 按標的代碼彙總多空 MV
        long_by_code = atoms_df[atoms_df['side'] == 'LONG'].groupby('code').agg({
            'mv': 'sum',
            'base_im': 'sum',
            'bucket': 'first',
            'is_from_etf': 'any'
        }).rename(columns={'mv': 'long_mv', 'base_im': 'long_base_im'})

        short_by_code = atoms_df[atoms_df['side'] == 'SHORT'].groupby('code').agg({
            'mv': 'sum',
            'base_im': 'sum',
            'bucket': 'first',
            'is_from_etf': 'any'
        }).rename(columns={'mv': 'short_mv', 'base_im': 'short_base_im'})

        # 合併多空
        pairing = long_by_code.join(short_by_code, how='outer', lsuffix='_l', rsuffix='_s').fillna(0)

        # 整理欄位
        pairing['bucket'] = pairing['bucket_l'].combine_first(pairing['bucket_s'])
        pairing['is_from_etf'] = pairing['is_from_etf_l'] | pairing['is_from_etf_s']

        # 清理
        pairing = pairing.drop(columns=['bucket_l', 'bucket_s', 'is_from_etf_l', 'is_from_etf_s'], errors='ignore')

        # 計算配對金額
        pairing['hedged_mv'] = pairing[['long_mv', 'short_mv']].min(axis=1)

        # 判斷減收類型與金額
        records = []
        for code, row in pairing.iterrows():
            long_mv = row.get('long_mv', 0)
            short_mv = row.get('short_mv', 0)
            hedged_mv = row.get('hedged_mv', 0)
            bucket = row.get('bucket', '非金電')
            is_from_etf = row.get('is_from_etf', False)

            # 判斷小邊 MV
            if small_side == 'LONG':
                small_mv = long_mv
                small_base_im = row.get('long_base_im', 0)
            else:
                small_mv = short_mv
                small_base_im = row.get('short_base_im', 0)

            # 減收類型判定
            reduction_type = ''
            reduction_rate = 0.0
            reduction_im = 0.0

            if hedged_mv > 0:
                if is_from_etf:
                    # ETF 完全對沖
                    reduction_type = 'ETF完全對沖'
                    reduction_rate = 1.0
                    reduction_im = small_base_im * min(1.0, hedged_mv / small_mv) if small_mv > 0 else 0
                else:
                    # 同桶對沖
                    bucket_enum = Bucket.ELECT if bucket == '電子' else (Bucket.FIN if bucket == '金融' else Bucket.NON)
                    rate_result = reduction_rates.get(bucket_enum)
                    if rate_result and rate_result.same_bucket_rate > 0:
                        reduction_type = '同桶對沖'
                        reduction_rate = rate_result.same_bucket_rate
                        hedge_ratio = min(1.0, hedged_mv / small_mv) if small_mv > 0 else 0
                        reduction_im = small_base_im * reduction_rate * hedge_ratio

            records.append({
                '代碼': code,
                '產業桶': bucket,
                '多方MV': round(long_mv),
                '空方MV': round(short_mv),
                '配對MV': round(hedged_mv),
                '來源': 'ETF拆解' if is_from_etf else '直接持有',
                '減收類型': reduction_type if reduction_type else '無配對',
                '減收率': f'{reduction_rate:.0%}' if reduction_rate > 0 else '-',
                '減收IM': round(reduction_im),
            })

        result_df = pd.DataFrame(records)

        # 只顯示有配對或有曝險的標的
        result_df = result_df[(result_df['多方MV'] > 0) | (result_df['空方MV'] > 0)]

        # 按減收金額排序
        result_df = result_df.sort_values('減收IM', ascending=False)

        return result_df

    def _empty_result(self, asof_date: pd.Timestamp, equity: float) -> MarginResult:
        """返回空部位結果"""
        return MarginResult(
            asof_date=asof_date, im_today=0, mm_today=0, equity=equity,
            margin_call=False, required_deposit=0, big_side='LONG', small_side='SHORT',
            long_mv=0, short_mv=0, base_im_long=0, base_im_short=0,
            im_big=0, im_small_before=0, im_small_after=0,
            reduction_etf_100=0, reduction_same_bucket=0, reduction_cross_bucket=0,
            total_reduction=0, gross_leverage=0, raw_leverage=0,
            summary_df=pd.DataFrame(), side_totals_df=pd.DataFrame(),
            bucket_hedge_df=pd.DataFrame(), small_side_detail_df=pd.DataFrame(),
            atom_detail_df=pd.DataFrame(), reduction_breakdown_df=pd.DataFrame(),
            hedge_pairing_df=pd.DataFrame()
        )

    def calculate(self,
                 positions: pd.DataFrame,
                 prices_today: Dict[str, float],
                 asof_date: pd.Timestamp,
                 equity: Optional[float] = None) -> MarginResult:
        """
        執行完整保證金計算

        Args:
            positions: 部位 DataFrame
            prices_today: 當日價格字典
            asof_date: 估值日
            equity: 權益（若未提供，以 IM × 1.2 估算）

        Returns:
            MarginResult
        """
        # =====================================================================
        # Step 1: ETF Look-through 拆解
        # 制度條款 5.1：把 ETF 名目市值依成份股權重拆解
        # =====================================================================
        atoms, lookthrough_detail = self.etf_lookthrough.lookthrough(
            positions, prices_today
        )

        # =====================================================================
        # Step 2: 計算各曝險單元的 Base IM（向量化）
        # 制度條款 3.1：Base IM = MV ÷ Leverage（逐檔計算）
        # =====================================================================
        # 快速建立 DataFrame
        atom_records = [{
            'origin': a.origin, 'parent': a.parent, 'code': a.code,
            'side': a.side, 'mv': a.mv, 'instrument': a.instrument,
            'is_from_etf': a.is_from_etf, 'weight': a.weight
        } for a in atoms]

        atoms_df = pd.DataFrame(atom_records)

        if len(atoms_df) == 0:
            # 空部位處理
            return self._empty_result(asof_date, equity or 0.0)

        # 批次計算槓桿（使用 to_dict 避免 iterrows）
        leverages = []
        leverage_sources = []
        for row in atoms_df.to_dict('records'):
            if row['is_from_etf']:
                lev_info = self.classifier.classify_leverage_for_etf_component(
                    row['parent'], row['code']
                )
            else:
                lev_info = self.classifier.classify_leverage(
                    row['code'], row['instrument']
                )
            leverages.append(lev_info.leverage)
            leverage_sources.append(lev_info.source)

        atoms_df['leverage'] = leverages
        atoms_df['leverage_source'] = leverage_sources

        # 批次計算產業桶（使用 map 加速）
        atoms_df['bucket'] = atoms_df['code'].map(
            lambda c: self.bucket_classifier.classify(c).value
        )

        # 向量化計算 Base IM
        atoms_df['base_im'] = atoms_df['mv'] / atoms_df['leverage']

        # =====================================================================
        # Step 3: 大小邊判定
        # 制度條款 3.2：以分邊 Base IM 判定
        # =====================================================================
        base_im_by_side = atoms_df.groupby('side')['base_im'].sum()
        base_im_long = base_im_by_side.get('LONG', 0.0)
        base_im_short = base_im_by_side.get('SHORT', 0.0)

        # 制度條款 3.2：大小邊判定
        if base_im_long >= base_im_short:
            big_side = 'LONG'
            small_side = 'SHORT'
            im_big = base_im_long
            im_small_before = base_im_short
        else:
            big_side = 'SHORT'
            small_side = 'LONG'
            im_big = base_im_short
            im_small_before = base_im_long

        # MV 彙總
        mv_by_side = atoms_df.groupby('side')['mv'].sum()
        long_mv = mv_by_side.get('LONG', 0.0)
        short_mv = mv_by_side.get('SHORT', 0.0)

        # =====================================================================
        # Step 4: ETF 完全對沖 100% 減收
        # 制度條款 5.2：MV_hedged_i = min(MV_long_i, MV_short_i)
        # =====================================================================
        # 計算完全對沖金額
        hedged_amounts = self.etf_lookthrough.compute_full_hedge_amounts(
            [ExplodedPosition(**{k: r[k] for k in ['origin', 'parent', 'code', 'side', 'mv', 'instrument', 'is_from_etf', 'weight']})
             for r in atom_records],
            big_side, small_side
        )

        # 套用 100% 折減（只對小邊 ETF look-through）
        atoms_df['disc100_mv'] = 0.0
        atoms_df['disc100_im'] = 0.0

        for code, hedged_mv in hedged_amounts.items():
            # 找出小邊中該 code 的 ETF look-through 曝險
            mask = (
                (atoms_df['side'] == small_side) &
                (atoms_df['is_from_etf'] == True) &
                (atoms_df['code'] == code)
            )
            if not mask.any():
                continue

            # 計算對沖比例
            total_etf_mv = atoms_df.loc[mask, 'mv'].sum()
            if total_etf_mv > 0:
                hedge_ratio = min(hedged_mv, total_etf_mv) / total_etf_mv
                atoms_df.loc[mask, 'disc100_mv'] = atoms_df.loc[mask, 'mv'] * hedge_ratio
                atoms_df.loc[mask, 'disc100_im'] = atoms_df.loc[mask, 'disc100_mv'] / atoms_df.loc[mask, 'leverage']

        # 計算 100% 折減後的殘餘
        atoms_df['mv_after100'] = (atoms_df['mv'] - atoms_df['disc100_mv']).clip(lower=0)
        atoms_df['base_im_after100'] = atoms_df['mv_after100'] / atoms_df['leverage']

        reduction_etf_100 = atoms_df[atoms_df['side'] == small_side]['disc100_im'].sum()

        # =====================================================================
        # Step 5: 產業桶分析與折減率計算
        # 制度條款 4.3/4.4：同桶 50%/20%、跨桶 20%
        # =====================================================================
        # 快速建立殘餘 atoms 列表（使用 to_dict 代替 iterrows）
        residual_atoms = [
            ExplodedPosition(
                origin=row['origin'],
                parent=row['parent'],
                code=row['code'],
                side=row['side'],
                mv=row['mv_after100'],
                instrument=row['instrument'],
                is_from_etf=row['is_from_etf'],
                weight=row['weight']
            )
            for row in atoms_df.to_dict('records')
        ]

        # 分析各桶指標
        bucket_metrics = self.bucket_analyzer.analyze(residual_atoms, asof_date)

        # 計算折減率
        reduction_rates = self.bucket_analyzer.compute_reduction_rates(
            bucket_metrics, small_side
        )

        # =====================================================================
        # Step 6: 套用折減（只對小邊）
        # 制度條款 4.5：折減只作用於小邊
        # =====================================================================
        atoms_df['disc_same_im'] = 0.0
        atoms_df['disc_same_rate'] = 0.0
        atoms_df['disc_cross_im'] = 0.0
        atoms_df['disc_cross_rate'] = 0.0

        # 同桶折減
        for bucket in Bucket:
            rate_result = reduction_rates[bucket]
            if rate_result.same_bucket_rate <= 0:
                continue

            # 可對沖比例內的部分給予同桶折減
            mask = (
                (atoms_df['side'] == small_side) &
                (atoms_df['bucket'] == bucket.value)
            )
            if not mask.any():
                continue

            # 有效折減率 = 同桶折減率 × 可對沖比例
            effective_rate = rate_result.same_bucket_rate * rate_result.eligible_hedged_ratio
            atoms_df.loc[mask, 'disc_same_rate'] = effective_rate
            atoms_df.loc[mask, 'disc_same_im'] = atoms_df.loc[mask, 'base_im_after100'] * effective_rate

        # 跨桶折減（對小邊未被同桶覆蓋的部分）
        # 計算大邊各桶 MV
        big_side_mv_by_bucket = atoms_df[atoms_df['side'] == big_side].groupby('bucket')['mv_after100'].sum().to_dict()
        total_big_mv = sum(big_side_mv_by_bucket.values())

        for bucket in Bucket:
            rate_result = reduction_rates[bucket]
            residual_ratio = max(0.0, 1.0 - rate_result.eligible_hedged_ratio)
            if residual_ratio <= 0:
                continue

            mask = (
                (atoms_df['side'] == small_side) &
                (atoms_df['bucket'] == bucket.value)
            )
            if not mask.any():
                continue

            # 跨桶對沖：其他桶的大邊 MV
            other_bucket_mv = total_big_mv - big_side_mv_by_bucket.get(bucket.value, 0)
            small_bucket_mv = atoms_df.loc[mask, 'mv_after100'].sum()
            small_residual_mv = small_bucket_mv * residual_ratio

            if small_residual_mv <= 0 or other_bucket_mv <= 0:
                continue

            # 跨桶覆蓋比例
            cross_ratio = min(1.0, other_bucket_mv / small_residual_mv)
            effective_cross_rate = 0.20 * cross_ratio * residual_ratio

            atoms_df.loc[mask, 'disc_cross_rate'] = effective_cross_rate
            atoms_df.loc[mask, 'disc_cross_im'] = atoms_df.loc[mask, 'base_im_after100'] * effective_cross_rate

        # =====================================================================
        # Step 7: 彙總計算 IM_today
        # 制度條款 4.6：IM_today = IM_big + IM_small_after
        # =====================================================================
        atoms_df['total_disc_im'] = atoms_df['disc100_im'] + atoms_df['disc_same_im'] + atoms_df['disc_cross_im']
        atoms_df['im_after_disc'] = atoms_df['base_im'] - atoms_df['total_disc_im']

        # 小邊折減彙總
        small_side_df = atoms_df[atoms_df['side'] == small_side]
        reduction_same_bucket = small_side_df['disc_same_im'].sum()
        reduction_cross_bucket = small_side_df['disc_cross_im'].sum()
        total_reduction = reduction_etf_100 + reduction_same_bucket + reduction_cross_bucket

        im_small_after = im_small_before - total_reduction

        # 制度條款 4.6：IM_today
        im_today = im_big + im_small_after

        # =====================================================================
        # Step 8: MM 與追繳判定
        # 制度條款 6：MM = 70% × IM_today
        # =====================================================================
        mm_today = self.mm_ratio * im_today

        # 權益（由外部提供，若未提供則暫設為 0，由引擎後續更新）
        if equity is None:
            equity = 0.0

        # 制度條款 6.2：追繳判定（若 equity=0 表示尚未設定，暫不判定追繳）
        if equity > 0:
            margin_call = equity < mm_today
            required_deposit = max(0, im_today - equity) if margin_call else 0.0
        else:
            margin_call = False
            required_deposit = 0.0

        # =====================================================================
        # Step 9: 槓桿指標
        # =====================================================================
        gross_mv = long_mv + short_mv
        base_im_total = base_im_long + base_im_short  # 無折減的總 Base IM

        gross_leverage = gross_mv / im_today if im_today > 0 else 0.0
        raw_leverage = gross_mv / base_im_total if base_im_total > 0 else 0.0  # 無折減槓桿

        # =====================================================================
        # Step 10: 建立報表
        # =====================================================================
        # 摘要表
        summary_df = pd.DataFrame([{
            '估值日': asof_date.strftime('%Y-%m-%d'),
            '大邊': big_side,
            '小邊': small_side,
            '多方MV': round(long_mv),
            '空方MV': round(short_mv),
            '多方Base_IM': round(base_im_long),
            '空方Base_IM': round(base_im_short),
            'IM_big': round(im_big),
            'IM_small(折減前)': round(im_small_before),
            'ETF_100%折減': round(reduction_etf_100),
            '同桶折減': round(reduction_same_bucket),
            '跨桶折減': round(reduction_cross_bucket),
            '總折減': round(total_reduction),
            'IM_small(折減後)': round(im_small_after),
            'IM_today': round(im_today),
            'MM_today(70%)': round(mm_today),
            '權益': round(equity),
            '追繳觸發': '是' if margin_call else '否',
            '追繳金額': round(required_deposit),
            'Gross槓桿': round(gross_leverage, 2),
            '無折減槓桿': round(raw_leverage, 2),
        }])

        # 分邊彙總表
        side_totals_df = atoms_df.groupby('side').agg({
            'mv': 'sum',
            'base_im': 'sum',
            'disc100_im': 'sum',
            'disc_same_im': 'sum',
            'disc_cross_im': 'sum',
            'total_disc_im': 'sum',
            'im_after_disc': 'sum'
        }).reset_index()
        side_totals_df.columns = ['邊', 'MV', 'Base_IM', 'ETF_100%折減', '同桶折減', '跨桶折減', '總折減', 'IM淨額']

        # 產業桶對沖表
        bucket_hedge_df = self.bucket_analyzer.get_bucket_summary(
            bucket_metrics, reduction_rates
        )

        # 小邊明細表
        small_side_detail_df = small_side_df.groupby(['code', 'bucket']).agg({
            'mv_after100': 'sum',
            'base_im_after100': 'sum',
            'disc100_im': 'sum',
            'disc_same_im': 'sum',
            'disc_cross_im': 'sum',
            'total_disc_im': 'sum'
        }).reset_index()
        small_side_detail_df.columns = ['代碼', '產業桶', 'MV', 'Base_IM', 'ETF折減', '同桶折減', '跨桶折減', '總折減']

        # 折減分解表
        reduction_breakdown_df = pd.DataFrame([
            {'折減類型': 'ETF完全對沖(100%)', '金額': round(reduction_etf_100), '說明': 'ETF成分股與對向曝險完全對沖'},
            {'折減類型': '同桶對沖', '金額': round(reduction_same_bucket), '說明': '同產業桶多空對沖(50%或20%)'},
            {'折減類型': '跨桶對沖', '金額': round(reduction_cross_bucket), '說明': '不同產業桶多空對沖(20%)'},
            {'折減類型': '合計', '金額': round(total_reduction), '說明': ''},
        ])

        # =====================================================================
        # 多空配對明細表
        # =====================================================================
        hedge_pairing_df = self._build_hedge_pairing(atoms_df, big_side, small_side, reduction_rates)

        return MarginResult(
            asof_date=asof_date,
            im_today=im_today,
            mm_today=mm_today,
            equity=equity,
            margin_call=margin_call,
            required_deposit=required_deposit,
            big_side=big_side,
            small_side=small_side,
            long_mv=long_mv,
            short_mv=short_mv,
            base_im_long=base_im_long,
            base_im_short=base_im_short,
            im_big=im_big,
            im_small_before=im_small_before,
            im_small_after=im_small_after,
            reduction_etf_100=reduction_etf_100,
            reduction_same_bucket=reduction_same_bucket,
            reduction_cross_bucket=reduction_cross_bucket,
            total_reduction=total_reduction,
            gross_leverage=gross_leverage,
            raw_leverage=raw_leverage,
            summary_df=summary_df,
            side_totals_df=side_totals_df,
            bucket_hedge_df=bucket_hedge_df,
            small_side_detail_df=small_side_detail_df,
            atom_detail_df=atoms_df,
            reduction_breakdown_df=reduction_breakdown_df,
            hedge_pairing_df=hedge_pairing_df,
        )


# =============================================================================
# 便捷函式
# =============================================================================

def compute_mv(positions: pd.DataFrame,
              prices: pd.DataFrame) -> pd.DataFrame:
    """
    計算逐日 MV

    Args:
        positions: 部位 DataFrame
        prices: 股價 DataFrame

    Returns:
        逐日 MV DataFrame（含 Long/Short 分邊彙總）
    """
    prices = prices.copy()
    prices['date'] = pd.to_datetime(prices['date'])

    # 取得所有交易日
    dates = sorted(prices['date'].unique())

    records = []
    for date in dates:
        date_prices = prices[prices['date'] == date].set_index('code')['close'].to_dict()

        long_mv = 0.0
        short_mv = 0.0

        for _, row in positions.iterrows():
            code = str(row['code']).strip()
            qty = float(row['qty'])
            side = row['side']

            price = date_prices.get(code, 0)
            mv = qty * price

            if side == 'LONG':
                long_mv += mv
            else:
                short_mv += mv

        records.append({
            'date': date,
            'long_mv': long_mv,
            'short_mv': short_mv,
            'gross_mv': long_mv + short_mv,
            'net_mv': abs(long_mv - short_mv)
        })

    return pd.DataFrame(records)


def compute_base_im(mv_df: pd.DataFrame) -> pd.DataFrame:
    """
    計算 Base IM（簡化版，使用平均槓桿）

    注意：完整計算應使用 MarginCalculator
    """
    df = mv_df.copy()
    # 假設平均槓桿 4x
    df['base_im_long'] = df['long_mv'] / 4.0
    df['base_im_short'] = df['short_mv'] / 4.0
    return df


def compute_im_today(reduced_df: pd.DataFrame) -> pd.Series:
    """
    計算 IM_today

    Args:
        reduced_df: 折減後的 DataFrame

    Returns:
        IM_today Series
    """
    return reduced_df['im_today']


def compute_mm_today(im_today: pd.Series, ratio: float = 0.70) -> pd.Series:
    """
    計算 MM_today

    Args:
        im_today: IM_today Series
        ratio: MM 比例（預設 70%）

    Returns:
        MM_today Series
    """
    return im_today * ratio


def margin_call(equity_series: pd.Series,
               mm_today: pd.Series,
               im_today: pd.Series) -> pd.DataFrame:
    """
    追繳判定

    Args:
        equity_series: 權益 Series
        mm_today: MM Series
        im_today: IM Series

    Returns:
        追繳狀態 DataFrame
    """
    df = pd.DataFrame({
        'equity': equity_series,
        'mm_today': mm_today,
        'im_today': im_today
    })

    df['margin_call'] = df['equity'] < df['mm_today']
    df['required_deposit'] = (df['im_today'] - df['equity']).clip(lower=0)
    df.loc[~df['margin_call'], 'required_deposit'] = 0

    return df


def apply_reductions(base_im_df: pd.DataFrame,
                    bucket_metrics: Dict,
                    rules: Dict) -> pd.DataFrame:
    """
    套用折減規則（簡化版）

    注意：完整計算應使用 MarginCalculator
    """
    df = base_im_df.copy()
    # 簡化：假設 20% 折減
    df['reduction'] = df['base_im_small'] * 0.20
    df['im_small_after'] = df['base_im_small'] - df['reduction']
    df['im_today'] = df['base_im_big'] + df['im_small_after']
    return df
