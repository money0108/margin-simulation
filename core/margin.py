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

    def _build_atom_templates(self,
                              positions: pd.DataFrame,
                              prices_today: Dict[str, float]) -> List[Dict]:
        """
        從部位建立可重複使用的 atom 模板（含 leverage/bucket），
        後續只需套用當日價格即可快速算出 MV。

        Args:
            positions: 部位 DataFrame
            prices_today: 建倉日價格（用於初次 lookthrough）

        Returns:
            list of dict，每筆含 origin, parent, code, side, qty_factor,
            price_lookup_code, instrument, is_from_etf, weight,
            leverage, leverage_source, bucket
        """
        # Step 1: ETF Look-through 拆解
        atoms, _ = self.etf_lookthrough.lookthrough(positions, prices_today)

        templates = []
        for a in atoms:
            # 計算 qty_factor：MV / price → 後續可用 qty_factor * new_price 算 MV
            price = prices_today.get(a.code, 0)
            qty_factor = a.mv / price if price > 0 else 0.0

            # 槓桿
            if a.is_from_etf:
                lev_info = self.classifier.classify_leverage_for_etf_component(
                    a.parent, a.code
                )
            else:
                lev_info = self.classifier.classify_leverage(
                    a.code, a.instrument
                )

            # 產業桶
            bucket = self.bucket_classifier.classify(a.code).value

            templates.append({
                'origin': a.origin,
                'parent': a.parent,
                'code': a.code,
                'side': a.side,
                'qty_factor': qty_factor,
                'price_lookup_code': a.code,
                'instrument': a.instrument,
                'is_from_etf': a.is_from_etf,
                'weight': a.weight,
                'leverage': lev_info.leverage,
                'leverage_source': lev_info.source,
                'bucket': bucket,
            })

        return templates

    def _build_fast_arrays(self, templates: List[Dict]) -> Dict[str, Any]:
        """
        從 atom templates 預建 numpy 陣列與索引結構，
        供 calculate_fast() 的全 numpy 路徑使用。

        Args:
            templates: _build_atom_templates() 的回傳結果

        Returns:
            dict 包含 numpy 陣列與索引
        """
        n = len(templates)
        if n == 0:
            return {}

        codes = [t['code'] for t in templates]
        qty_factors = np.array([t['qty_factor'] for t in templates], dtype=np.float64)
        leverages = np.array([t['leverage'] for t in templates], dtype=np.float64)
        is_from_etf = np.array([t['is_from_etf'] for t in templates], dtype=bool)
        weights = np.array([t['weight'] for t in templates], dtype=np.float64)

        sides = [t['side'] for t in templates]
        long_mask = np.array([s == 'LONG' for s in sides], dtype=bool)
        short_mask = ~long_mask

        # bucket masks
        bucket_values = [t['bucket'] for t in templates]
        bucket_masks: Dict[Bucket, np.ndarray] = {}
        for b in Bucket:
            bucket_masks[b] = np.array([bv == b.value for bv in bucket_values], dtype=bool)

        # code_indices_by_side: side -> code -> array of indices
        code_indices_by_side: Dict[str, Dict[str, np.ndarray]] = {'LONG': {}, 'SHORT': {}}
        # etf_code_indices_by_side: side -> code -> array of indices (ETF atoms only)
        etf_code_indices_by_side: Dict[str, Dict[str, np.ndarray]] = {'LONG': {}, 'SHORT': {}}

        for i, t in enumerate(templates):
            side = t['side']
            code = t['code']
            code_indices_by_side[side].setdefault(code, []).append(i)
            if t['is_from_etf']:
                etf_code_indices_by_side[side].setdefault(code, []).append(i)

        # Convert lists to numpy arrays
        for side in ('LONG', 'SHORT'):
            for code in code_indices_by_side[side]:
                code_indices_by_side[side][code] = np.array(code_indices_by_side[side][code], dtype=np.intp)
            for code in etf_code_indices_by_side[side]:
                etf_code_indices_by_side[side][code] = np.array(etf_code_indices_by_side[side][code], dtype=np.intp)

        # bucket_side_code_indices: (Bucket, side) -> list of (code, indices)
        bucket_side_code_indices: Dict[Tuple, List[Tuple[str, np.ndarray]]] = {}
        for b in Bucket:
            for side in ('LONG', 'SHORT'):
                side_mask = long_mask if side == 'LONG' else short_mask
                combined = side_mask & bucket_masks[b]
                # group by code within this bucket+side
                code_idx_map: Dict[str, List[int]] = {}
                for i in np.where(combined)[0]:
                    code_idx_map.setdefault(codes[i], []).append(i)
                bucket_side_code_indices[(b, side)] = [
                    (c, np.array(idxs, dtype=np.intp)) for c, idxs in code_idx_map.items()
                ]

        origins = [t['origin'] for t in templates]
        parents = [t['parent'] for t in templates]
        instruments = [t['instrument'] for t in templates]

        return {
            'n': n,
            'codes': codes,
            'qty_factors': qty_factors,
            'leverages': leverages,
            'is_from_etf': is_from_etf,
            'weights': weights,
            'long_mask': long_mask,
            'short_mask': short_mask,
            'bucket_masks': bucket_masks,
            'bucket_values': bucket_values,
            'code_indices_by_side': code_indices_by_side,
            'etf_code_indices_by_side': etf_code_indices_by_side,
            'bucket_side_code_indices': bucket_side_code_indices,
            'origins': origins,
            'parents': parents,
            'instruments': instruments,
            'sides': sides,
        }

    def calculate_fast(self,
                       templates: List[Dict],
                       prices_today: Dict[str, float],
                       asof_date: pd.Timestamp,
                       equity: Optional[float] = None,
                       build_reports: bool = False,
                       fast_arrays: Optional[Dict[str, Any]] = None) -> MarginResult:
        """
        使用預建模板的快速保證金計算。跳過 ETF lookthrough / leverage / bucket 步驟。

        當 fast_arrays 提供時使用全 numpy 路徑（不建 DataFrame），
        否則 fallback 到 DataFrame 路徑。

        Args:
            templates: _build_atom_templates() 的回傳結果
            prices_today: 當日價格字典
            asof_date: 估值日
            equity: 權益
            build_reports: 是否建立報表
            fast_arrays: _build_fast_arrays() 的回傳結果（可選）

        Returns:
            MarginResult
        """
        if not templates:
            return self._empty_result(asof_date, equity or 0.0)

        # ---- 若無 fast_arrays 或需要報表，fallback 到 DataFrame 路徑 ----
        if fast_arrays is None or build_reports:
            atom_records = []
            for t in templates:
                price = prices_today.get(t['price_lookup_code'], 0)
                mv = t['qty_factor'] * price
                atom_records.append({
                    'origin': t['origin'],
                    'parent': t['parent'],
                    'code': t['code'],
                    'side': t['side'],
                    'mv': mv,
                    'instrument': t['instrument'],
                    'is_from_etf': t['is_from_etf'],
                    'weight': t['weight'],
                    'leverage': t['leverage'],
                    'leverage_source': t['leverage_source'],
                    'bucket': t['bucket'],
                })
            atoms_df = pd.DataFrame(atom_records)
            atoms_df['base_im'] = atoms_df['mv'] / atoms_df['leverage']
            return self._compute_from_atoms_df(
                atoms_df, atom_records, asof_date, equity, build_reports
            )

        # ==================================================================
        # 全 numpy 快速路徑（Phase 4）
        # ==================================================================
        fa = fast_arrays
        n = fa['n']
        codes = fa['codes']
        qty_factors = fa['qty_factors']
        leverages = fa['leverages']
        is_from_etf_arr = fa['is_from_etf']
        weights_arr = fa['weights']
        long_mask = fa['long_mask']
        short_mask = fa['short_mask']
        bucket_masks = fa['bucket_masks']
        code_indices_by_side = fa['code_indices_by_side']
        etf_code_indices_by_side = fa['etf_code_indices_by_side']
        bucket_side_code_indices = fa['bucket_side_code_indices']

        # Step 1: 向量化計算 MV
        prices_arr = np.array([prices_today.get(c, 0) for c in codes], dtype=np.float64)
        mv = qty_factors * prices_arr

        # Step 2: 向量化計算 Base IM
        base_im = mv / leverages

        # Step 3: 大小邊判定（numpy mask sum 取代 groupby）
        base_im_long = base_im[long_mask].sum()
        base_im_short = base_im[short_mask].sum()

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

        long_mv = mv[long_mask].sum()
        short_mv = mv[short_mask].sum()

        big_mask = long_mask if big_side == 'LONG' else short_mask
        small_mask_side = short_mask if big_side == 'LONG' else long_mask

        # Step 4: ETF 完全對沖 100% 減收（內聯，不建 ExplodedPosition）
        disc100_mv = np.zeros(n, dtype=np.float64)
        disc100_im = np.zeros(n, dtype=np.float64)

        # 彙總大邊各 code 的 MV
        big_side_code_mv: Dict[str, float] = {}
        for code, idx in code_indices_by_side[big_side].items():
            big_side_code_mv[code] = mv[idx].sum()

        # 小邊 ETF atoms 的完全對沖
        small_etf_indices = etf_code_indices_by_side[small_side]
        for code, idx in small_etf_indices.items():
            big_mv_for_code = big_side_code_mv.get(code, 0.0)
            if big_mv_for_code <= 0:
                continue
            total_etf_mv = mv[idx].sum()
            if total_etf_mv <= 0:
                continue
            hedged_mv = min(big_mv_for_code, total_etf_mv)
            hedge_ratio = hedged_mv / total_etf_mv
            disc100_mv[idx] = mv[idx] * hedge_ratio
            disc100_im[idx] = disc100_mv[idx] / leverages[idx]

        mv_after100 = np.maximum(mv - disc100_mv, 0.0)
        base_im_after100 = mv_after100 / leverages

        reduction_etf_100 = disc100_im[small_mask_side].sum()

        # Step 5: 產業桶分析（內聯 bucket_analyzer.analyze + compute_reduction_rates）
        # 5a: 彙總各桶的 MV 與 code-mv 列表（用 mv_after100 作 residual）
        bucket_metrics_data: Dict[Bucket, Dict] = {}
        for b in Bucket:
            bm = bucket_masks[b]
            b_long = bm & long_mask
            b_short = bm & short_mask
            mv_long_b = mv_after100[b_long].sum()
            mv_short_b = mv_after100[b_short].sum()

            # 收集 code-mv pairs for weighted return
            long_code_mv = []
            for code, idx in bucket_side_code_indices.get((b, 'LONG'), []):
                code_mv_sum = mv_after100[idx].sum()
                if code_mv_sum > 0:
                    long_code_mv.append((code, code_mv_sum))

            short_code_mv = []
            for code, idx in bucket_side_code_indices.get((b, 'SHORT'), []):
                code_mv_sum = mv_after100[idx].sum()
                if code_mv_sum > 0:
                    short_code_mv.append((code, code_mv_sum))

            bucket_metrics_data[b] = {
                'mv_long': mv_long_b,
                'mv_short': mv_short_b,
                'long_code_mv': long_code_mv,
                'short_code_mv': short_code_mv,
            }

        # 5b: 計算各桶的加權報酬率與折減率
        reduction_rates_data: Dict[Bucket, Dict] = {}
        for b in Bucket:
            bmd = bucket_metrics_data[b]
            mv_long_b = bmd['mv_long']
            mv_short_b = bmd['mv_short']

            # 加權報酬
            r_3m_long = None
            r_3m_short = None
            if bmd['long_code_mv']:
                r_long, _, _ = self.return_calculator.compute_weighted_return(
                    bmd['long_code_mv'], asof_date
                )
                r_3m_long = r_long
            if bmd['short_code_mv']:
                r_short, _, _ = self.return_calculator.compute_weighted_return(
                    bmd['short_code_mv'], asof_date
                )
                r_3m_short = r_short

            # 可對沖比例
            if small_side == 'LONG':
                small_mv_b = mv_long_b
                big_mv_b = mv_short_b
                r_small = r_3m_long
                r_big = r_3m_short
            else:
                small_mv_b = mv_short_b
                big_mv_b = mv_long_b
                r_small = r_3m_short
                r_big = r_3m_long

            if small_mv_b > 0 and big_mv_b > 0:
                eligible_hedged_ratio = min(big_mv_b, small_mv_b) / small_mv_b
            else:
                eligible_hedged_ratio = 0.0

            # 同桶折減率
            same_bucket_rate = 0.0
            if mv_long_b > 0 and mv_short_b > 0:
                if r_big is not None and r_small is not None:
                    r_diff = abs(r_big - r_small)
                    same_bucket_rate = 0.50 if r_diff >= 0.10 else 0.20
                else:
                    same_bucket_rate = 0.20
            else:
                same_bucket_rate = 0.0

            reduction_rates_data[b] = {
                'same_bucket_rate': same_bucket_rate,
                'eligible_hedged_ratio': eligible_hedged_ratio,
            }

        # Step 6: 套用折減（numpy mask 賦值）
        disc_same_im = np.zeros(n, dtype=np.float64)
        disc_cross_im = np.zeros(n, dtype=np.float64)

        for b in Bucket:
            rr = reduction_rates_data[b]
            if rr['same_bucket_rate'] <= 0:
                continue
            mask = small_mask_side & bucket_masks[b]
            if not mask.any():
                continue
            effective_rate = rr['same_bucket_rate'] * rr['eligible_hedged_ratio']
            disc_same_im[mask] = base_im_after100[mask] * effective_rate

        # 跨桶折減
        big_side_mv_by_bucket: Dict[str, float] = {}
        for b in Bucket:
            bmd = bucket_metrics_data[b]
            if big_side == 'LONG':
                big_side_mv_by_bucket[b.value] = bmd['mv_long']
            else:
                big_side_mv_by_bucket[b.value] = bmd['mv_short']
        total_big_mv = sum(big_side_mv_by_bucket.values())

        for b in Bucket:
            rr = reduction_rates_data[b]
            residual_ratio = max(0.0, 1.0 - rr['eligible_hedged_ratio'])
            if residual_ratio <= 0:
                continue
            mask = small_mask_side & bucket_masks[b]
            if not mask.any():
                continue
            other_bucket_mv = total_big_mv - big_side_mv_by_bucket.get(b.value, 0)
            small_bucket_mv = mv_after100[mask].sum()
            small_residual_mv = small_bucket_mv * residual_ratio
            if small_residual_mv <= 0 or other_bucket_mv <= 0:
                continue
            cross_ratio = min(1.0, other_bucket_mv / small_residual_mv)
            effective_cross_rate = 0.20 * cross_ratio * residual_ratio
            disc_cross_im[mask] = base_im_after100[mask] * effective_cross_rate

        # Step 7: 彙總計算 IM_today
        reduction_same_bucket = disc_same_im[small_mask_side].sum()
        reduction_cross_bucket = disc_cross_im[small_mask_side].sum()
        total_reduction = reduction_etf_100 + reduction_same_bucket + reduction_cross_bucket

        im_small_after = im_small_before - total_reduction
        im_today = im_big + im_small_after

        # Step 8: MM 與追繳判定
        mm_today = self.mm_ratio * im_today
        if equity is None:
            equity = 0.0
        if equity > 0:
            margin_call = equity < mm_today
            required_deposit = max(0, im_today - equity) if margin_call else 0.0
        else:
            margin_call = False
            required_deposit = 0.0

        # Step 9: 槓桿指標
        gross_mv = long_mv + short_mv
        base_im_total = base_im_long + base_im_short
        gross_leverage = gross_mv / im_today if im_today > 0 else 0.0
        raw_leverage = gross_mv / base_im_total if base_im_total > 0 else 0.0

        # Step 10: 快速路徑不產報表
        return MarginResult(
            asof_date=asof_date, im_today=im_today, mm_today=mm_today,
            equity=equity, margin_call=margin_call, required_deposit=required_deposit,
            big_side=big_side, small_side=small_side,
            long_mv=long_mv, short_mv=short_mv,
            base_im_long=base_im_long, base_im_short=base_im_short,
            im_big=im_big, im_small_before=im_small_before, im_small_after=im_small_after,
            reduction_etf_100=reduction_etf_100,
            reduction_same_bucket=reduction_same_bucket,
            reduction_cross_bucket=reduction_cross_bucket,
            total_reduction=total_reduction,
            gross_leverage=gross_leverage, raw_leverage=raw_leverage,
            summary_df=pd.DataFrame(), side_totals_df=pd.DataFrame(),
            bucket_hedge_df=pd.DataFrame(), small_side_detail_df=pd.DataFrame(),
            atom_detail_df=pd.DataFrame(), reduction_breakdown_df=pd.DataFrame(),
            hedge_pairing_df=pd.DataFrame(),
        )

    def _compute_from_atoms_df(self,
                               atoms_df: pd.DataFrame,
                               atom_records: List[Dict],
                               asof_date: pd.Timestamp,
                               equity: Optional[float],
                               build_reports: bool) -> MarginResult:
        """
        從已建好的 atoms_df（含 leverage/bucket/base_im）執行 Step 3 以後的計算。
        calculate() 和 calculate_fast() 共用此方法。
        """
        # =====================================================================
        # Step 3: 大小邊判定
        # =====================================================================
        base_im_by_side = atoms_df.groupby('side')['base_im'].sum()
        base_im_long = base_im_by_side.get('LONG', 0.0)
        base_im_short = base_im_by_side.get('SHORT', 0.0)

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

        mv_by_side = atoms_df.groupby('side')['mv'].sum()
        long_mv = mv_by_side.get('LONG', 0.0)
        short_mv = mv_by_side.get('SHORT', 0.0)

        # =====================================================================
        # Step 4: ETF 完全對沖 100% 減收
        # =====================================================================
        hedged_amounts = self.etf_lookthrough.compute_full_hedge_amounts(
            [ExplodedPosition(**{k: r[k] for k in ['origin', 'parent', 'code', 'side', 'mv', 'instrument', 'is_from_etf', 'weight']})
             for r in atom_records],
            big_side, small_side
        )

        atoms_df['disc100_mv'] = 0.0
        atoms_df['disc100_im'] = 0.0

        for code, hedged_mv in hedged_amounts.items():
            mask = (
                (atoms_df['side'] == small_side) &
                (atoms_df['is_from_etf'] == True) &
                (atoms_df['code'] == code)
            )
            if not mask.any():
                continue
            total_etf_mv = atoms_df.loc[mask, 'mv'].sum()
            if total_etf_mv > 0:
                hedge_ratio = min(hedged_mv, total_etf_mv) / total_etf_mv
                atoms_df.loc[mask, 'disc100_mv'] = atoms_df.loc[mask, 'mv'] * hedge_ratio
                atoms_df.loc[mask, 'disc100_im'] = atoms_df.loc[mask, 'disc100_mv'] / atoms_df.loc[mask, 'leverage']

        atoms_df['mv_after100'] = (atoms_df['mv'] - atoms_df['disc100_mv']).clip(lower=0)
        atoms_df['base_im_after100'] = atoms_df['mv_after100'] / atoms_df['leverage']

        reduction_etf_100 = atoms_df[atoms_df['side'] == small_side]['disc100_im'].sum()

        # =====================================================================
        # Step 5: 產業桶分析與折減率計算
        # =====================================================================
        residual_atoms = [
            ExplodedPosition(
                origin=row['origin'], parent=row['parent'], code=row['code'],
                side=row['side'], mv=row['mv_after100'], instrument=row['instrument'],
                is_from_etf=row['is_from_etf'], weight=row['weight']
            )
            for row in atoms_df.to_dict('records')
        ]
        bucket_metrics = self.bucket_analyzer.analyze(residual_atoms, asof_date)
        reduction_rates = self.bucket_analyzer.compute_reduction_rates(bucket_metrics, small_side)

        # =====================================================================
        # Step 6: 套用折減
        # =====================================================================
        atoms_df['disc_same_im'] = 0.0
        atoms_df['disc_same_rate'] = 0.0
        atoms_df['disc_cross_im'] = 0.0
        atoms_df['disc_cross_rate'] = 0.0

        for bucket in Bucket:
            rate_result = reduction_rates[bucket]
            if rate_result.same_bucket_rate <= 0:
                continue
            mask = (
                (atoms_df['side'] == small_side) &
                (atoms_df['bucket'] == bucket.value)
            )
            if not mask.any():
                continue
            effective_rate = rate_result.same_bucket_rate * rate_result.eligible_hedged_ratio
            atoms_df.loc[mask, 'disc_same_rate'] = effective_rate
            atoms_df.loc[mask, 'disc_same_im'] = atoms_df.loc[mask, 'base_im_after100'] * effective_rate

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
            other_bucket_mv = total_big_mv - big_side_mv_by_bucket.get(bucket.value, 0)
            small_bucket_mv = atoms_df.loc[mask, 'mv_after100'].sum()
            small_residual_mv = small_bucket_mv * residual_ratio
            if small_residual_mv <= 0 or other_bucket_mv <= 0:
                continue
            cross_ratio = min(1.0, other_bucket_mv / small_residual_mv)
            effective_cross_rate = 0.20 * cross_ratio * residual_ratio
            atoms_df.loc[mask, 'disc_cross_rate'] = effective_cross_rate
            atoms_df.loc[mask, 'disc_cross_im'] = atoms_df.loc[mask, 'base_im_after100'] * effective_cross_rate

        # =====================================================================
        # Step 7: 彙總計算 IM_today
        # =====================================================================
        atoms_df['total_disc_im'] = atoms_df['disc100_im'] + atoms_df['disc_same_im'] + atoms_df['disc_cross_im']
        atoms_df['im_after_disc'] = atoms_df['base_im'] - atoms_df['total_disc_im']

        small_side_df = atoms_df[atoms_df['side'] == small_side]
        reduction_same_bucket = small_side_df['disc_same_im'].sum()
        reduction_cross_bucket = small_side_df['disc_cross_im'].sum()
        total_reduction = reduction_etf_100 + reduction_same_bucket + reduction_cross_bucket

        im_small_after = im_small_before - total_reduction
        im_today = im_big + im_small_after

        # =====================================================================
        # Step 8: MM 與追繳判定
        # =====================================================================
        mm_today = self.mm_ratio * im_today
        if equity is None:
            equity = 0.0
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
        base_im_total = base_im_long + base_im_short
        gross_leverage = gross_mv / im_today if im_today > 0 else 0.0
        raw_leverage = gross_mv / base_im_total if base_im_total > 0 else 0.0

        # =====================================================================
        # Step 10: 建立報表（可選）
        # =====================================================================
        if build_reports:
            summary_df = pd.DataFrame([{
                '估值日': asof_date.strftime('%Y-%m-%d'),
                '大邊': big_side, '小邊': small_side,
                '多方MV': round(long_mv), '空方MV': round(short_mv),
                '多方Base_IM': round(base_im_long), '空方Base_IM': round(base_im_short),
                'IM_big': round(im_big), 'IM_small(折減前)': round(im_small_before),
                'ETF_100%折減': round(reduction_etf_100),
                '同桶折減': round(reduction_same_bucket), '跨桶折減': round(reduction_cross_bucket),
                '總折減': round(total_reduction), 'IM_small(折減後)': round(im_small_after),
                'IM_today': round(im_today), 'MM_today(70%)': round(mm_today),
                '權益': round(equity),
                '追繳觸發': '是' if margin_call else '否',
                '追繳金額': round(required_deposit),
                'Gross槓桿': round(gross_leverage, 2), '無折減槓桿': round(raw_leverage, 2),
            }])
            side_totals_df = atoms_df.groupby('side').agg({
                'mv': 'sum', 'base_im': 'sum', 'disc100_im': 'sum',
                'disc_same_im': 'sum', 'disc_cross_im': 'sum',
                'total_disc_im': 'sum', 'im_after_disc': 'sum'
            }).reset_index()
            side_totals_df.columns = ['邊', 'MV', 'Base_IM', 'ETF_100%折減', '同桶折減', '跨桶折減', '總折減', 'IM淨額']
            bucket_hedge_df = self.bucket_analyzer.get_bucket_summary(bucket_metrics, reduction_rates)
            small_side_detail_df = small_side_df.groupby(['code', 'bucket']).agg({
                'mv_after100': 'sum', 'base_im_after100': 'sum', 'disc100_im': 'sum',
                'disc_same_im': 'sum', 'disc_cross_im': 'sum', 'total_disc_im': 'sum'
            }).reset_index()
            small_side_detail_df.columns = ['代碼', '產業桶', 'MV', 'Base_IM', 'ETF折減', '同桶折減', '跨桶折減', '總折減']
            reduction_breakdown_df = pd.DataFrame([
                {'折減類型': 'ETF完全對沖(100%)', '金額': round(reduction_etf_100), '說明': 'ETF成分股與對向曝險完全對沖'},
                {'折減類型': '同桶對沖', '金額': round(reduction_same_bucket), '說明': '同產業桶多空對沖(50%或20%)'},
                {'折減類型': '跨桶對沖', '金額': round(reduction_cross_bucket), '說明': '不同產業桶多空對沖(20%)'},
                {'折減類型': '合計', '金額': round(total_reduction), '說明': ''},
            ])
            hedge_pairing_df = self._build_hedge_pairing(atoms_df, big_side, small_side, reduction_rates)
        else:
            summary_df = pd.DataFrame()
            side_totals_df = pd.DataFrame()
            bucket_hedge_df = pd.DataFrame()
            small_side_detail_df = pd.DataFrame()
            reduction_breakdown_df = pd.DataFrame()
            hedge_pairing_df = pd.DataFrame()

        return MarginResult(
            asof_date=asof_date, im_today=im_today, mm_today=mm_today,
            equity=equity, margin_call=margin_call, required_deposit=required_deposit,
            big_side=big_side, small_side=small_side,
            long_mv=long_mv, short_mv=short_mv,
            base_im_long=base_im_long, base_im_short=base_im_short,
            im_big=im_big, im_small_before=im_small_before, im_small_after=im_small_after,
            reduction_etf_100=reduction_etf_100,
            reduction_same_bucket=reduction_same_bucket,
            reduction_cross_bucket=reduction_cross_bucket,
            total_reduction=total_reduction,
            gross_leverage=gross_leverage, raw_leverage=raw_leverage,
            summary_df=summary_df, side_totals_df=side_totals_df,
            bucket_hedge_df=bucket_hedge_df, small_side_detail_df=small_side_detail_df,
            atom_detail_df=atoms_df, reduction_breakdown_df=reduction_breakdown_df,
            hedge_pairing_df=hedge_pairing_df,
        )

    def _build_hedge_pairing(self,
                             atoms_df: pd.DataFrame,
                             big_side: str,
                             small_side: str,
                             reduction_rates: Dict) -> pd.DataFrame:
        """
        建立多空配對明細表

        保留 code-level 多空配對視角，同時從 atoms_df 取得實際折減值。
        同桶/跨桶折減在產業桶層級運作，即使不同代碼也可對沖。
        """
        if atoms_df.empty:
            return pd.DataFrame()

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
        pairing = long_by_code.join(short_by_code, how='outer', lsuffix='_l', rsuffix='_s')

        # 數值欄填 0，非數值保留 NaN 再用 combine_first
        for c in ['long_mv', 'long_base_im', 'short_mv', 'short_base_im']:
            if c in pairing.columns:
                pairing[c] = pairing[c].fillna(0)
        pairing['bucket'] = pairing['bucket_l'].combine_first(pairing['bucket_s'])
        pairing['is_from_etf'] = pairing['is_from_etf_l'].fillna(False) | pairing['is_from_etf_s'].fillna(False)
        pairing = pairing.drop(columns=['bucket_l', 'bucket_s', 'is_from_etf_l', 'is_from_etf_s'], errors='ignore')

        # 計算配對金額
        pairing['hedged_mv'] = pairing[['long_mv', 'short_mv']].min(axis=1)

        # 每檔槓桿（同代碼不分邊，取 first）
        code_leverage = atoms_df.groupby('code')['leverage'].first()
        pairing = pairing.join(code_leverage, how='left')

        # 從 atoms_df 取得小邊的實際折減值（桶層級已計算好）
        small_disc = atoms_df[atoms_df['side'] == small_side].groupby('code').agg({
            'disc100_im': 'sum',
            'disc_same_im': 'sum',
            'disc_cross_im': 'sum',
            'total_disc_im': 'sum',
        })
        pairing = pairing.join(small_disc, how='left').fillna(0)

        records = []
        for code, row in pairing.iterrows():
            long_mv = row.get('long_mv', 0)
            short_mv = row.get('short_mv', 0)
            hedged_mv = row.get('hedged_mv', 0)
            bucket = row.get('bucket', '非金電')
            leverage = row.get('leverage', 0)
            long_base_im = row.get('long_base_im', 0)
            short_base_im = row.get('short_base_im', 0)
            base_im = long_base_im + short_base_im

            # 從 atoms_df 取得的實際折減值
            disc100 = row.get('disc100_im', 0)
            disc_same = row.get('disc_same_im', 0)
            disc_cross = row.get('disc_cross_im', 0)
            total_disc = row.get('total_disc_im', 0)
            net_im = base_im - total_disc

            # 減收類型：依實際折減值判定（含比率與詳細資訊）
            parts = []
            if disc100 > 0:
                parts.append('ETF完全對沖(100%)')
            if disc_same > 0:
                bucket_enum = (Bucket.ELECT if bucket == '電子'
                               else (Bucket.FIN if bucket == '金融'
                                     else Bucket.NON))
                rate_result = reduction_rates.get(bucket_enum)
                if rate_result:
                    base_rate = rate_result.same_bucket_rate
                    r_diff_str = f'{rate_result.r_diff:.1%}' if rate_result.r_diff is not None else 'N/A'
                    ratio_str = f'{rate_result.eligible_hedged_ratio:.0%}'
                    parts.append(f'同桶對沖({base_rate:.0%}|報酬差{r_diff_str}|可沖比{ratio_str})')
                else:
                    parts.append('同桶對沖')
            if disc_cross > 0:
                parts.append('跨桶對沖(20%)')

            records.append({
                '代碼': code,
                '產業桶': bucket,
                '槓桿': round(leverage, 2),
                '多方MV': round(long_mv),
                '空方MV': round(short_mv),
                '配對MV': round(hedged_mv),
                'Base_IM': round(base_im),
                'ETF折減': round(disc100),
                '同桶折減': round(disc_same),
                '跨桶折減': round(disc_cross),
                '總折減': round(total_disc),
                '淨IM': round(net_im),
                '減收類型': ' + '.join(parts) if parts else '—',
            })

        result_df = pd.DataFrame(records)

        # 只顯示有曝險的標的
        if not result_df.empty:
            result_df = result_df[(result_df['多方MV'] > 0) | (result_df['空方MV'] > 0)]
            result_df = result_df.sort_values('總折減', ascending=False)

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
                 equity: Optional[float] = None,
                 build_reports: bool = True) -> MarginResult:
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

        # Step 3 以後委派給共用方法
        return self._compute_from_atoms_df(
            atoms_df, atom_records, asof_date, equity, build_reports
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
