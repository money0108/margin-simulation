# =============================================================================
# 遠期契約保證金模擬平台 - 產業桶分類與報酬計算模組
# 功能：產業桶歸類、3M 加權累積報酬率計算
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Bucket(Enum):
    """產業桶列舉"""
    ELECT = "電子"    # 電子
    FIN = "金融"      # 金融
    NON = "非金電"    # 非金電


@dataclass
class BucketMetrics:
    """產業桶指標"""
    bucket: Bucket
    mv_long: float = 0.0
    mv_short: float = 0.0
    base_im_long: float = 0.0
    base_im_short: float = 0.0
    r_3m_long: Optional[float] = None  # 多方 3M 加權報酬
    r_3m_short: Optional[float] = None  # 空方 3M 加權報酬
    is_insufficient_data: bool = False  # 資料不足 3M


@dataclass
class BucketReductionResult:
    """產業桶折減結果"""
    bucket: Bucket
    same_bucket_rate: float = 0.0  # 同桶折減率（50% 或 20%）
    cross_bucket_rate: float = 0.20  # 跨桶折減率（固定 20%）
    eligible_hedged_ratio: float = 0.0  # 可對沖比例
    r_diff: Optional[float] = None  # 3M 報酬率差
    reduction_source: str = ""  # 折減來源說明


class BucketClassifier:
    """
    產業桶分類器

    制度規則：
    - 電子：半導體、電子零組件、電腦週邊、光電、通訊網路等
    - 金融：銀行、保險、金控、證券等
    - 非金電：除電子、金融外之其它產業
    """

    def __init__(self, industry_df: pd.DataFrame):
        """
        初始化產業桶分類器

        Args:
            industry_df: 產業分類 DataFrame，需包含 code, bucket 欄位
        """
        self.industry_df = industry_df

        # 建立快速查詢表
        self._build_lookup()

        # 缺碼記錄
        self.missing_codes: List[str] = []

    def _build_lookup(self):
        """建立代碼對應桶的查詢表"""
        self.bucket_lookup: Dict[str, Bucket] = {}

        # 使用向量化操作
        df = self.industry_df.copy()
        df['code'] = df['code'].astype(str).str.strip()

        # 處理 bucket 欄位可能不存在的情況
        if 'bucket' in df.columns:
            df['bucket_str'] = df['bucket'].astype(str).str.strip().str.upper()
        else:
            df['bucket_str'] = 'NON'

        bucket_map = {'ELECT': Bucket.ELECT, 'FIN': Bucket.FIN}
        for code, bucket_str in zip(df['code'], df['bucket_str']):
            self.bucket_lookup[code] = bucket_map.get(bucket_str, Bucket.NON)

    def classify(self, code: str) -> Bucket:
        """
        依代碼判定產業桶

        若缺碼，以保守口徑歸入非金電並記錄

        Args:
            code: 股票代碼

        Returns:
            Bucket 列舉
        """
        code = str(code).strip()

        if code in self.bucket_lookup:
            return self.bucket_lookup[code]

        # 缺碼處理：記錄並以保守口徑歸入非金電
        if code not in self.missing_codes:
            self.missing_codes.append(code)

        return Bucket.NON

    def classify_batch(self, codes: List[str]) -> Dict[str, Bucket]:
        """批次分類"""
        return {code: self.classify(code) for code in codes}


class ReturnCalculator:
    """
    3M 報酬率計算器

    制度規則：
    - 單檔 3M 累計報酬率：R_i(3M) = (P_end / P_start) - 1
    - P_start 為 t-3M 交易日價格
    - 若不足 3M，標記為「新建倉區間」並採保守折減（20%）
    """

    def __init__(self,
                 prices_df: pd.DataFrame,
                 lookback_days: int = 63):
        """
        初始化報酬計算器

        Args:
            prices_df: 股價 DataFrame，需包含 date, code, close
            lookback_days: 回溯交易日數（預設 63 天，約 3 個月）
        """
        self.prices_df = prices_df
        self.lookback_days = lookback_days

        # 預處理
        self._preprocess()

    def _preprocess(self):
        """預處理股價資料"""
        df = self.prices_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['code', 'date'])

        # 建立日期索引
        self.trading_dates = sorted(df['date'].unique())
        self.date_to_idx = {d: i for i, d in enumerate(self.trading_dates)}

        # 建立股價查詢表（使用 groupby 加速）
        self.prices_lookup: Dict[str, Dict[pd.Timestamp, float]] = {}
        for code, group in df.groupby('code'):
            self.prices_lookup[str(code)] = dict(zip(group['date'], group['close']))

        # 預計算所有股票的 3M 報酬率（快取）
        self._return_cache: Dict[Tuple[str, pd.Timestamp], Tuple[Optional[float], bool]] = {}

    def get_3m_start_date(self, asof_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """
        取得 3M 起始日期

        Args:
            asof_date: 估值日

        Returns:
            3M 前的交易日，若不足則返回 None
        """
        asof_date = pd.Timestamp(asof_date)

        if asof_date not in self.date_to_idx:
            # 找最近的交易日
            dates = [d for d in self.trading_dates if d <= asof_date]
            if not dates:
                return None
            asof_date = max(dates)

        idx = self.date_to_idx[asof_date]

        if idx < self.lookback_days:
            # 資料不足 3M
            return None

        return self.trading_dates[idx - self.lookback_days]

    def compute_return(self,
                      code: str,
                      asof_date: pd.Timestamp) -> Tuple[Optional[float], bool]:
        """
        計算單檔 3M 累計報酬率

        制度條款 4.2：R_i(3M) = (P_end / P_start) - 1

        Args:
            code: 股票代碼
            asof_date: 估值日

        Returns:
            (報酬率, 是否資料不足)
        """
        code = str(code).strip()
        asof_date = pd.Timestamp(asof_date)

        # 檢查快取
        cache_key = (code, asof_date)
        if cache_key in self._return_cache:
            return self._return_cache[cache_key]

        # 取得價格
        code_prices = self.prices_lookup.get(code, {})

        # P_end
        p_end = code_prices.get(asof_date)
        if p_end is None:
            # 嘗試找最近的價格
            dates = [d for d in code_prices.keys() if d <= asof_date]
            if dates:
                p_end = code_prices[max(dates)]

        if p_end is None or p_end == 0:
            self._return_cache[cache_key] = (None, True)
            return None, True

        # P_start
        start_date = self.get_3m_start_date(asof_date)

        if start_date is None:
            # 資料不足 3M
            self._return_cache[cache_key] = (None, True)
            return None, True

        p_start = code_prices.get(start_date)
        if p_start is None:
            # 嘗試找最近的價格
            dates = [d for d in code_prices.keys() if d >= start_date]
            if dates:
                p_start = code_prices[min(dates)]

        if p_start is None or p_start == 0:
            self._return_cache[cache_key] = (None, True)
            return None, True

        # 制度條款 4.2：計算報酬率
        r_3m = (p_end / p_start) - 1.0

        self._return_cache[cache_key] = (r_3m, False)
        return r_3m, False

    def compute_weighted_return(self,
                               code_mv_pairs: List[Tuple[str, float]],
                               asof_date: pd.Timestamp) -> Tuple[Optional[float], bool, Dict]:
        """
        計算 MV 加權平均報酬率

        制度條款 4.2：R_bucket,side(3M) = Σ(w_i × R_i)
        其中 w_i = MV_i / Σ MV_i

        Args:
            code_mv_pairs: (code, mv) 列表
            asof_date: 估值日

        Returns:
            (加權報酬率, 是否有資料不足, 詳細資訊)
        """
        if not code_mv_pairs:
            return None, True, {}

        total_mv = sum(mv for _, mv in code_mv_pairs)
        if total_mv == 0:
            return None, True, {}

        weighted_sum = 0.0
        valid_mv = 0.0
        has_insufficient = False
        details = {}

        for code, mv in code_mv_pairs:
            r_3m, insufficient = self.compute_return(code, asof_date)
            details[code] = {
                'mv': mv,
                'r_3m': r_3m,
                'insufficient_data': insufficient
            }

            if r_3m is not None:
                weight = mv / total_mv
                weighted_sum += weight * r_3m
                valid_mv += mv
            else:
                has_insufficient = True

        if valid_mv == 0:
            return None, True, details

        # 重新計算有效部分的加權
        result = 0.0
        for code, mv in code_mv_pairs:
            r_3m = details[code]['r_3m']
            if r_3m is not None:
                weight = mv / valid_mv
                result += weight * r_3m

        return result, has_insufficient, details


def compute_bucket_returns(prices: pd.DataFrame,
                          lookback_3m_days: int = 63) -> Dict[Bucket, pd.DataFrame]:
    """
    便捷函式：建立報酬計算器

    Args:
        prices: 股價 DataFrame
        lookback_3m_days: 回溯天數

    Returns:
        報酬計算器實例的包裝
    """
    calculator = ReturnCalculator(prices, lookback_3m_days)

    # 這裡返回一個可用於查詢的結構
    result = {}
    for bucket in Bucket:
        result[bucket] = pd.DataFrame()  # 佔位符，實際使用時會動態計算

    return result


class BucketAnalyzer:
    """
    產業桶分析器

    整合產業桶分類與 3M 報酬計算，
    產出對沖折減所需的各項指標
    """

    def __init__(self,
                 bucket_classifier: BucketClassifier,
                 return_calculator: ReturnCalculator):
        """
        初始化分析器

        Args:
            bucket_classifier: 產業桶分類器
            return_calculator: 報酬計算器
        """
        self.bucket_classifier = bucket_classifier
        self.return_calculator = return_calculator

    def analyze(self,
               atoms: List,  # List[ExplodedPosition]
               asof_date: pd.Timestamp) -> Dict[Bucket, BucketMetrics]:
        """
        分析各產業桶指標

        Args:
            atoms: 拆解後的曝險單元列表
            asof_date: 估值日

        Returns:
            Dict[Bucket, BucketMetrics]
        """
        # 初始化各桶指標
        metrics = {bucket: BucketMetrics(bucket=bucket) for bucket in Bucket}

        # 彙總各桶的 MV 與 code-mv 列表
        bucket_long_mv: Dict[Bucket, List[Tuple[str, float]]] = {b: [] for b in Bucket}
        bucket_short_mv: Dict[Bucket, List[Tuple[str, float]]] = {b: [] for b in Bucket}

        for atom in atoms:
            bucket = self.bucket_classifier.classify(atom.code)

            if atom.side == 'LONG':
                metrics[bucket].mv_long += atom.mv
                bucket_long_mv[bucket].append((atom.code, atom.mv))
            else:
                metrics[bucket].mv_short += atom.mv
                bucket_short_mv[bucket].append((atom.code, atom.mv))

        # 計算各桶的加權報酬率
        for bucket in Bucket:
            # 多方 3M 加權報酬
            if bucket_long_mv[bucket]:
                r_long, insuff, _ = self.return_calculator.compute_weighted_return(
                    bucket_long_mv[bucket], asof_date
                )
                metrics[bucket].r_3m_long = r_long
                if insuff:
                    metrics[bucket].is_insufficient_data = True

            # 空方 3M 加權報酬
            if bucket_short_mv[bucket]:
                r_short, insuff, _ = self.return_calculator.compute_weighted_return(
                    bucket_short_mv[bucket], asof_date
                )
                metrics[bucket].r_3m_short = r_short
                if insuff:
                    metrics[bucket].is_insufficient_data = True

        return metrics

    def compute_reduction_rates(self,
                               metrics: Dict[Bucket, BucketMetrics],
                               small_side: str) -> Dict[Bucket, BucketReductionResult]:
        """
        計算各桶的對沖折減率

        制度規則：
        - 同桶：3M 報酬差 >= 10% → 50%，否則 20%
        - 跨桶：一律 20%

        Args:
            metrics: 各桶指標
            small_side: 小邊（LONG/SHORT）

        Returns:
            Dict[Bucket, BucketReductionResult]
        """
        results = {}

        for bucket, m in metrics.items():
            result = BucketReductionResult(bucket=bucket)

            # 計算可對沖比例
            if small_side == 'LONG':
                small_mv = m.mv_long
                big_mv = m.mv_short
                r_small = m.r_3m_long
                r_big = m.r_3m_short
            else:
                small_mv = m.mv_short
                big_mv = m.mv_long
                r_small = m.r_3m_short
                r_big = m.r_3m_long

            if small_mv > 0 and big_mv > 0:
                result.eligible_hedged_ratio = min(big_mv, small_mv) / small_mv
            else:
                result.eligible_hedged_ratio = 0.0

            # 制度條款 4.3：同桶折減率判定
            if m.mv_long > 0 and m.mv_short > 0:
                if r_big is not None and r_small is not None:
                    result.r_diff = r_big - r_small

                    # 制度條款 4.3：若 R_big - R_small >= 10% → 50%
                    if result.r_diff >= 0.10:
                        result.same_bucket_rate = 0.50
                        result.reduction_source = f'同桶_3M報酬差{result.r_diff:.1%}>=10%_折減50%'
                    else:
                        result.same_bucket_rate = 0.20
                        result.reduction_source = f'同桶_3M報酬差{result.r_diff:.1%}<10%_折減20%'
                else:
                    # 資料不足，保守採 20%
                    result.same_bucket_rate = 0.20
                    result.reduction_source = '同桶_資料不足_保守折減20%'
            else:
                result.same_bucket_rate = 0.0
                result.reduction_source = '無對沖'

            # 制度條款 4.4：跨桶一律 20%
            result.cross_bucket_rate = 0.20

            results[bucket] = result

        return results

    def get_bucket_summary(self,
                          metrics: Dict[Bucket, BucketMetrics],
                          reduction_results: Dict[Bucket, BucketReductionResult]) -> pd.DataFrame:
        """
        產出產業桶摘要表

        Returns:
            DataFrame 包含各桶指標與折減率
        """
        records = []

        for bucket in Bucket:
            m = metrics[bucket]
            r = reduction_results[bucket]

            records.append({
                '產業桶': bucket.value,
                '多方MV': m.mv_long,
                '空方MV': m.mv_short,
                '多方3M報酬': f'{m.r_3m_long:.2%}' if m.r_3m_long is not None else 'N/A',
                '空方3M報酬': f'{m.r_3m_short:.2%}' if m.r_3m_short is not None else 'N/A',
                '3M報酬差': f'{r.r_diff:.2%}' if r.r_diff is not None else 'N/A',
                '可對沖比例': f'{r.eligible_hedged_ratio:.2%}',
                '同桶折減率': f'{r.same_bucket_rate:.0%}',
                '折減來源': r.reduction_source,
                '資料不足': '是' if m.is_insufficient_data else '否'
            })

        return pd.DataFrame(records)
