# =============================================================================
# 遠期契約保證金模擬平台 - 標的分類與槓桿模組
# 功能：標的類型判定、槓桿倍數計算、ETF look-through 拆解
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class LeverageInfo:
    """槓桿資訊"""
    leverage: float
    source: str  # 槓桿來源說明


@dataclass
class ExplodedPosition:
    """拆解後的曝險單元"""
    origin: str  # 來源類型：DIRECT / ETF_0050 / ETF_0056
    parent: str  # 原始標的代碼
    code: str  # 實際標的代碼
    side: str  # LONG / SHORT
    mv: float  # 名目市值
    instrument: str  # STK / ETF
    is_from_etf: bool  # 是否來自 ETF 拆解
    weight: float = 1.0  # ETF 拆解權重


class InstrumentClassifier:
    """
    標的分類器
    負責判定標的類型與對應槓桿倍數

    槓桿規則：
    - 股票期貨標的（現股屬於股票期貨標的股）：5x
    - 0050/0056 成份股：4x
    - 其他股票：3x
    - 0050 ETF：7x
    - 0056 ETF：7x
    """

    def __init__(self,
                 futures_underlying_set: Set[str],
                 constituent_set_0050_0056: Set[str],
                 leverage_config: Optional[Dict] = None):
        """
        初始化標的分類器

        Args:
            futures_underlying_set: 股票期貨標的代碼集合
            constituent_set_0050_0056: 0050/0056 成分股代碼集合
            leverage_config: 槓桿設定（可選）
        """
        self.futures_underlying_set = futures_underlying_set
        self.constituent_set = constituent_set_0050_0056

        # 預設槓桿設定
        self.leverage_config = leverage_config or {
            'futures_underlying': 5.0,
            'etf_constituent': 4.0,
            'other_stock': 3.0,
            'etf_0050': 7.0,
            'etf_0056': 7.0,
            'default': 3.0,
        }

        # 缺碼記錄（用於稽核）
        self.missing_codes: List[str] = []

    def classify_leverage(self, code: str, instrument: str) -> LeverageInfo:
        """
        依標的類型判定槓桿倍數

        制度規則（依優先順序）：
        1. ETF (0050/0056) => 7x
        2. 股票期貨標的 => 5x
        3. 0050/0056 成份股 => 4x
        4. 其它股票 => 3x

        Args:
            code: 標的代碼
            instrument: 標的類型（STK/ETF）

        Returns:
            LeverageInfo 包含槓桿倍數與來源說明
        """
        code = str(code).strip()

        # 制度條款 2.1：ETF (0050/0056) => 7x
        if instrument == 'ETF':
            if code in {'0050', '50'}:
                return LeverageInfo(
                    leverage=self.leverage_config['etf_0050'],
                    source='ETF_0050_7x'
                )
            if code in {'0056', '56'}:
                return LeverageInfo(
                    leverage=self.leverage_config['etf_0056'],
                    source='ETF_0056_7x'
                )
            # 其他 ETF 預設 3x（保守口徑）
            return LeverageInfo(
                leverage=self.leverage_config['default'],
                source='other_ETF_3x_conservative'
            )

        # 制度條款 2.2：股票期貨標的 => 5x
        if code in self.futures_underlying_set:
            return LeverageInfo(
                leverage=self.leverage_config['futures_underlying'],
                source='futures_underlying_5x'
            )

        # 制度條款 2.3：0050/0056 成份股 => 4x
        if code in self.constituent_set:
            return LeverageInfo(
                leverage=self.leverage_config['etf_constituent'],
                source='etf_constituent_4x'
            )

        # 制度條款 2.4：其它股票 => 3x
        return LeverageInfo(
            leverage=self.leverage_config['other_stock'],
            source='other_stock_3x'
        )

    def classify_leverage_for_etf_component(self,
                                            parent_etf: str,
                                            component_code: str) -> LeverageInfo:
        """
        ETF look-through 後成分股的槓桿判定

        制度規定：拆解後每一檔成份股視為「曝險單元」，
        並以 ETF 槓桿 7x 計算其 Base IM

        Args:
            parent_etf: 母 ETF 代碼（0050/0056）
            component_code: 成分股代碼

        Returns:
            LeverageInfo
        """
        # 制度條款 5.1：ETF look-through 後仍以 7x 計算
        if parent_etf in {'0050', '50'}:
            return LeverageInfo(
                leverage=self.leverage_config['etf_0050'],
                source='ETF_0050_lookthrough_7x'
            )
        if parent_etf in {'0056', '56'}:
            return LeverageInfo(
                leverage=self.leverage_config['etf_0056'],
                source='ETF_0056_lookthrough_7x'
            )

        # 非 0050/0056 的 ETF（保守處理）
        return LeverageInfo(
            leverage=self.leverage_config['default'],
            source='other_ETF_lookthrough_3x_conservative'
        )


class ETFLookthrough:
    """
    ETF Look-through 處理器
    負責將 ETF 曝險拆解至成分股

    制度規則：
    - 把 ETF 名目市值依成份股權重拆解為成份股曝險
    - MV_i_from_ETF = ETF_MV × weight_i
    - 拆解後每檔成份股以 7x 槓桿計算 Base IM
    - 完全對沖（同檔股票反向曝險）可 100% 減收
    """

    def __init__(self,
                 etf_weights: Dict[str, pd.DataFrame],
                 classifier: InstrumentClassifier):
        """
        初始化 ETF Look-through 處理器

        Args:
            etf_weights: ETF 成分股權重，{'0050': df, '0056': df}
            classifier: 標的分類器
        """
        self.etf_weights = etf_weights
        self.classifier = classifier

        # 建立權重查詢表
        self._build_weight_lookup()

    def _build_weight_lookup(self):
        """建立權重快速查詢表"""
        self.weight_lookup: Dict[str, Dict[str, float]] = {}

        for etf_code, df in self.etf_weights.items():
            # 使用向量化操作建立字典
            df = df.copy()
            df['code'] = df['code'].astype(str).str.strip()
            self.weight_lookup[etf_code] = dict(zip(df['code'], df['weight'].astype(float)))

    def is_lookthrough_etf(self, code: str) -> bool:
        """判斷是否為需要 look-through 的 ETF"""
        return str(code).strip() in {'0050', '0056', '50', '56'}

    def normalize_etf_code(self, code: str) -> str:
        """標準化 ETF 代碼"""
        code = str(code).strip()
        if code == '50':
            return '0050'
        if code == '56':
            return '0056'
        return code

    def lookthrough(self,
                   positions: pd.DataFrame,
                   prices_today: Dict[str, float]) -> Tuple[List[ExplodedPosition], pd.DataFrame]:
        """
        執行 ETF look-through 拆解

        制度條款 5.1：把 ETF 名目市值依成份股權重拆解為成份股曝險

        Args:
            positions: 部位 DataFrame，需包含 code, side, qty, instrument
            prices_today: 當日價格字典 {code: price}

        Returns:
            (exploded_positions, detail_df)
        """
        atoms: List[ExplodedPosition] = []
        detail_records = []

        # 使用 to_dict('records') 代替 iterrows（快 10-100 倍）
        for row in positions.to_dict('records'):
            code = str(row['code']).strip()
            side = row['side']
            qty = float(row['qty'])
            instrument = row.get('instrument', 'STK')

            # 取得當日價格
            norm_code = self.normalize_etf_code(code)
            price = prices_today.get(code) or prices_today.get(norm_code, 0)

            mv = qty * price

            # 非 look-through ETF：直接加入
            if not self.is_lookthrough_etf(code) or instrument != 'ETF':
                atoms.append(ExplodedPosition(
                    origin='DIRECT', parent=code, code=code, side=side,
                    mv=mv, instrument=instrument, is_from_etf=False, weight=1.0
                ))
                detail_records.append({
                    'origin': 'DIRECT', 'parent': code, 'code': code,
                    'side': side, 'qty': qty, 'price': price, 'mv': mv,
                    'weight': 1.0, 'is_from_etf': False
                })
                continue

            # ETF look-through 拆解
            etf_code = norm_code
            weights = self.weight_lookup.get(etf_code, {})

            if not weights:
                # 無權重資料，視為直接曝險（保守處理）
                atoms.append(ExplodedPosition(
                    origin=f'ETF_{etf_code}_NO_WEIGHTS', parent=code, code=code,
                    side=side, mv=mv, instrument='ETF', is_from_etf=False, weight=1.0
                ))
                detail_records.append({
                    'origin': f'ETF_{etf_code}_NO_WEIGHTS', 'parent': code,
                    'code': code, 'side': side, 'qty': qty, 'price': price,
                    'mv': mv, 'weight': 1.0, 'is_from_etf': False
                })
                continue

            # 制度條款 5.1：拆解至成分股
            origin = f'ETF_{etf_code}'
            for comp_code, weight in weights.items():
                comp_mv = mv * weight
                atoms.append(ExplodedPosition(
                    origin=origin, parent=code, code=comp_code, side=side,
                    mv=comp_mv, instrument='STK', is_from_etf=True, weight=weight
                ))
                detail_records.append({
                    'origin': origin, 'parent': code, 'code': comp_code,
                    'side': side, 'qty': qty * weight, 'price': price,
                    'mv': comp_mv, 'weight': weight, 'is_from_etf': True
                })

        detail_df = pd.DataFrame(detail_records)
        return atoms, detail_df

    def compute_full_hedge_amounts(self,
                                   atoms: List[ExplodedPosition],
                                   big_side: str,
                                   small_side: str) -> Dict[str, float]:
        """
        計算完全對沖金額

        制度條款 5.2：同一檔股票形成反向曝險，可認定「完全對沖市值」
        MV_hedged_i = min(MV_long_i, MV_short_i)

        只對小邊的 ETF look-through 曝險計算

        Args:
            atoms: 拆解後的曝險單元列表
            big_side: 大邊（LONG/SHORT）
            small_side: 小邊（LONG/SHORT）

        Returns:
            Dict[code, hedged_mv] 各標的的完全對沖金額
        """
        # 彙總大邊各標的 MV
        big_side_mv: Dict[str, float] = {}
        for atom in atoms:
            if atom.side == big_side:
                code = atom.code
                big_side_mv[code] = big_side_mv.get(code, 0) + atom.mv

        # 計算小邊 ETF look-through 曝險的完全對沖金額
        hedged_amounts: Dict[str, float] = {}
        small_side_etf_mv: Dict[str, float] = {}

        # 先彙總小邊 ETF look-through 各標的 MV
        for atom in atoms:
            if atom.side == small_side and atom.is_from_etf:
                code = atom.code
                small_side_etf_mv[code] = small_side_etf_mv.get(code, 0) + atom.mv

        # 計算完全對沖金額
        for code, small_mv in small_side_etf_mv.items():
            big_mv = big_side_mv.get(code, 0)
            if big_mv > 0 and small_mv > 0:
                # 制度條款 5.2：MV_hedged = min(MV_long, MV_short)
                hedged_amounts[code] = min(small_mv, big_mv)

        return hedged_amounts


def etf_lookthrough(positions: pd.DataFrame,
                   etf_weights: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    便捷函式：執行 ETF look-through

    Args:
        positions: 部位 DataFrame
        etf_weights: ETF 成分股權重

    Returns:
        拆解後的部位 DataFrame
    """
    classifier = InstrumentClassifier(
        futures_underlying_set=set(),
        constituent_set_0050_0056=set()
    )
    lt = ETFLookthrough(etf_weights, classifier)

    # 假設價格為 1（僅用於結構展示）
    prices = {str(row['code']): 1.0 for _, row in positions.iterrows()}
    atoms, detail_df = lt.lookthrough(positions, prices)

    return detail_df


def classify_leverage(code: str,
                     instrument: str,
                     futures_underlying_set: Set[str],
                     constituent_set: Set[str]) -> float:
    """
    便捷函式：判定槓桿倍數

    Args:
        code: 標的代碼
        instrument: 標的類型
        futures_underlying_set: 股票期貨標的集合
        constituent_set: 0050/0056 成分股集合

    Returns:
        槓桿倍數（5x/4x/3x/7x）
    """
    classifier = InstrumentClassifier(
        futures_underlying_set=futures_underlying_set,
        constituent_set_0050_0056=constituent_set
    )
    info = classifier.classify_leverage(code, instrument)
    return info.leverage
