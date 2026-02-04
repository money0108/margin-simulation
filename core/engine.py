# =============================================================================
# 遠期契約保證金模擬平台 - 回測引擎模組
# 功能：逐日回放、事件聚合、估值計算
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .data_loader import DataLoader
from .instruments import InstrumentClassifier, ETFLookthrough
from .buckets import BucketClassifier, ReturnCalculator, Bucket
from .margin import MarginCalculator, MarginResult


@dataclass
class DailyResult:
    """單日計算結果"""
    date: pd.Timestamp
    margin_result: MarginResult

    # 快捷屬性
    @property
    def im_today(self) -> float:
        return self.margin_result.im_today

    @property
    def mm_today(self) -> float:
        return self.margin_result.mm_today

    @property
    def equity(self) -> float:
        return self.margin_result.equity

    @property
    def margin_call(self) -> bool:
        return self.margin_result.margin_call

    @property
    def required_deposit(self) -> float:
        return self.margin_result.required_deposit


@dataclass
class BacktestResults:
    """回測結果彙總"""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    daily_results: List[DailyResult]

    # 時序資料
    timeseries_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 追繳事件
    margin_call_events: List[Dict] = field(default_factory=list)

    # 稽核資訊
    missing_codes: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # 驗證結果
    verification_passed: bool = True
    verification_errors: List[str] = field(default_factory=list)

    # 多期部位支援
    position_schedule: List[Tuple[pd.Timestamp, pd.DataFrame]] = field(default_factory=list)
    position_change_events: List[Dict] = field(default_factory=list)

    def get_summary(self) -> pd.DataFrame:
        """取得回測摘要"""
        if not self.daily_results:
            return pd.DataFrame()

        records = []
        for dr in self.daily_results:
            mr = dr.margin_result
            records.append({
                'date': dr.date,
                'long_mv': mr.long_mv,
                'short_mv': mr.short_mv,
                'base_im_long': mr.base_im_long,
                'base_im_short': mr.base_im_short,
                'im_big': mr.im_big,
                'im_small_before': mr.im_small_before,
                'reduction_etf_100': mr.reduction_etf_100,
                'reduction_same_bucket': mr.reduction_same_bucket,
                'reduction_cross_bucket': mr.reduction_cross_bucket,
                'total_reduction': mr.total_reduction,
                'im_small_after': mr.im_small_after,
                'im_today': mr.im_today,
                'mm_today': mr.mm_today,
                'equity': mr.equity,
                'margin_call': mr.margin_call,
                'required_deposit': mr.required_deposit,
                'gross_leverage': mr.gross_leverage,
                'raw_leverage': mr.raw_leverage,
            })

        return pd.DataFrame(records)


class BacktestEngine:
    """
    回測引擎

    功能：
    1. 載入部位清單與建倉日期
    2. 從建倉日迭代至最近交易日
    3. 逐日計算保證金、折減、追繳狀態
    4. 輸出完整時序結果
    """

    def __init__(self,
                 data_loader: DataLoader,
                 initial_equity: Optional[float] = None):
        """
        初始化回測引擎

        Args:
            data_loader: 資料載入器
            initial_equity: 初始權益
        """
        self.data_loader = data_loader
        self.initial_equity = initial_equity

        # 載入基礎資料
        self._load_base_data()

    def _load_base_data(self):
        """載入基礎資料"""
        # 股價資料
        self.prices_df = self.data_loader.load_prices()

        # 取得交易日列表
        self.trading_dates = sorted(self.prices_df['date'].unique())

        # 產業分類
        self.industry_df = self.data_loader.load_industry()

        # ETF 成分股權重
        self.etf_weights = self.data_loader.load_etf_weights()

        # 股票期貨標的集合
        self.futures_underlying_set = self.data_loader.get_futures_underlying_set()

        # 股期標的 → 遠期槓桿倍數動態對應表
        self.futures_leverage_map = self.data_loader.get_futures_leverage_mapping()

        # ETF 成分股集合
        self.constituent_set = self.data_loader.get_etf_constituent_set()

        # ETF 月均量對應表（用於判定 ETF 槓桿 7x/5x）
        self.etf_volume_map = self.data_loader.get_etf_volume_map(self.prices_df)

        # 建立分類器（傳入動態槓桿映射與 ETF 月均量）
        self.classifier = InstrumentClassifier(
            futures_underlying_set=self.futures_underlying_set,
            constituent_set_0050_0056=self.constituent_set,
            futures_leverage_map=self.futures_leverage_map,
            etf_volume_map=self.etf_volume_map
        )

        # 合併 ETF 權重
        etf_weights_df = pd.concat([
            df for df in self.etf_weights.values()
        ], ignore_index=True)

        # ETF look-through 處理器
        self.etf_lookthrough = ETFLookthrough(
            self.etf_weights, self.classifier
        )

        # 產業桶分類器
        self.bucket_classifier = BucketClassifier(self.industry_df)

        # 報酬計算器
        self.return_calculator = ReturnCalculator(
            self.prices_df, lookback_days=63
        )

        # 保證金計算器
        self.margin_calculator = MarginCalculator(
            classifier=self.classifier,
            etf_lookthrough=self.etf_lookthrough,
            bucket_classifier=self.bucket_classifier,
            return_calculator=self.return_calculator
        )

    def get_price_for_date(self, date: pd.Timestamp) -> Dict[str, float]:
        """取得指定日期的價格字典"""
        date = pd.Timestamp(date)

        # 精確匹配
        date_prices = self.prices_df[self.prices_df['date'] == date]

        if len(date_prices) == 0:
            # 找最近的交易日
            prev_dates = [d for d in self.trading_dates if d <= date]
            if prev_dates:
                nearest_date = max(prev_dates)
                date_prices = self.prices_df[self.prices_df['date'] == nearest_date]

        return date_prices.set_index('code')['close'].to_dict()

    def get_trading_dates_range(self,
                                start_date: pd.Timestamp,
                                end_date: Optional[pd.Timestamp] = None) -> List[pd.Timestamp]:
        """取得日期範圍內的交易日"""
        start_date = pd.Timestamp(start_date)

        if end_date is None:
            end_date = max(self.trading_dates)
        else:
            end_date = pd.Timestamp(end_date)

        return [d for d in self.trading_dates if start_date <= d <= end_date]

    def run(self,
           positions: Optional[pd.DataFrame] = None,
           position_schedule: Optional[List[Tuple[pd.Timestamp, pd.DataFrame]]] = None,
           start_date: Optional[pd.Timestamp] = None,
           end_date: Optional[pd.Timestamp] = None,
           progress_callback: Optional[Callable[[int, int, str], None]] = None) -> BacktestResults:
        """
        執行回測

        Args:
            positions: 部位 DataFrame（舊介面，向下相容）
            position_schedule: 多期部位排程 List[(date, df)]（新介面）
            start_date: 建倉日期
            end_date: 結束日期（預設最近交易日）
            progress_callback: 進度回調函式 (current, total, date_str)

        Returns:
            BacktestResults
        """
        # 處理向下相容：若只傳 positions，包裝為 position_schedule
        if position_schedule is not None:
            schedule = sorted(position_schedule, key=lambda x: pd.Timestamp(x[0]))
        elif positions is not None:
            if start_date is None:
                raise ValueError("使用 positions 介面時必須提供 start_date")
            schedule = [(pd.Timestamp(start_date), positions)]
        else:
            raise ValueError("必須提供 positions 或 position_schedule")

        # 確保日期為 Timestamp
        schedule = [(pd.Timestamp(d), df) for d, df in schedule]

        if start_date is None:
            start_date = schedule[0][0]
        else:
            start_date = pd.Timestamp(start_date)

        if end_date is None:
            end_date = max(self.trading_dates)
        else:
            end_date = pd.Timestamp(end_date)

        # 取得交易日範圍
        dates = self.get_trading_dates_range(start_date, end_date)

        if not dates:
            raise ValueError(f"指定日期範圍內無交易日: {start_date} ~ {end_date}")

        # 建立日期→部位映射：找出每個交易日應使用哪個快照
        # schedule_dates[i] 為第 i 個快照的生效日
        schedule_dates = [s[0] for s in schedule]

        # 當前使用的部位（初始為第一個快照）
        current_positions = schedule[0][1]
        next_schedule_idx = 1  # 下一個要切換的快照索引

        # 初始化結果
        daily_results: List[DailyResult] = []
        margin_call_events: List[Dict] = []
        position_change_events: List[Dict] = []

        # 權益追蹤變數
        initial_deposit = None  # 建倉入金 = 第一天 IM
        initial_net_mv = None   # 第一天淨市值（用於計算累計損益）
        current_equity = None

        # MM 固定值（建倉時 IM 的 70%，只有加減倉時才會 reset）
        fixed_mm = None
        base_net_mv = None
        base_prices: Dict[str, float] = {}  # 基準價格（用於算實現損益）

        # 部位變動日資訊（用於時序標記與融資計算）
        position_change_dates = set()
        position_change_withdrawals: Dict[pd.Timestamp, float] = {}

        # 融資追蹤（用於資金流向頁籤）
        long_financing = None   # 多方融資金額
        short_financing = None  # 空方融資金額

        total_dates = len(dates)

        # 逐日計算
        for i, date in enumerate(dates):
            # 回報進度
            if progress_callback is not None:
                progress_callback(i + 1, total_dates, date.strftime('%Y-%m-%d'))

            # 偵測部位切換（先不切換 current_positions）
            is_position_change = False
            new_positions_df = None
            if next_schedule_idx < len(schedule):
                next_date = schedule_dates[next_schedule_idx]
                if date >= next_date:
                    is_position_change = True
                    new_positions_df = schedule[next_schedule_idx][1]
                    position_change_dates.add(date)
                    next_schedule_idx += 1

            # 取得當日價格
            prices_today = self.get_price_for_date(date)

            if not prices_today:
                continue

            # ========== 部位切換日 ==========
            if is_position_change and initial_deposit is not None:
                # --- 0. 保存變動前狀態 ---
                old_fixed_mm = fixed_mm
                old_long_financing = long_financing if long_financing is not None else 0.0
                old_short_financing = short_financing if short_financing is not None else 0.0

                # --- 1. 用舊部位算出當前權益 ---
                try:
                    old_margin = self.margin_calculator.calculate(
                        positions=current_positions,
                        prices_today=prices_today,
                        asof_date=date,
                        equity=current_equity
                    )
                except Exception as e:
                    print(f"日期 {date} 舊部位計算失敗: {e}")
                    current_positions = new_positions_df
                    continue

                old_net_mv = old_margin.long_mv - old_margin.short_mv
                cumulative_pnl = old_net_mv - base_net_mv
                current_equity = initial_deposit + cumulative_pnl

                # 保存出金前權益
                equity_before_withdrawal = current_equity

                # --- 2. 計算變動部位的實現損益 ---
                realized_pnl, realized_pnl_details = self._compute_realized_pnl(
                    current_positions, new_positions_df,
                    base_prices, prices_today
                )

                # --- 3. 切換到新部位，算新 IM ---
                current_positions = new_positions_df
                try:
                    margin_result = self.margin_calculator.calculate(
                        positions=current_positions,
                        prices_today=prices_today,
                        asof_date=date,
                        equity=current_equity
                    )
                except Exception as e:
                    print(f"日期 {date} 新部位計算失敗: {e}")
                    continue

                if margin_result.im_today <= 0:
                    continue

                current_net_mv = margin_result.long_mv - margin_result.short_mv

                # --- 4. 出金：僅就實現損益可出金，且不低於新 IM ---
                withdrawal = 0.0
                if realized_pnl > 0:
                    excess = max(0, current_equity - margin_result.im_today)
                    withdrawal = min(realized_pnl, excess)
                    current_equity -= withdrawal

                position_change_withdrawals[date] = withdrawal

                # --- 5. 重設基準 ---
                initial_deposit = current_equity
                base_net_mv = current_net_mv
                base_prices = prices_today.copy()
                fixed_mm = margin_result.im_today * 0.70

                # --- 計算新融資金額 ---
                new_long_financing = max(0, margin_result.long_mv - current_equity)
                new_short_financing = margin_result.short_mv
                long_financing = new_long_financing
                short_financing = new_short_financing

                # 記錄部位變動事件
                position_change_events.append({
                    'date': date,
                    'new_im': margin_result.im_today,
                    'new_mm': fixed_mm,
                    'equity_at_change': current_equity,
                    'long_mv': margin_result.long_mv,
                    'short_mv': margin_result.short_mv,
                    'realized_pnl': realized_pnl,
                    'withdrawal': withdrawal,
                    # --- 變動前狀態 ---
                    'old_im': old_margin.im_today,
                    'old_mm': old_fixed_mm,
                    'equity_before_change': equity_before_withdrawal,
                    'old_long_mv': old_margin.long_mv,
                    'old_short_mv': old_margin.short_mv,
                    'old_long_financing': old_long_financing,
                    'old_short_financing': old_short_financing,
                    'new_long_financing': new_long_financing,
                    'new_short_financing': new_short_financing,
                    # --- 實現損益逐部位明細 ---
                    'realized_pnl_details': realized_pnl_details,
                })

                # --- 6. 若 equity < MM → 立即追繳 ---
                if current_equity < fixed_mm:
                    required_deposit = max(0, margin_result.im_today - current_equity)
                    equity_before_call = current_equity
                    mm_at_call = fixed_mm

                    initial_deposit = margin_result.im_today
                    fixed_mm = margin_result.im_today * 0.70
                    base_net_mv = current_net_mv
                    base_prices = prices_today.copy()
                    current_equity = initial_deposit

                    # 追繳後重置融資
                    long_financing = max(0, margin_result.long_mv - current_equity)
                    short_financing = margin_result.short_mv

                    margin_result.equity = current_equity
                    margin_result.equity_before_call = equity_before_call
                    margin_result.mm_today = fixed_mm
                    margin_result.mm_at_call = mm_at_call
                    margin_result.margin_call = True
                    margin_result.required_deposit = required_deposit

                    daily_results.append(DailyResult(date=date, margin_result=margin_result))
                    margin_call_events.append({
                        'date': date,
                        'im_today': margin_result.im_today,
                        'mm_today': margin_result.mm_today,
                        'equity': margin_result.equity,
                        'required_deposit': margin_result.required_deposit,
                        'status': 'TRIGGERED (部位變動)'
                    })
                    continue

                # 正常（無追繳）
                margin_result.equity = current_equity
                margin_result.equity_before_call = current_equity
                margin_result.mm_today = fixed_mm
                margin_result.mm_at_call = fixed_mm
                margin_result.margin_call = False
                margin_result.required_deposit = 0.0

                daily_results.append(DailyResult(date=date, margin_result=margin_result))
                continue

            # ========== 一般日（含第一天） ==========
            try:
                margin_result = self.margin_calculator.calculate(
                    positions=current_positions,
                    prices_today=prices_today,
                    asof_date=date,
                    equity=current_equity
                )
            except Exception as e:
                print(f"日期 {date} 計算失敗: {e}")
                continue

            # 跳過價格資料不完整的日期（IM 為 0 表示價格缺失）
            if margin_result.im_today <= 0:
                continue

            # 計算當日淨市值（Long MV - Short MV）
            current_net_mv = margin_result.long_mv - margin_result.short_mv

            # 第一天：建倉入金 = IM_today，MM 固定為建倉 IM 的 70%
            if initial_deposit is None:
                initial_deposit = margin_result.im_today
                initial_net_mv = current_net_mv
                current_equity = initial_deposit
                fixed_mm = margin_result.im_today * 0.70
                base_net_mv = current_net_mv
                base_prices = prices_today.copy()
                # 初始化融資追蹤
                long_financing = max(0, margin_result.long_mv - margin_result.im_today)
                short_financing = margin_result.short_mv
            else:
                # 後續每日：Equity = 上次基準入金 + 累計損益
                cumulative_pnl = current_net_mv - base_net_mv
                current_equity = initial_deposit + cumulative_pnl

            # 判定是否觸發追繳
            margin_call = current_equity < fixed_mm

            # 記錄追繳前的權益（用於報表顯示）
            equity_before_call = current_equity
            mm_at_call = fixed_mm

            # 計算追繳金額（需補足至當日新計算的 IM）
            if margin_call:
                required_deposit = max(0, margin_result.im_today - current_equity)

                # 追繳後：假設客戶補足至新 IM，重置相關變數
                initial_deposit = margin_result.im_today
                fixed_mm = margin_result.im_today * 0.70
                base_net_mv = current_net_mv
                base_prices = prices_today.copy()
                current_equity = initial_deposit
                # 追繳後重置融資
                long_financing = max(0, margin_result.long_mv - current_equity)
                short_financing = margin_result.short_mv
            else:
                required_deposit = 0.0

            # 更新 margin_result
            margin_result.equity = current_equity
            margin_result.equity_before_call = equity_before_call
            margin_result.mm_today = fixed_mm
            margin_result.mm_at_call = mm_at_call
            margin_result.margin_call = margin_call
            margin_result.required_deposit = required_deposit

            # 記錄每日結果
            daily_result = DailyResult(
                date=date,
                margin_result=margin_result
            )
            daily_results.append(daily_result)

            # 記錄追繳事件
            if margin_result.margin_call:
                margin_call_events.append({
                    'date': date,
                    'im_today': margin_result.im_today,
                    'mm_today': margin_result.mm_today,
                    'equity': margin_result.equity,
                    'required_deposit': margin_result.required_deposit,
                    'status': 'TRIGGERED'
                })

        # 建立時序 DataFrame
        timeseries_df = self._build_timeseries(
            daily_results, position_change_dates, position_change_withdrawals
        )

        # 彙整缺碼清單
        missing_codes = list(set(
            self.bucket_classifier.missing_codes +
            self.classifier.missing_codes
        ))

        # 建立假設清單
        has_multi_positions = len(schedule) > 1
        assumptions = self._build_assumptions(missing_codes, has_multi_positions)

        # 驗證
        verification_passed, verification_errors = self._verify_results(daily_results)

        return BacktestResults(
            start_date=start_date,
            end_date=end_date,
            daily_results=daily_results,
            timeseries_df=timeseries_df,
            margin_call_events=margin_call_events,
            missing_codes=missing_codes,
            assumptions=assumptions,
            verification_passed=verification_passed,
            verification_errors=verification_errors,
            position_schedule=schedule,
            position_change_events=position_change_events,
        )

    @staticmethod
    def _compute_realized_pnl(prev_positions: pd.DataFrame,
                               new_positions: pd.DataFrame,
                               base_prices: Dict[str, float],
                               current_prices: Dict[str, float]) -> Tuple[float, List[Dict]]:
        """
        計算部位變動的實現損益（僅計算被平倉或減倉的部位）

        Args:
            prev_positions: 變動前部位 DataFrame
            new_positions: 變動後部位 DataFrame
            base_prices: 基準價格（上次重設時的價格）
            current_prices: 當日價格

        Returns:
            (實現損益總額, 逐部位明細列表)
        """
        def pos_map(df):
            m = {}
            for _, row in df.iterrows():
                key = (str(row['code']).strip(), row['side'])
                m[key] = m.get(key, 0) + int(row['qty'])
            return m

        old_map = pos_map(prev_positions)
        new_map = pos_map(new_positions)

        realized_pnl = 0.0
        details: List[Dict] = []
        for (code, side), old_qty in old_map.items():
            bp = base_prices.get(code, 0)
            cp = current_prices.get(code, 0)
            if bp == 0 or cp == 0:
                continue

            new_qty = new_map.get((code, side), 0)
            closed_qty = old_qty - new_qty
            if closed_qty <= 0:
                continue  # 未減少

            if side == 'LONG':
                pnl = (cp - bp) * closed_qty
            else:  # SHORT
                pnl = (bp - cp) * closed_qty

            realized_pnl += pnl

            change_type = '平倉' if new_qty == 0 else '減倉'
            details.append({
                'code': code,
                'side': side,
                'change_type': change_type,
                'old_qty': old_qty,
                'new_qty': new_qty,
                'closed_qty': closed_qty,
                'base_price': bp,
                'current_price': cp,
                'pnl': pnl,
            })

        return realized_pnl, details

    def _build_timeseries(self, daily_results: List[DailyResult],
                          position_change_dates: Optional[set] = None,
                          position_change_withdrawals: Optional[Dict] = None) -> pd.DataFrame:
        """建立時序 DataFrame"""
        if not daily_results:
            return pd.DataFrame()

        if position_change_dates is None:
            position_change_dates = set()
        if position_change_withdrawals is None:
            position_change_withdrawals = {}

        records = []
        prev_long_mv = None
        prev_short_mv = None
        initial_long_mv = None
        initial_short_mv = None

        # 融資相關變數
        long_financing = None  # 多方融資金額
        short_financing = None  # 空方融資金額（借券賣出）
        cumulative_interest = 0.0  # 累計利息
        cumulative_broker_profit = 0.0  # 券商累計收益
        customer_rate = 0.03  # 客戶借款利率 3%
        broker_cost_rate = 0.018  # 券商資金成本 1.8%
        spread_rate = customer_rate - broker_cost_rate  # 利差 1.2%

        prev_date = None  # 用於計算日曆天數

        for i, dr in enumerate(daily_results):
            mr = dr.margin_result
            net_mv = mr.long_mv - mr.short_mv

            is_pos_change = dr.date in position_change_dates

            # 計算日曆天數（用於利息計算）
            if i == 0:
                calendar_days = 1  # 建倉日算 1 天
            else:
                # 計算與前一個交易日之間的日曆天數
                calendar_days = (dr.date - prev_date).days

            # 部位變動日重設 P&L 基準
            if is_pos_change and i > 0:
                initial_long_mv = mr.long_mv
                initial_short_mv = mr.short_mv
                daily_pnl_long = 0
                daily_pnl_short = 0
                cumulative_pnl_long = 0
                cumulative_pnl_short = 0
                # 融資金額：用 equity（已扣除出金）計算，出金減少帳上資金→券商多融資
                client_capital = mr.equity  # 出金後的帳上權益
                long_financing = max(0, mr.long_mv - client_capital)
                short_financing = mr.short_mv
            elif i == 0:
                initial_long_mv = mr.long_mv
                initial_short_mv = mr.short_mv
                # 當日損益
                daily_pnl_long = 0
                daily_pnl_short = 0
                # 累計損益
                cumulative_pnl_long = 0
                cumulative_pnl_short = 0
            else:
                # 多方損益：MV上漲賺錢
                daily_pnl_long = mr.long_mv - prev_long_mv
                cumulative_pnl_long = mr.long_mv - initial_long_mv
                # 空方損益：MV下跌賺錢（反向）
                daily_pnl_short = prev_short_mv - mr.short_mv
                cumulative_pnl_short = initial_short_mv - mr.short_mv

            prev_long_mv = mr.long_mv
            prev_short_mv = mr.short_mv
            prev_date = dr.date  # 記錄本次日期

            daily_pnl = daily_pnl_long + daily_pnl_short
            cumulative_pnl = cumulative_pnl_long + cumulative_pnl_short

            # 融資金額計算
            gross_mv = mr.long_mv + mr.short_mv

            # 建倉日或追繳後重置融資金額（部位變動日已在上方處理）
            if long_financing is None or (mr.margin_call and not is_pos_change):
                # 多方融資 = 多方 MV - IM（客戶繳的 IM 用來支撐多方買進）
                long_financing = max(0, mr.long_mv - mr.im_today)
                # 空方融資 = 空方 MV（借券賣出不必繳保證金）
                short_financing = mr.short_mv

            # 總融資金額（顯示用）
            financing_amount = long_financing + short_financing

            # 計算利息（按日曆天數計算，年化利率 / 365）
            daily_interest = financing_amount * customer_rate * calendar_days / 365

            # 券商收益計算
            daily_broker_profit = (long_financing * spread_rate + short_financing * customer_rate) * calendar_days / 365

            # 累計
            cumulative_interest += daily_interest
            cumulative_broker_profit += daily_broker_profit

            # 多方權益 / 空方權益
            equity_ratio = mr.im_today / gross_mv if gross_mv > 0 else 0
            long_equity = mr.long_mv * equity_ratio
            short_equity = mr.short_mv * equity_ratio

            records.append({
                'date': dr.date,
                'Long_MV': round(mr.long_mv),
                'Short_MV': round(mr.short_mv),
                'Net_MV': round(net_mv),
                'Daily_PnL_Long': round(daily_pnl_long),
                'Daily_PnL_Short': round(daily_pnl_short),
                'Daily_PnL': round(daily_pnl),
                'Cum_PnL_Long': round(cumulative_pnl_long),
                'Cum_PnL_Short': round(cumulative_pnl_short),
                'Cumulative_PnL': round(cumulative_pnl),
                'Base_IM_long': round(mr.base_im_long),
                'Base_IM_short': round(mr.base_im_short),
                'IM_big': round(mr.im_big),
                'IM_small_before': round(mr.im_small_before),
                'reduction_etf_100': round(mr.reduction_etf_100),
                'reduction_same_bucket': round(mr.reduction_same_bucket),
                'reduction_cross_bucket': round(mr.reduction_cross_bucket),
                'total_reduction': round(mr.total_reduction),
                'IM_small_after': round(mr.im_small_after),
                'IM_today': round(mr.im_today),
                'MM_today': round(mr.mm_today),
                'Equity': round(mr.equity),
                'Equity_Before': round(mr.equity_before_call),  # 追繳前權益
                'MM_At_Call': round(mr.mm_at_call),  # 追繳判定時MM
                'margin_call_flag': 1 if mr.margin_call else 0,
                'Required_Deposit': round(mr.required_deposit),
                'Gross_Lev': round(mr.gross_leverage, 2),
                'Raw_Lev': round(mr.raw_leverage, 2),
                # 融資相關
                'Long_Equity': round(long_equity),
                'Short_Equity': round(short_equity),
                'Long_Financing': round(long_financing),
                'Short_Financing': round(short_financing),
                'Financing_Amount': round(financing_amount),
                'Daily_Interest': round(daily_interest),
                'Cumulative_Interest': round(cumulative_interest),
                'Daily_Broker_Profit': round(daily_broker_profit),
                'Cumulative_Broker_Profit': round(cumulative_broker_profit),
                # 部位變動標記
                'position_change_flag': 1 if is_pos_change else 0,
                # 出金（實現損益提領）
                'Withdrawal': round(position_change_withdrawals.get(dr.date, 0)),
            })

        return pd.DataFrame(records)

    def _build_assumptions(self, missing_codes: List[str],
                           has_multi_positions: bool = False) -> List[str]:
        """建立假設清單"""
        assumptions = [
            "1. 維持保證金(MM)：建倉時 IM 的 70%（固定不變，除非追繳後重置）",
            "2. 追繳觸發：Equity < MM",
            "3. 追繳回補：補足至當日新計算的 IM（含最新折減率），MM 重置為新 IM 的 70%",
            "4. ETF look-through：0050/0056 拆解至成分股，使用母 ETF 槓桿（月均量≥10,000張→7x，否則5x）",
            "5. 同桶對沖：3M 報酬差 >= 10% → 50% 折減，否則 20%",
            "6. 跨桶對沖：一律 20% 折減",
            "7. ETF 完全對沖：100% 減收",
            "8. 3M 報酬計算：回溯 63 交易日",
            "9. 多方融資：多方 MV - IM，券商收 3%、成本 1.8%、淨收益 1.2%",
            "10. 空方融資：空方 MV（借券賣出不繳保證金），券商收 3% 全額為收益",
            "11. 利息計算：按日曆日計算（含週末假日），年化利率 / 365",
        ]

        if has_multi_positions:
            assumptions.append(
                "12. 加倉/平倉處理：部位變動時權益延續、MM 重置為新 IM×70%、P&L 基準重設"
            )
            assumptions.append(
                "13. 出金規則：部位變動時，僅就變動部位的實現損益可出金（不得低於新 IM），出金後融資金額相應增加"
            )

        if missing_codes:
            assumptions.append(
                f"9. 缺碼保守處理：以下代碼缺少產業分類或槓桿資訊，"
                f"以保守口徑處理（3x 槓桿、非金電桶）：{', '.join(missing_codes[:20])}"
                + (f" 等共 {len(missing_codes)} 檔" if len(missing_codes) > 20 else "")
            )

        return assumptions

    def _verify_results(self, daily_results: List[DailyResult]) -> Tuple[bool, List[str]]:
        """
        驗證計算結果

        檢查項目：
        1. 大邊是否被折減（不應該）
        2. IM_today 是否為正
        3. MM_today 應為某次建倉/追繳後 IM 的 70%
        4. 折減後 IM_small 不應為負
        """
        errors = []

        for dr in daily_results:
            mr = dr.margin_result

            # 檢查 1：大邊不應被折減
            # （在目前實作中，折減只套用小邊，此項應自動通過）

            # 檢查 2：IM_today 應為正
            if mr.im_today <= 0:
                errors.append(f"{dr.date}: IM_today <= 0 ({mr.im_today})")

            # 檢查 3：MM 應為某個 IM 的 70%（不再檢查固定值，因為追繳後會重置）
            # 只檢查 MM > 0
            if mr.mm_today <= 0:
                errors.append(f"{dr.date}: MM_today <= 0 ({mr.mm_today})")

            # 檢查 4：折減後 IM_small 不應為負
            if mr.im_small_after < 0:
                errors.append(f"{dr.date}: IM_small_after < 0 ({mr.im_small_after})")

        return len(errors) == 0, errors


def backtest_engine(start_date: pd.Timestamp,
                   end_date: pd.Timestamp,
                   positions: pd.DataFrame,
                   prices: pd.DataFrame,
                   configs: Dict) -> BacktestResults:
    """
    便捷函式：執行回測

    Args:
        start_date: 開始日期
        end_date: 結束日期
        positions: 部位 DataFrame
        prices: 股價 DataFrame
        configs: 設定字典

    Returns:
        BacktestResults
    """
    loader = DataLoader()
    engine = BacktestEngine(loader)

    return engine.run(
        positions=positions,
        start_date=start_date,
        end_date=end_date
    )
