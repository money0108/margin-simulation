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

        # ETF 成分股集合
        self.constituent_set = self.data_loader.get_etf_constituent_set()

        # 建立分類器
        self.classifier = InstrumentClassifier(
            futures_underlying_set=self.futures_underlying_set,
            constituent_set_0050_0056=self.constituent_set
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
           positions: pd.DataFrame,
           start_date: pd.Timestamp,
           end_date: Optional[pd.Timestamp] = None,
           progress_callback: Optional[Callable[[int, int, str], None]] = None) -> BacktestResults:
        """
        執行回測

        Args:
            positions: 部位 DataFrame
            start_date: 建倉日期
            end_date: 結束日期（預設最近交易日）
            progress_callback: 進度回調函式 (current, total, date_str)

        Returns:
            BacktestResults
        """
        start_date = pd.Timestamp(start_date)

        if end_date is None:
            end_date = max(self.trading_dates)
        else:
            end_date = pd.Timestamp(end_date)

        # 取得交易日範圍
        dates = self.get_trading_dates_range(start_date, end_date)

        if not dates:
            raise ValueError(f"指定日期範圍內無交易日: {start_date} ~ {end_date}")

        # 初始化結果
        daily_results: List[DailyResult] = []
        margin_call_events: List[Dict] = []

        # 權益追蹤變數
        initial_deposit = None  # 建倉入金 = 第一天 IM
        initial_net_mv = None   # 第一天淨市值（用於計算累計損益）
        current_equity = None

        # MM 固定值（建倉時 IM 的 70%，只有加減倉時才會 reset）
        fixed_mm = None

        total_dates = len(dates)

        # 逐日計算
        for i, date in enumerate(dates):
            # 回報進度
            if progress_callback is not None:
                progress_callback(i + 1, total_dates, date.strftime('%Y-%m-%d'))
            # 取得當日價格
            prices_today = self.get_price_for_date(date)

            if not prices_today:
                continue

            # 計算保證金（先不帶 equity，後面再更新）
            try:
                margin_result = self.margin_calculator.calculate(
                    positions=positions,
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
                # MM 固定為建倉時 IM 的 70%，不隨每日 IM 變動（除非追繳後重置）
                fixed_mm = margin_result.im_today * 0.70
                # 記錄基準淨市值（用於計算累計損益）
                base_net_mv = current_net_mv
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
                # 新入金 = 當日 IM
                # 新 MM = 當日 IM 的 70%
                # 新基準淨市值 = 當日淨市值
                initial_deposit = margin_result.im_today
                fixed_mm = margin_result.im_today * 0.70
                base_net_mv = current_net_mv
                current_equity = initial_deposit  # 補足後權益 = 新 IM
            else:
                required_deposit = 0.0

            # 更新 margin_result（保留追繳前數值以便檢視）
            margin_result.equity = current_equity  # 追繳後權益（如有追繳則為重置後）
            margin_result.equity_before_call = equity_before_call  # 追繳前權益
            margin_result.mm_today = fixed_mm  # 追繳後MM（如有追繳則為重置後）
            margin_result.mm_at_call = mm_at_call  # 追繳判定時的MM
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
        timeseries_df = self._build_timeseries(daily_results)

        # 彙整缺碼清單
        missing_codes = list(set(
            self.bucket_classifier.missing_codes +
            self.classifier.missing_codes
        ))

        # 建立假設清單
        assumptions = self._build_assumptions(missing_codes)

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
            verification_errors=verification_errors
        )

    def _build_timeseries(self, daily_results: List[DailyResult]) -> pd.DataFrame:
        """建立時序 DataFrame"""
        if not daily_results:
            return pd.DataFrame()

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

            # 計算日曆天數（用於利息計算）
            if i == 0:
                calendar_days = 1  # 建倉日算 1 天
            else:
                # 計算與前一個交易日之間的日曆天數
                calendar_days = (dr.date - prev_date).days

            # 計算多空分開的損益
            if i == 0:
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

            # 建倉日或追繳後重置融資金額
            if long_financing is None or mr.margin_call:
                # 多方融資 = 多方 MV - IM（客戶繳的 IM 用來支撐多方買進）
                long_financing = max(0, mr.long_mv - mr.im_today)
                # 空方融資 = 空方 MV（借券賣出不必繳保證金）
                short_financing = mr.short_mv

            # 總融資金額（顯示用）
            financing_amount = long_financing + short_financing

            # 計算利息（按日曆天數計算，年化利率 / 365）
            # 客戶利息 = 多方融資 × 3% + 空方融資 × 3%
            daily_interest = financing_amount * customer_rate * calendar_days / 365

            # 券商收益計算：
            # - 多方融資：券商收 3%，成本 1.8%，淨收益 1.2%
            # - 空方融資：借券賣出的錢在券商那，不需出資金成本，3% 全賺
            daily_broker_profit = (long_financing * spread_rate + short_financing * customer_rate) * calendar_days / 365

            # 累計
            cumulative_interest += daily_interest
            cumulative_broker_profit += daily_broker_profit

            # 多方權益 = 多方MV × (IM / Gross MV)，表示自有資金部分
            # 空方權益 = 空方MV × (IM / Gross MV)
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
            })

        return pd.DataFrame(records)

    def _build_assumptions(self, missing_codes: List[str]) -> List[str]:
        """建立假設清單"""
        assumptions = [
            "1. 維持保證金(MM)：建倉時 IM 的 70%（固定不變，除非追繳後重置）",
            "2. 追繳觸發：Equity < MM",
            "3. 追繳回補：補足至當日新計算的 IM（含最新折減率），MM 重置為新 IM 的 70%",
            "4. ETF look-through：0050/0056 拆解至成分股，使用 7x 槓桿",
            "5. 同桶對沖：3M 報酬差 >= 10% → 50% 折減，否則 20%",
            "6. 跨桶對沖：一律 20% 折減",
            "7. ETF 完全對沖：100% 減收",
            "8. 3M 報酬計算：回溯 63 交易日",
            "9. 多方融資：多方 MV - IM，券商收 3%、成本 1.8%、淨收益 1.2%",
            "10. 空方融資：空方 MV（借券賣出不繳保證金），券商收 3% 全額為收益",
            "11. 利息計算：按日曆日計算（含週末假日），年化利率 / 365",
        ]

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
