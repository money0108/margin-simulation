# =============================================================================
# 遠期契約保證金模擬平台 - 資料載入模組
# 功能：CSV/XLSX/MD 載入與驗證、快取機制、缺值處理、雲端下載
# =============================================================================

import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple, Any
from dataclasses import dataclass, field
from io import BytesIO
import warnings
import tempfile
import requests

warnings.filterwarnings('ignore')

# 嘗試導入 gdown（用於大文件下載）
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False


def download_from_google_drive(file_id: str, destination: str, is_sheet: bool = False) -> bool:
    """
    從 Google Drive 或 Google Sheets 下載檔案

    Args:
        file_id: Google Drive/Sheets 文件 ID
        destination: 目標檔案路徑
        is_sheet: 是否為 Google Sheets（需要用不同的下載方式）

    Returns:
        是否下載成功
    """
    try:
        if is_sheet:
            # Google Sheets 用匯出連結
            URL = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"
            response = requests.get(URL, stream=True)
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            return True
        else:
            # Google Drive 大文件使用 gdown
            if HAS_GDOWN:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, destination, quiet=False)
                return True
            else:
                # 備用方案：使用 requests（可能對大文件不穩定）
                URL = "https://drive.google.com/uc?export=download"
                session = requests.Session()
                response = session.get(URL, params={'id': file_id}, stream=True)

                # 檢查是否需要確認
                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                        break

                if token:
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(URL, params=params, stream=True)

                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                return True
    except Exception as e:
        print(f"下載失敗: {e}")
        return False


@dataclass
class DataValidationResult:
    """資料驗證結果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_ratio: float = 0.0
    repair_suggestions: List[str] = field(default_factory=list)


class DataLoader:
    """
    資料載入器
    - 支援 CSV, XLSX, MD 格式
    - 路徑驗證與錯誤處理
    - 快取機制
    - 缺值處理與報告
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化資料載入器

        Args:
            config_path: 設定檔路徑，若為 None 則使用預設路徑
        """
        # 設定快取
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # 載入設定
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        self.config = self._load_config(config_path)

        # 缺值處理報告
        self.missing_data_report: Dict[str, Dict] = {}

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """載入設定檔"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"設定檔不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _validate_path(self, path: Union[str, Path]) -> Tuple[bool, str]:
        """
        驗證路徑是否存在

        Returns:
            (是否有效, 錯誤訊息)
        """
        path = Path(path)
        if not path.exists():
            return False, f"檔案不存在: {path}"
        if not path.is_file():
            return False, f"路徑不是檔案: {path}"
        return True, ""

    def _get_cache_key(self, path: str, **kwargs) -> str:
        """產生快取鍵值"""
        return f"{path}_{hash(frozenset(kwargs.items()))}"

    def _check_cache(self, cache_key: str, path: str) -> Optional[Any]:
        """檢查快取是否有效"""
        if cache_key not in self._cache:
            return None

        # 檢查檔案是否有更新
        file_mtime = os.path.getmtime(path)
        cache_time = self._cache_timestamps.get(cache_key, 0)

        if file_mtime > cache_time:
            return None

        return self._cache[cache_key]

    def _set_cache(self, cache_key: str, path: str, data: Any):
        """設定快取"""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = os.path.getmtime(path)

    def clear_cache(self):
        """清除所有快取"""
        self._cache.clear()
        self._cache_timestamps.clear()

    # =========================================================================
    # 股價資料載入
    # =========================================================================

    def _get_cloud_data_path(self, data_type: str) -> str:
        """
        取得雲端數據的本地快取路徑

        Args:
            data_type: 數據類型（stock_prices, symbols_summary 等）

        Returns:
            本地快取檔案路徑
        """
        cache_dir = Path(tempfile.gettempdir()) / "margin_simulation_data"
        cache_dir.mkdir(exist_ok=True)

        filenames = {
            'stock_prices': 'stock_price_last2y.csv',
            'symbols_summary': 'symbols_summary.csv',
            'industry_classification': 'industry_classification.xlsx',
            'etf_weights': 'etf_weights.xlsx',
            'leverage_table': 'leverage_table.xlsx',
            'sample_positions': 'sample_positions.xlsx',
        }
        return str(cache_dir / filenames.get(data_type, f'{data_type}.csv'))

    def _download_cloud_data(self, data_type: str) -> Optional[str]:
        """
        從 Google Drive/Sheets 下載數據

        Args:
            data_type: 數據類型

        Returns:
            下載後的本地路徑，失敗返回 None
        """
        google_drive_config = self.config.get('google_drive', {})
        file_id = google_drive_config.get(data_type)

        if not file_id:
            return None

        local_path = self._get_cloud_data_path(data_type)

        # 檢查是否已經下載過
        if Path(local_path).exists():
            return local_path

        # 判斷是否為 Google Sheets（xlsx 檔案）
        is_sheet = data_type in ['industry_classification', 'etf_weights', 'leverage_table', 'sample_positions']

        print(f"正在從雲端下載 {data_type}...")
        if download_from_google_drive(file_id, local_path, is_sheet=is_sheet):
            print(f"下載完成: {local_path}")
            return local_path
        else:
            return None

    def load_prices(self,
                   path: Optional[str] = None,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        載入股價歷史數據

        Args:
            path: CSV 檔案路徑，若為 None 則使用設定檔路徑
            use_cache: 是否使用快取

        Returns:
            DataFrame，必含欄位: date, code, close
        """
        if path is None:
            path = self.config['paths']['stock_prices']

        # 路徑驗證
        is_valid, error_msg = self._validate_path(path)
        if not is_valid:
            # 嘗試從雲端下載
            cloud_path = self._download_cloud_data('stock_prices')
            if cloud_path:
                path = cloud_path
            else:
                raise FileNotFoundError(error_msg)

        # 快取檢查
        cache_key = self._get_cache_key(path, type='prices')
        if use_cache:
            cached = self._check_cache(cache_key, path)
            if cached is not None:
                return cached.copy()

        # 載入資料
        col_map = self.config['column_mapping']['prices']

        df = pd.read_csv(path, encoding='utf-8-sig')

        # 標準化欄位名稱
        rename_map = {}
        for std_name, orig_name in col_map.items():
            if orig_name in df.columns:
                rename_map[orig_name] = std_name

        df = df.rename(columns=rename_map)

        # 確保必要欄位存在
        required_cols = ['date', 'code', 'close']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"股價資料缺少必要欄位: {missing_cols}")

        # 日期格式轉換（支援 YYYYMMDD 格式）
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')

        # 股票代號標準化（轉字串，移除前後空白）
        df['code'] = df['code'].astype(str).str.strip()

        # 收盤價轉數值
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # 移除無效資料
        original_len = len(df)
        df = df.dropna(subset=['date', 'code', 'close'])

        # 記錄缺值處理
        missing_count = original_len - len(df)
        self.missing_data_report['prices'] = {
            'original_count': original_len,
            'valid_count': len(df),
            'missing_count': missing_count,
            'missing_ratio': missing_count / original_len if original_len > 0 else 0
        }

        # 排序
        df = df.sort_values(['code', 'date']).reset_index(drop=True)

        # 設定快取
        if use_cache:
            self._set_cache(cache_key, path, df)

        return df

    # =========================================================================
    # 公司名稱對照載入
    # =========================================================================

    def load_mapping(self,
                    path: Optional[str] = None,
                    use_cache: bool = True) -> pd.DataFrame:
        """
        載入股票代號公司名稱對照

        Args:
            path: CSV 檔案路徑
            use_cache: 是否使用快取

        Returns:
            DataFrame，包含 code, name 欄位
        """
        if path is None:
            path = self.config['paths']['company_mapping']

        is_valid, error_msg = self._validate_path(path)
        if not is_valid:
            raise FileNotFoundError(error_msg)

        cache_key = self._get_cache_key(path, type='mapping')
        if use_cache:
            cached = self._check_cache(cache_key, path)
            if cached is not None:
                return cached.copy()

        df = pd.read_csv(path, encoding='utf-8-sig')

        # 標準化欄位
        if '原始代碼' in df.columns:
            df = df.rename(columns={'原始代碼': 'code', '股票名稱': 'name'})

        df['code'] = df['code'].astype(str).str.strip()

        if use_cache:
            self._set_cache(cache_key, path, df)

        return df

    # =========================================================================
    # 產業分類載入
    # =========================================================================

    def load_industry(self,
                     path: Optional[str] = None,
                     use_cache: bool = True) -> pd.DataFrame:
        """
        載入產業分類資料

        Args:
            path: XLSX 檔案路徑
            use_cache: 是否使用快取

        Returns:
            DataFrame，包含 code, bucket 欄位（bucket ∈ {ELECT, FIN, NON}）
        """
        if path is None:
            path = self.config['paths']['industry_classification']

        is_valid, error_msg = self._validate_path(path)
        if not is_valid:
            # 嘗試從雲端下載
            cloud_path = self._download_cloud_data('industry_classification')
            if cloud_path:
                path = cloud_path
            else:
                raise FileNotFoundError(error_msg)

        cache_key = self._get_cache_key(path, type='industry')
        if use_cache:
            cached = self._check_cache(cache_key, path)
            if cached is not None:
                return cached.copy()

        df = pd.read_excel(path)

        # 標準化欄位名稱
        col_map = self.config['column_mapping']['industry']
        rename_map = {}
        for std_name, orig_name in col_map.items():
            if orig_name in df.columns:
                rename_map[orig_name] = std_name
        df = df.rename(columns=rename_map)

        # 確保 code 欄位存在
        if 'code' not in df.columns:
            # 嘗試尋找代號欄位
            for col in df.columns:
                if '代號' in col:
                    df = df.rename(columns={col: 'code'})
                    break

        df['code'] = df['code'].astype(str).str.strip()

        # 產業桶分類邏輯
        bucket_config = self.config.get('bucket_mapping', {})
        electronic_keywords = bucket_config.get('electronic', ['電子'])
        financial_keywords = bucket_config.get('financial', ['金融'])

        def classify_bucket(row) -> str:
            """依主產業分類判定桶"""
            main_ind = str(row.get('main_industry', '')).strip()
            sub_ind = str(row.get('sub_industry', '')).strip()
            ind_name = str(row.get('industry_name', '')).strip()

            # 電子桶判定
            for kw in electronic_keywords:
                if kw in main_ind or kw in sub_ind or kw in ind_name:
                    return 'ELECT'

            # 金融桶判定
            for kw in financial_keywords:
                if kw in main_ind or kw in sub_ind or kw in ind_name:
                    return 'FIN'

            # 預設非金電
            return 'NON'

        df['bucket'] = df.apply(classify_bucket, axis=1)

        if use_cache:
            self._set_cache(cache_key, path, df)

        return df

    # =========================================================================
    # 槓桿倍數表載入
    # =========================================================================

    def load_leverage_table(self,
                           path: Optional[str] = None,
                           use_cache: bool = True) -> pd.DataFrame:
        """
        載入股票期貨標的與槓桿倍數對照表

        Args:
            path: XLSX 檔案路徑
            use_cache: 是否使用快取

        Returns:
            DataFrame，包含現貨標的代碼與槓桿資訊
        """
        if path is None:
            path = self.config['paths']['leverage_table']

        is_valid, error_msg = self._validate_path(path)
        if not is_valid:
            # 嘗試從雲端下載
            cloud_path = self._download_cloud_data('leverage_table')
            if cloud_path:
                path = cloud_path
            else:
                raise FileNotFoundError(error_msg)

        cache_key = self._get_cache_key(path, type='leverage')
        if use_cache:
            cached = self._check_cache(cache_key, path)
            if cached is not None:
                return cached.copy()

        df = pd.read_excel(path)

        # 標準化欄位
        col_map = self.config['column_mapping']['leverage']
        rename_map = {}
        for std_name, orig_name in col_map.items():
            if orig_name in df.columns:
                rename_map[orig_name] = std_name
        df = df.rename(columns=rename_map)

        # 提取現貨標的代碼
        if 'underlying_code' in df.columns:
            df['underlying_code'] = df['underlying_code'].astype(str).str.strip()

        if use_cache:
            self._set_cache(cache_key, path, df)

        return df

    def get_futures_underlying_set(self,
                                   path: Optional[str] = None) -> set:
        """
        取得股票期貨標的代碼集合（用於判定 5x 槓桿）

        Returns:
            股票期貨標的代碼集合
        """
        df = self.load_leverage_table(path)
        if 'underlying_code' in df.columns:
            return set(df['underlying_code'].dropna().astype(str).str.strip())
        return set()

    # =========================================================================
    # ETF 成分股權重載入
    # =========================================================================

    def load_etf_weights(self,
                        path: Optional[str] = None,
                        use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        載入 ETF 成分股與權重

        Args:
            path: XLSX 檔案路徑
            use_cache: 是否使用快取

        Returns:
            Dict[str, DataFrame]，key 為 ETF 代碼（'0050', '0056'）
        """
        if path is None:
            path = self.config['paths']['etf_weights']

        is_valid, error_msg = self._validate_path(path)
        if not is_valid:
            # 嘗試從雲端下載
            cloud_path = self._download_cloud_data('etf_weights')
            if cloud_path:
                path = cloud_path
            else:
                raise FileNotFoundError(error_msg)

        cache_key = self._get_cache_key(path, type='etf_weights')
        if use_cache:
            cached = self._check_cache(cache_key, path)
            if cached is not None:
                return {k: v.copy() for k, v in cached.items()}

        df = pd.read_excel(path, header=None)

        result = {}

        # 解析 0050 資料（前三欄）
        # 第一列是標題列（0050成股, ...），第二列是欄位名（代號, 名稱, 成分股權重）
        df_0050 = df.iloc[2:, :3].copy()
        df_0050.columns = ['code', 'name', 'weight']
        df_0050 = df_0050.dropna(subset=['code'])
        df_0050['code'] = df_0050['code'].astype(str).str.strip()
        df_0050['weight'] = pd.to_numeric(df_0050['weight'], errors='coerce') / 100.0  # 轉為比例
        df_0050 = df_0050[df_0050['code'] != '']
        df_0050['etf_code'] = '0050'
        result['0050'] = df_0050[['etf_code', 'code', 'name', 'weight']].reset_index(drop=True)

        # 解析 0056 資料（後三欄，從第7欄開始）
        df_0056 = df.iloc[2:, 6:9].copy()
        df_0056.columns = ['code', 'name', 'weight']
        df_0056 = df_0056.dropna(subset=['code'])
        df_0056['code'] = df_0056['code'].astype(str).str.strip()
        df_0056['weight'] = pd.to_numeric(df_0056['weight'], errors='coerce') / 100.0
        df_0056 = df_0056[df_0056['code'] != '']
        df_0056['etf_code'] = '0056'
        result['0056'] = df_0056[['etf_code', 'code', 'name', 'weight']].reset_index(drop=True)

        if use_cache:
            self._set_cache(cache_key, path, result)

        return result

    def get_etf_constituent_set(self, path: Optional[str] = None) -> set:
        """
        取得 0050/0056 成分股代碼集合（用於判定 4x 槓桿）

        Returns:
            成分股代碼集合
        """
        etf_weights = self.load_etf_weights(path)
        codes = set()
        for etf_code, df in etf_weights.items():
            codes.update(df['code'].astype(str).str.strip())
        return codes

    # =========================================================================
    # 部位資料載入
    # =========================================================================

    def load_positions(self,
                      xlsx_path_or_buffer: Union[str, BytesIO, None] = None,
                      use_cache: bool = False) -> pd.DataFrame:
        """
        載入部位清單（Excel）

        Args:
            xlsx_path_or_buffer: Excel 檔案路徑或上傳的檔案緩衝區
            use_cache: 是否使用快取（上傳檔案不使用快取）

        Returns:
            DataFrame，欄位: trade_date, code, side, qty, instrument, entry_price
        """
        if xlsx_path_or_buffer is None:
            xlsx_path_or_buffer = self.config['paths']['sample_positions']

        # 判斷是路徑還是緩衝區
        is_path = isinstance(xlsx_path_or_buffer, str)

        if is_path:
            is_valid, error_msg = self._validate_path(xlsx_path_or_buffer)
            if not is_valid:
                # 嘗試從雲端下載
                cloud_path = self._download_cloud_data('sample_positions')
                if cloud_path:
                    xlsx_path_or_buffer = cloud_path
                else:
                    raise FileNotFoundError(error_msg)

        # 載入資料
        df = pd.read_excel(xlsx_path_or_buffer)

        # 處理「模擬1l部位.xlsx」格式：代號, 買進張數, 賣出張數
        if '代號' in df.columns and ('買進張數' in df.columns or '賣出張數' in df.columns):
            positions = []
            multiplier = self.config.get('contract_multiplier', 1000)

            for _, row in df.iterrows():
                code = str(row['代號']).strip()
                # 移除數字後的 .0（Excel 讀取數字會變成 float）
                if code.endswith('.0'):
                    code = code[:-2]
                if code == '' or code == 'nan':
                    continue

                # 標準化 ETF 代碼
                if code in ['50', '56']:
                    code = '00' + code

                # 判斷是 ETF 還是股票
                instrument = 'ETF' if code in ['0050', '0056'] else 'STK'

                # 買進（LONG）
                buy_qty = row.get('買進張數', 0)
                if pd.notna(buy_qty) and float(buy_qty) != 0:
                    qty_shares = abs(float(buy_qty)) * multiplier
                    positions.append({
                        'code': code,
                        'side': 'LONG',
                        'qty': int(qty_shares),
                        'instrument': instrument,
                    })

                # 賣出（SHORT）
                sell_qty = row.get('賣出張數', 0)
                if pd.notna(sell_qty) and float(sell_qty) != 0:
                    qty_shares = abs(float(sell_qty)) * multiplier
                    positions.append({
                        'code': code,
                        'side': 'SHORT',
                        'qty': int(qty_shares),
                        'instrument': instrument,
                    })

            result = pd.DataFrame(positions)

        # 標準格式：trade_date, code, side, qty, instrument, entry_price
        else:
            result = df.copy()

            # 欄位標準化
            if 'trade_date' in result.columns:
                result['trade_date'] = pd.to_datetime(result['trade_date'])

            if 'code' in result.columns:
                result['code'] = result['code'].astype(str).str.strip()

            # 確保必要欄位
            required = ['code', 'side', 'qty']
            missing = [c for c in required if c not in result.columns]
            if missing:
                raise ValueError(f"部位資料缺少必要欄位: {missing}")

        return result

    # =========================================================================
    # 資料驗證
    # =========================================================================

    def validate_data(self) -> DataValidationResult:
        """
        驗證所有資料檔案

        Returns:
            DataValidationResult 包含驗證結果
        """
        result = DataValidationResult(is_valid=True)

        # 檢查各資料檔案
        paths_to_check = [
            ('stock_prices', '股價歷史數據'),
            ('company_mapping', '公司名稱對照'),
            ('industry_classification', '產業分類'),
            ('leverage_table', '槓桿倍數表'),
            ('etf_weights', 'ETF成分股權重'),
        ]

        for path_key, desc in paths_to_check:
            path = self.config['paths'].get(path_key)
            if path:
                is_valid, error_msg = self._validate_path(path)
                if not is_valid:
                    result.is_valid = False
                    result.errors.append(f"{desc}: {error_msg}")
                    result.repair_suggestions.append(
                        f"請確認 {desc} 檔案路徑是否正確: {path}"
                    )

        # 載入並驗證資料
        try:
            prices = self.load_prices()
            if len(prices) == 0:
                result.warnings.append("股價資料為空")

            # 檢查日期範圍
            date_range = prices['date'].max() - prices['date'].min()
            if date_range.days < 180:
                result.warnings.append(
                    f"股價資料日期範圍不足 6 個月 ({date_range.days} 天)"
                )
        except Exception as e:
            result.errors.append(f"載入股價資料失敗: {str(e)}")
            result.is_valid = False

        try:
            industry = self.load_industry()
            if len(industry) == 0:
                result.warnings.append("產業分類資料為空")
        except Exception as e:
            result.errors.append(f"載入產業分類失敗: {str(e)}")
            result.is_valid = False

        try:
            etf_weights = self.load_etf_weights()
            for etf_code in ['0050', '0056']:
                if etf_code not in etf_weights or len(etf_weights[etf_code]) == 0:
                    result.warnings.append(f"{etf_code} 成分股權重資料為空")
        except Exception as e:
            result.errors.append(f"載入 ETF 權重失敗: {str(e)}")
            result.is_valid = False

        # 計算整體缺值比率
        total_missing = sum(
            r.get('missing_count', 0)
            for r in self.missing_data_report.values()
        )
        total_original = sum(
            r.get('original_count', 0)
            for r in self.missing_data_report.values()
        )
        if total_original > 0:
            result.missing_ratio = total_missing / total_original

        return result

    def get_missing_data_report(self) -> pd.DataFrame:
        """取得缺值處理報告"""
        if not self.missing_data_report:
            return pd.DataFrame()

        records = []
        for source, info in self.missing_data_report.items():
            records.append({
                '資料來源': source,
                '原始筆數': info.get('original_count', 0),
                '有效筆數': info.get('valid_count', 0),
                '缺值筆數': info.get('missing_count', 0),
                '缺值比率': f"{info.get('missing_ratio', 0):.2%}"
            })

        return pd.DataFrame(records)


# =============================================================================
# 便捷函式（符合任務規格的函式簽名）
# =============================================================================

_default_loader: Optional[DataLoader] = None

def _get_loader() -> DataLoader:
    """取得預設載入器實例"""
    global _default_loader
    if _default_loader is None:
        _default_loader = DataLoader()
    return _default_loader


def load_prices(path: str = None) -> pd.DataFrame:
    """
    載入股價歷史數據

    Args:
        path: CSV 檔案路徑

    Returns:
        DataFrame，必含 columns: date, code, close
    """
    return _get_loader().load_prices(path)


def load_mapping(path: str = None) -> pd.DataFrame:
    """
    載入股票代號-公司名對照

    Args:
        path: CSV 檔案路徑

    Returns:
        DataFrame，包含 code, name 欄位
    """
    return _get_loader().load_mapping(path)


def load_industry(path: str = None) -> pd.DataFrame:
    """
    載入產業分類

    Args:
        path: XLSX 檔案路徑

    Returns:
        DataFrame，code -> bucket ∈ {ELECT, FIN, NON}
    """
    return _get_loader().load_industry(path)


def load_leverage_table(path: str = None) -> pd.DataFrame:
    """
    載入槓桿倍數對照表

    Args:
        path: XLSX 檔案路徑

    Returns:
        DataFrame，code/類別 -> leverage
    """
    return _get_loader().load_leverage_table(path)


def load_etf_weights(path: str = None) -> Dict[str, pd.DataFrame]:
    """
    載入 ETF 成分股權重

    Args:
        path: XLSX 檔案路徑

    Returns:
        Dict，{'0050': df, '0056': df}
    """
    return _get_loader().load_etf_weights(path)


def load_positions(xlsx_path_or_buffer) -> pd.DataFrame:
    """
    載入部位清單

    Args:
        xlsx_path_or_buffer: Excel 路徑或上傳緩衝區

    Returns:
        DataFrame，包含部位資訊
    """
    return _get_loader().load_positions(xlsx_path_or_buffer)
