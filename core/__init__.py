# =============================================================================
# 遠期契約保證金模擬平台 - Core 模組
# =============================================================================

from .data_loader import DataLoader
from .instruments import InstrumentClassifier, ETFLookthrough
from .buckets import BucketClassifier, ReturnCalculator
from .margin import MarginCalculator, MarginResult
from .engine import BacktestEngine, DailyResult
from .reporting import ReportGenerator, AuditPackage

__all__ = [
    'DataLoader',
    'InstrumentClassifier',
    'ETFLookthrough',
    'BucketClassifier',
    'ReturnCalculator',
    'MarginCalculator',
    'MarginResult',
    'BacktestEngine',
    'DailyResult',
    'ReportGenerator',
    'AuditPackage',
]
