# =============================================================================
# 遠期契約保證金模擬平台 - Streamlit 主程式
# 版本：v1.0.0
# 功能：上傳部位、選擇建倉日期、回放逐日結果、輸出稽核包
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile
import os

# 載入核心模組
from core.data_loader import DataLoader
from core.engine import BacktestEngine, BacktestResults
from core.reporting import ReportGenerator, verify
from core.data_loader import DataLoader as _DL  # for compute_position_diffs

# =============================================================================
# 頁面設定
# =============================================================================
st.set_page_config(
    page_title="遠期契約保證金模擬平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 樣式設定
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 初始化 Session State
# =============================================================================
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'positions' not in st.session_state:
    st.session_state.positions = None
if 'position_schedule' not in st.session_state:
    st.session_state.position_schedule = None
if 'position_diffs' not in st.session_state:
    st.session_state.position_diffs = None
if 'multi_mode' not in st.session_state:
    st.session_state.multi_mode = False


# =============================================================================
# 輔助函式
# =============================================================================
@st.cache_resource
def init_data_loader():
    """初始化資料載入器（快取）"""
    try:
        loader = DataLoader()
        return loader, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def get_trading_dates(_loader):
    """取得交易日列表（快取）"""
    try:
        prices = _loader.load_prices()
        dates = sorted(prices['date'].unique())
        return dates, None
    except Exception as e:
        return [], str(e)


def estimate_mv(positions_df: pd.DataFrame, prices_df: pd.DataFrame,
                as_of_date=None) -> dict:
    """
    用部位 + 股價估算多空市值摘要

    Returns:
        dict with long_mv, short_mv, long_count, short_count, total_count
    """
    if prices_df is None or len(prices_df) == 0:
        return None

    # 取最近可用日期的價格
    if as_of_date is not None:
        mask = prices_df['date'] <= pd.Timestamp(as_of_date)
        if mask.any():
            latest_date = prices_df.loc[mask, 'date'].max()
        else:
            latest_date = prices_df['date'].min()
    else:
        latest_date = prices_df['date'].max()

    price_map = prices_df[prices_df['date'] == latest_date].set_index('code')['close'].to_dict()

    long_mv = 0.0
    short_mv = 0.0
    long_count = 0
    short_count = 0
    missing = []

    for _, row in positions_df.iterrows():
        code = str(row['code']).strip()
        qty = float(row['qty'])
        price = price_map.get(code)
        if price is None:
            missing.append(code)
            continue
        mv = qty * price
        if row['side'] == 'LONG':
            long_mv += mv
            long_count += 1
        else:
            short_mv += mv
            short_count += 1

    return {
        'long_mv': long_mv, 'short_mv': short_mv,
        'long_count': long_count, 'short_count': short_count,
        'total_count': long_count + short_count,
        'missing': missing,
    }


def load_prices_from_upload(uploaded_file):
    """從上傳的文件載入股價數據"""
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        # 標準化欄位名稱
        col_map = {'日期': 'date', '股票代號': 'code', '收盤價': 'close'}
        for std_name, orig_name in col_map.items():
            if orig_name in df.columns:
                df = df.rename(columns={orig_name: std_name})

        # 日期格式轉換
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
        df['code'] = df['code'].astype(str).str.strip()
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['date', 'code', 'close'])
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, str(e)


def create_timeseries_chart(df: pd.DataFrame) -> go.Figure:
    """建立 IM/Equity/MM 時序圖"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('IM/MM/Equity 時序', '槓桿倍數時序'),
        row_heights=[0.6, 0.4]
    )

    # 上圖：IM, MM, Equity
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['IM_today'], name='IM_today',
                  line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['MM_today'], name='MM_today (70%)',
                  line=dict(color='#ff7f0e', width=2, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['Equity'], name='Equity',
                  line=dict(color='#2ca02c', width=2)),
        row=1, col=1
    )

    # 標記追繳日
    margin_call_df = df[df['margin_call_flag'] == 1]
    if len(margin_call_df) > 0:
        fig.add_trace(
            go.Scatter(x=margin_call_df['date'], y=margin_call_df['Equity'],
                      mode='markers', name='追繳觸發',
                      marker=dict(color='red', size=12, symbol='x')),
            row=1, col=1
        )

    # 下圖：槓桿
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['Gross_Lev'], name='Gross槓桿(有折減)',
                  line=dict(color='#9467bd', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['Raw_Lev'], name='無折減槓桿',
                  line=dict(color='#8c564b', width=2, dash='dot')),
        row=2, col=1
    )

    # 部位變動日標記紫色虛線
    if 'position_change_flag' in df.columns:
        change_dates = df[df['position_change_flag'] == 1]['date']
        for cd in change_dates:
            cd_str = cd.isoformat() if hasattr(cd, 'isoformat') else str(cd)
            for row_idx in [1, 2]:
                fig.add_shape(
                    type="line", x0=cd_str, x1=cd_str, y0=0, y1=1,
                    yref="paper" if row_idx == 1 else f"y{row_idx} domain",
                    line=dict(dash="dash", color="purple", width=1.5),
                    row=row_idx, col=1
                )
            fig.add_annotation(
                x=cd_str, y=1, yref="paper",
                text="加減倉", showarrow=False,
                font=dict(color="purple", size=10),
                xanchor="left", yanchor="bottom"
            )

    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    fig.update_yaxes(title_text='金額 (TWD)', row=1, col=1)
    fig.update_yaxes(title_text='槓桿倍數', row=2, col=1)
    fig.update_xaxes(title_text='日期', row=2, col=1)

    return fig


def create_reduction_chart(df: pd.DataFrame) -> go.Figure:
    """建立折減來源堆疊圖"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['reduction_etf_100'],
        name='ETF 100% 折減',
        marker_color='#1f77b4'
    ))

    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['reduction_same_bucket'],
        name='同桶折減',
        marker_color='#ff7f0e'
    ))

    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['reduction_cross_bucket'],
        name='跨桶折減',
        marker_color='#2ca02c'
    ))

    fig.update_layout(
        barmode='stack',
        title='折減來源分解',
        xaxis_title='日期',
        yaxis_title='折減金額 (TWD)',
        height=400
    )

    return fig


def create_mv_chart(df: pd.DataFrame) -> go.Figure:
    """建立 MV 時序圖"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['Long_MV'],
        name='Long MV',
        fill='tozeroy',
        line=dict(color='#2ca02c')
    ))

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=-df['Short_MV'],  # 負值顯示在下方
        name='Short MV',
        fill='tozeroy',
        line=dict(color='#d62728')
    ))

    fig.update_layout(
        title='多空市值時序',
        xaxis_title='日期',
        yaxis_title='市值 (TWD)',
        height=400
    )

    return fig


def _generate_hedge_sections(results: BacktestResults) -> str:
    """生成多空配對與減收明細的 HTML 區塊（建倉日 + 追繳日）"""
    if not results.daily_results:
        return '<p>無配對明細</p>'

    sections = []

    # 收集需要顯示的日期：建倉日 + 所有追繳日
    display_dates = []

    # 建倉日
    first_result = results.daily_results[0]
    display_dates.append(('建倉日', first_result))

    # 追繳日
    for dr in results.daily_results[1:]:
        if dr.margin_result.margin_call:
            date_str = dr.date.strftime('%Y-%m-%d')
            display_dates.append((f'追繳日 {date_str}', dr))

    for label, dr in display_dates:
        hedge_df = dr.margin_result.hedge_pairing_df
        mr = dr.margin_result
        date_str = dr.date.strftime('%Y-%m-%d')

        section_html = f'''
        <div style="margin-bottom: 30px; padding: 15px; background: #f9f9f9; border-radius: 8px;">
            <h3 style="color: #1f77b4; margin-top: 0;">{label} - {date_str}</h3>
        '''

        if len(hedge_df) > 0:
            # 摘要統計
            paired_count = len(hedge_df[hedge_df['配對MV'] > 0])
            total_hedged = hedge_df['配對MV'].sum()
            total_reduction = hedge_df['減收IM'].sum()

            section_html += f'''
            <div class="summary-grid" style="margin-bottom: 15px;">
                <div class="metric-card" style="display: inline-block; margin-right: 15px; min-width: 120px;">
                    <div class="metric-value">{paired_count}</div>
                    <div class="metric-label">配對標的數</div>
                </div>
                <div class="metric-card" style="display: inline-block; margin-right: 15px; min-width: 120px;">
                    <div class="metric-value">{total_hedged:,.0f}</div>
                    <div class="metric-label">總配對MV</div>
                </div>
                <div class="metric-card" style="display: inline-block; margin-right: 15px; min-width: 120px;">
                    <div class="metric-value">{total_reduction:,.0f}</div>
                    <div class="metric-label">總減收IM</div>
                </div>
                <div class="metric-card" style="display: inline-block; margin-right: 15px; min-width: 120px;">
                    <div class="metric-value">{mr.im_today:,.0f}</div>
                    <div class="metric-label">當日IM</div>
                </div>
            </div>
            '''

            # 格式化配對明細表
            hedge_display = hedge_df.copy()
            for col in hedge_display.columns:
                if col not in ['代號', '方向', '產業桶', '配對類型', '減收率']:
                    hedge_display[col] = hedge_display[col].apply(
                        lambda x: f'{x:,.0f}' if pd.notna(x) and isinstance(x, (int, float)) else x
                    )

            section_html += f'''
            <div style="max-height: 300px; overflow-y: auto;">
                {hedge_display.to_html(index=False, classes='data-table', escape=False)}
            </div>
            '''

            # 折減來源分解
            section_html += f'''
            <p style="margin-top: 15px;"><strong>折減來源分解：</strong></p>
            <div style="display: flex; gap: 20px;">
                <div>ETF完全對沖(100%): <strong>{mr.reduction_etf_100:,.0f}</strong></div>
                <div>同桶對沖: <strong>{mr.reduction_same_bucket:,.0f}</strong></div>
                <div>跨桶對沖: <strong>{mr.reduction_cross_bucket:,.0f}</strong></div>
            </div>
            '''
        else:
            section_html += '<p style="color: #666;">無多空配對</p>'

        section_html += '</div>'
        sections.append(section_html)

    return '\n'.join(sections)


def _generate_position_change_html(results: BacktestResults) -> str:
    """生成部位變動資訊的 HTML 區塊"""
    if not results.position_change_events:
        return ''

    html = '<h2>📦 部位變動紀錄</h2>'

    # 時間線
    html += '<div style="margin-bottom:20px;">'
    html += '<table><tr><th>日期</th><th>新 IM</th><th>新 MM</th><th>變動時權益</th><th>多方 MV</th><th>空方 MV</th><th>實現損益</th><th>出金</th></tr>'
    for evt in results.position_change_events:
        html += f"<tr><td>{evt['date'].strftime('%Y-%m-%d')}</td>"
        html += f"<td>{evt['new_im']:,.0f}</td>"
        html += f"<td>{evt['new_mm']:,.0f}</td>"
        html += f"<td>{evt['equity_at_change']:,.0f}</td>"
        html += f"<td>{evt['long_mv']:,.0f}</td>"
        html += f"<td>{evt['short_mv']:,.0f}</td>"
        html += f"<td>{evt.get('realized_pnl', 0):+,.0f}</td>"
        html += f"<td>{evt.get('withdrawal', 0):,.0f}</td></tr>"
    html += '</table></div>'

    return html


def create_html_report(results: BacktestResults, positions: pd.DataFrame) -> str:
    """建立完整 HTML 報告（可獨立開啟）"""
    ts = results.timeseries_df

    # 建立圖表
    fig1 = create_timeseries_chart(ts)
    fig2 = create_mv_chart(ts)
    fig3 = create_reduction_chart(ts)

    chart1_html = fig1.to_html(full_html=False, include_plotlyjs='cdn')
    chart2_html = fig2.to_html(full_html=False, include_plotlyjs=False)
    chart3_html = fig3.to_html(full_html=False, include_plotlyjs=False)

    # 摘要數據
    first_day = ts.iloc[0]
    last_day = ts.iloc[-1]

    # 權益損益表
    equity_cols = ['date', 'Long_MV', 'Short_MV',
                  'Daily_PnL_Long', 'Daily_PnL_Short', 'Daily_PnL',
                  'Cum_PnL_Long', 'Cum_PnL_Short', 'Cumulative_PnL',
                  'Equity_Before', 'MM_At_Call', 'IM_today',
                  'margin_call_flag', 'Required_Deposit', 'Equity', 'MM_today']
    equity_df = ts[[c for c in equity_cols if c in ts.columns]].copy()
    equity_df['date'] = equity_df['date'].dt.strftime('%Y-%m-%d')

    # 格式化數字欄位
    for col in equity_df.columns:
        if col != 'date' and col != 'margin_call_flag':
            equity_df[col] = equity_df[col].apply(lambda x: f'{x:,.0f}' if pd.notna(x) else '')

    equity_df.columns = ['日期', '多方MV', '空方MV', '多方日損益', '空方日損益', '合計日損益',
                        '多方累計', '空方累計', '合計累計', '權益(判定)', 'MM(判定)', 'IM',
                        '追繳', '追繳金額', '權益(補後)', 'MM(補後)'][:len(equity_df.columns)]

    # 保證金明細表
    margin_cols = ['date', 'Base_IM_long', 'Base_IM_short', 'IM_big', 'IM_small_before',
                  'reduction_etf_100', 'reduction_same_bucket', 'reduction_cross_bucket',
                  'total_reduction', 'IM_small_after', 'IM_today', 'Gross_Lev', 'Raw_Lev']
    margin_df = ts[[c for c in margin_cols if c in ts.columns]].copy()
    margin_df['date'] = margin_df['date'].dt.strftime('%Y-%m-%d')

    # 格式化數字欄位
    for col in margin_df.columns:
        if col == 'date':
            continue
        elif col in ['Gross_Lev', 'Raw_Lev']:
            margin_df[col] = margin_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '')
        else:
            margin_df[col] = margin_df[col].apply(lambda x: f'{x:,.0f}' if pd.notna(x) else '')

    margin_df.columns = ['日期', '多方Base_IM', '空方Base_IM', 'IM大邊', 'IM小邊(折前)',
                        'ETF折減', '同桶折減', '跨桶折減', '總折減',
                        'IM小邊(折後)', 'IM_today', 'Gross槓桿', '無折減槓桿'][:len(margin_df.columns)]

    # 融資費用表
    financing_cols = ['date', 'Long_MV', 'Short_MV', 'IM_today',
                     'Long_Financing', 'Short_Financing', 'Financing_Amount',
                     'Daily_Interest', 'Cumulative_Interest',
                     'Daily_Broker_Profit', 'Cumulative_Broker_Profit']
    financing_df = ts[[c for c in financing_cols if c in ts.columns]].copy()
    financing_df['date'] = financing_df['date'].dt.strftime('%Y-%m-%d')

    for col in financing_df.columns:
        if col != 'date':
            financing_df[col] = financing_df[col].apply(lambda x: f'{x:,.0f}' if pd.notna(x) else '')

    financing_df.columns = ['日期', '多方MV', '空方MV', 'IM',
                           '多方融資', '空方融資', '總融資',
                           '當日利息', '累計利息', '當日券商收益', '累計券商收益'][:len(financing_df.columns)]

    # 多空配對明細
    hedge_df = results.daily_results[0].margin_result.hedge_pairing_df if results.daily_results else pd.DataFrame()

    # 生成 HTML
    html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>遠期契約保證金模擬報告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft JhengHei", sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #1f77b4; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; border-left: 4px solid #1f77b4; padding-left: 10px; }}
        .summary-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin: 20px 0;
        }}
        .metric-card {{
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1f77b4; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{
            width: 100%; border-collapse: collapse; background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0;
        }}
        th, td {{ padding: 10px; text-align: right; border: 1px solid #ddd; font-size: 13px; }}
        th {{ background: #1f77b4; color: white; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #e8f4fc; }}
        td:first-child, th:first-child {{ text-align: left; }}
        .table-wrapper {{ max-height: 400px; overflow-y: auto; margin: 20px 0; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; }}
        .info {{ background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 15px; margin: 10px 0; }}
        .footer {{ text-align: center; color: #666; margin-top: 40px; padding: 20px; border-top: 1px solid #ddd; }}
        .negative {{ color: #dc3545; }}
        .positive {{ color: #28a745; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 遠期契約保證金模擬報告</h1>
        <p style="color:#666;">產出時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>📋 模擬摘要</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-value">{first_day['date'].strftime('%Y-%m-%d') if hasattr(first_day['date'], 'strftime') else first_day['date']}</div>
                <div class="metric-label">建倉日期</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{last_day['date'].strftime('%Y-%m-%d') if hasattr(last_day['date'], 'strftime') else last_day['date']}</div>
                <div class="metric-label">結束日期</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(ts)}</div>
                <div class="metric-label">交易日數</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{first_day['IM_today']:,.0f}</div>
                <div class="metric-label">建倉日 IM</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{last_day['IM_today']:,.0f}</div>
                <div class="metric-label">最新 IM</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{last_day['Equity']:,.0f}</div>
                <div class="metric-label">最新權益</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'negative' if last_day['Cumulative_PnL'] < 0 else 'positive'}">{last_day['Cumulative_PnL']:+,.0f}</div>
                <div class="metric-label">累計損益</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{int(ts['margin_call_flag'].sum())}</div>
                <div class="metric-label">追繳次數</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{last_day['Financing_Amount']:,.0f}</div>
                <div class="metric-label">融資金額</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{last_day['Cumulative_Interest']:,.0f}</div>
                <div class="metric-label">累計利息支出</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{last_day['Cumulative_Broker_Profit']:,.0f}</div>
                <div class="metric-label">券商累計收益</div>
            </div>
        </div>

        <h2>📈 IM / MM / 權益走勢</h2>
        <div class="chart-container">{chart1_html}</div>

        <h2>📊 多空市值走勢</h2>
        <div class="chart-container">{chart2_html}</div>

        <h2>📉 折減來源分解</h2>
        <div class="chart-container">{chart3_html}</div>

        <h2>💰 權益與損益追蹤</h2>
        <div class="table-wrapper">
            {equity_df.to_html(index=False, classes='data-table', escape=False)}
        </div>

        <h2>🧮 保證金計算明細</h2>
        <div class="table-wrapper">
            {margin_df.to_html(index=False, classes='data-table', escape=False)}
        </div>

        <h2>💳 融資費用明細</h2>
        <p style="color:#666;font-size:13px;">多方融資 = 多方MV - IM｜空方融資 = 空方MV（借券賣出不繳保證金）｜客戶利率 3%｜券商收益 = 多方融資×1.2% + 空方融資×3%｜利息按日曆日計算</p>
        <div class="table-wrapper">
            {financing_df.to_html(index=False, classes='data-table', escape=False)}
        </div>

        <h2>🔄 多空配對與減收明細</h2>
        {_generate_hedge_sections(results)}

        <h2>📝 假設與說明</h2>
        <div class="info">
            <ul>
                {''.join(f'<li>{a}</li>' for a in results.assumptions)}
            </ul>
        </div>

        {f'<div class="warning"><strong>⚠️ 追繳事件：</strong>共 {len(results.margin_call_events)} 次追繳</div>' if results.margin_call_events else ''}

        {_generate_position_change_html(results)}

        <div class="footer">
            <p>遠期契約保證金模擬平台 v1.0</p>
            <p>此報告由系統自動產生，僅供參考</p>
        </div>
    </div>
</body>
</html>'''

    return html


def create_full_report_excel(results: BacktestResults, positions: pd.DataFrame) -> bytes:
    """建立完整報告 Excel（多工作表）"""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. 摘要
        ts = results.timeseries_df
        if len(ts) > 0:
            first_day = ts.iloc[0]
            last_day = ts.iloc[-1]
            summary_data = {
                '項目': [
                    '模擬期間', '交易日數', '建倉日IM', '最新IM',
                    '建倉日MM', '最新MM', '最新權益', '累計損益',
                    '追繳次數', '平均Gross槓桿', '平均無折減槓桿'
                ],
                '數值': [
                    f"{first_day['date'].strftime('%Y-%m-%d') if hasattr(first_day['date'], 'strftime') else first_day['date']} ~ {last_day['date'].strftime('%Y-%m-%d') if hasattr(last_day['date'], 'strftime') else last_day['date']}",
                    len(ts),
                    f"{first_day['IM_today']:,.0f}",
                    f"{last_day['IM_today']:,.0f}",
                    f"{first_day['MM_today']:,.0f}",
                    f"{last_day['MM_today']:,.0f}",
                    f"{last_day['Equity']:,.0f}",
                    f"{last_day['Cumulative_PnL']:,.0f}",
                    int(ts['margin_call_flag'].sum()),
                    f"{ts['Gross_Lev'].mean():.2f}x",
                    f"{ts['Raw_Lev'].mean():.2f}x"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='摘要', index=False)

        # 2. 權益與損益追蹤
        if len(ts) > 0:
            equity_cols = ['date', 'Long_MV', 'Short_MV',
                          'Daily_PnL_Long', 'Daily_PnL_Short', 'Daily_PnL',
                          'Cum_PnL_Long', 'Cum_PnL_Short', 'Cumulative_PnL',
                          'Equity', 'MM_today', 'IM_today', 'margin_call_flag', 'Required_Deposit']
            equity_df = ts[[c for c in equity_cols if c in ts.columns]].copy()
            equity_df.columns = ['日期', '多方MV', '空方MV',
                                '多方日損益', '空方日損益', '合計日損益',
                                '多方累計', '空方累計', '合計累計',
                                '權益', 'MM', 'IM', '追繳', '追繳金額'][:len(equity_df.columns)]
            equity_df.to_excel(writer, sheet_name='權益損益追蹤', index=False)

        # 3. 保證金計算明細
        if len(ts) > 0:
            margin_cols = ['date', 'Base_IM_long', 'Base_IM_short', 'IM_big', 'IM_small_before',
                          'reduction_etf_100', 'reduction_same_bucket', 'reduction_cross_bucket',
                          'total_reduction', 'IM_small_after', 'IM_today', 'Gross_Lev', 'Raw_Lev']
            margin_df = ts[[c for c in margin_cols if c in ts.columns]].copy()
            margin_df.columns = ['日期', '多方Base_IM', '空方Base_IM', 'IM大邊', 'IM小邊(折前)',
                                'ETF折減', '同桶折減', '跨桶折減', '總折減',
                                'IM小邊(折後)', 'IM_today', 'Gross槓桿', '無折減槓桿'][:len(margin_df.columns)]
            margin_df.to_excel(writer, sheet_name='保證金計算明細', index=False)

        # 4. 建倉日多空配對
        if results.daily_results:
            hedge_df = results.daily_results[0].margin_result.hedge_pairing_df
            if len(hedge_df) > 0:
                hedge_df.to_excel(writer, sheet_name='多空配對明細', index=False)

        # 5. 部位清單
        positions.to_excel(writer, sheet_name='部位清單', index=False)

        # 6. 追繳事件
        if results.margin_call_events:
            events_df = pd.DataFrame(results.margin_call_events)
            events_df.to_excel(writer, sheet_name='追繳事件', index=False)

        # 7. 假設說明
        assumptions_df = pd.DataFrame({
            '假設說明': results.assumptions
        })
        assumptions_df.to_excel(writer, sheet_name='假設說明', index=False)

        # 8. 部位變動事件
        if results.position_change_events:
            change_df = pd.DataFrame(results.position_change_events)
            change_df.to_excel(writer, sheet_name='部位變動事件', index=False)

        # 9. 缺碼清單
        if results.missing_codes:
            missing_df = pd.DataFrame({
                '缺碼代號': results.missing_codes
            })
            missing_df.to_excel(writer, sheet_name='缺碼清單', index=False)

    output.seek(0)
    return output.getvalue()


def create_audit_zip(results: BacktestResults, positions: pd.DataFrame) -> bytes:
    """建立稽核包 ZIP"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 最終時序
        csv_buffer = io.StringIO()
        results.timeseries_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        zf.writestr('final_timeseries.csv', csv_buffer.getvalue().encode('utf-8-sig'))

        # 部位快照
        csv_buffer = io.StringIO()
        positions.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        zf.writestr('inputs_snapshot/positions.csv', csv_buffer.getvalue().encode('utf-8-sig'))

        # 追繳事件
        if results.margin_call_events:
            csv_buffer = io.StringIO()
            pd.DataFrame(results.margin_call_events).to_csv(csv_buffer, index=False)
            zf.writestr('margin_call_events.csv', csv_buffer.getvalue().encode('utf-8-sig'))

        # 假設說明
        assumptions_content = "# 假設與保守口徑說明\n\n"
        for assumption in results.assumptions:
            assumptions_content += f"- {assumption}\n"
        if results.missing_codes:
            assumptions_content += f"\n## 缺碼清單（共 {len(results.missing_codes)} 檔）\n"
            for code in results.missing_codes[:50]:
                assumptions_content += f"- {code}\n"
        zf.writestr('assumptions.md', assumptions_content.encode('utf-8'))

        # 驗證結果
        verification = verify(results)
        import json
        zf.writestr('verification.json', json.dumps(verification, ensure_ascii=False, indent=2).encode('utf-8'))

        # 部位變動事件
        if results.position_change_events:
            change_df = pd.DataFrame(results.position_change_events)
            csv_buffer = io.StringIO()
            change_df.to_csv(csv_buffer, index=False)
            zf.writestr('position_change_events.csv', csv_buffer.getvalue().encode('utf-8-sig'))

        # 多期部位快照
        if results.position_schedule and len(results.position_schedule) > 1:
            for s_idx, (s_date, s_df) in enumerate(results.position_schedule):
                csv_buffer = io.StringIO()
                s_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                zf.writestr(
                    f'inputs_snapshot/positions_{s_date.strftime("%Y%m%d")}.csv',
                    csv_buffer.getvalue().encode('utf-8-sig')
                )

        # 逐日明細（首/中/末日）
        if results.daily_results:
            n = len(results.daily_results)
            sample_indices = [0, n // 2, n - 1] if n >= 3 else list(range(n))

            for idx in sample_indices:
                dr = results.daily_results[idx]
                date_str = dr.date.strftime('%Y%m%d')
                mr = dr.margin_result

                csv_buffer = io.StringIO()
                mr.summary_df.to_csv(csv_buffer, index=False)
                zf.writestr(f'calc_steps/{date_str}_summary.csv', csv_buffer.getvalue().encode('utf-8-sig'))

                csv_buffer = io.StringIO()
                mr.bucket_hedge_df.to_csv(csv_buffer, index=False)
                zf.writestr(f'calc_steps/{date_str}_bucket_hedge.csv', csv_buffer.getvalue().encode('utf-8-sig'))

                csv_buffer = io.StringIO()
                mr.reduction_breakdown_df.to_csv(csv_buffer, index=False)
                zf.writestr(f'calc_steps/{date_str}_reduction_breakdown.csv', csv_buffer.getvalue().encode('utf-8-sig'))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# =============================================================================
# 主程式
# =============================================================================
def main():
    # 標題
    st.markdown('<p class="main-header">遠期契約保證金模擬平台</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Forward Contract Margin Simulation Platform v1.0</p>', unsafe_allow_html=True)

    # 制度摘要
    with st.expander("📋 制度口徑一句話摘要", expanded=False):
        st.info("""
        **本制度以固定槓桿計算分邊 Base IM，並以 Base IM 判定大小邊；**
        對沖折減僅適用於小邊，依三產業桶與 3M 加權累積報酬率決定折減率（50% 或 20%）；
        0050/0056 ETF 採 look-through，成份股完全對沖部分可 100% 減收；
        維持保證金為當日 IM 的 70%，跌破維持保證金時需追繳回補至當日 IM（100%）。
        """)

        st.markdown("""
        **關鍵規則清單：**
        | 類別 | 槓桿倍數 |
        |------|---------|
        | ETF（月均量 ≥ 10,000 張） | 7x |
        | ETF（月均量 < 10,000 張） | 5x |
        | 股期標的（股期槓桿 > 7） | 5x |
        | 股期標的（股期槓桿 > 6 且 ≤ 7） | 4x |
        | 股期標的（股期槓桿 ≤ 6）/ 其他股票 | 3x |

        | 對沖類型 | 折減率 |
        |---------|-------|
        | 同桶（3M 報酬差 ≥ 10%）| 50% |
        | 同桶（3M 報酬差 < 10%）| 20% |
        | 跨桶 | 20% |
        | ETF 完全對沖 | 100% |
        """)

    # 初始化資料載入器
    loader, error = init_data_loader()
    if error:
        st.error(f"資料載入器初始化失敗：{error}")
        st.stop()

    st.session_state.data_loader = loader

    # ==========================================================================
    # 側邊欄設定
    # ==========================================================================
    st.sidebar.header("⚙️ 參數設定")

    # 數據來源設定
    st.sidebar.subheader("0️⃣ 股價數據")

    # 初始化 session state
    if 'prices_df' not in st.session_state:
        st.session_state.prices_df = None
    if 'trading_dates' not in st.session_state:
        st.session_state.trading_dates = []

    # 嘗試從雲端載入，如果失敗則讓用戶上傳
    trading_dates, price_error = get_trading_dates(loader)

    if price_error:
        st.sidebar.warning("雲端數據載入失敗，請上傳股價檔案")
        price_file = st.sidebar.file_uploader(
            "上傳股價 CSV",
            type=['csv'],
            help="必須包含欄位：日期、股票代號、收盤價",
            key="price_upload"
        )
        if price_file is not None:
            prices_df, err = load_prices_from_upload(price_file)
            if err:
                st.sidebar.error(f"解析失敗：{err}")
            else:
                st.session_state.prices_df = prices_df
                st.session_state.trading_dates = sorted(prices_df['date'].unique())
                # 將股價數據注入到 DataLoader
                loader.set_prices_df(prices_df)
                st.sidebar.success(f"已載入 {len(prices_df)} 筆股價數據")
                trading_dates = st.session_state.trading_dates
    else:
        st.sidebar.success("✓ 股價數據已載入")
        st.session_state.trading_dates = trading_dates

    # 如果之前已經上傳過股價，確保 DataLoader 也有這份數據
    if st.session_state.prices_df is not None:
        loader.set_prices_df(st.session_state.prices_df)

    # 部位上傳（統一介面：支援單檔或多檔）
    st.sidebar.subheader("1️⃣ 部位資料")

    uploaded_files = st.sidebar.file_uploader(
        "上傳部位 Excel（可多選）",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        help="單檔＝初始建倉；多檔＝最早日期為建倉，其餘為加倉/平倉。檔案第一列若為「日期 + 日期值」會自動偵測，否則需手動指定日期。",
        key="position_upload"
    )

    positions = None

    if uploaded_files and len(uploaded_files) > 0:
        # 解析每個檔案，檢查是否有日期標頭
        parsed_files = []
        needs_date = []

        for idx, uf in enumerate(uploaded_files):
            try:
                raw = pd.read_excel(uf, header=None)
                cell_00 = str(raw.iloc[0, 0]).strip() if len(raw) > 0 else ''

                if cell_00 == '日期':
                    date_val = raw.iloc[0, 1]
                    pos_date = pd.Timestamp(date_val)
                    parsed_files.append({'file': uf, 'name': uf.name, 'date': pos_date, 'has_date': True})
                else:
                    parsed_files.append({'file': uf, 'name': uf.name, 'date': None, 'has_date': False})
                    needs_date.append(idx)
            except Exception as e:
                st.sidebar.error(f"解析 {uf.name} 失敗：{e}")

        # 對缺日期的檔案要求手動輸入
        for idx in needs_date:
            pf = parsed_files[idx]
            user_date = st.sidebar.date_input(
                f"指定 {pf['name']} 的日期",
                key=f"pos_date_{idx}"
            )
            pf['date'] = pd.Timestamp(user_date)

        # 所有檔案都有日期後，載入
        all_have_dates = all(pf['date'] is not None for pf in parsed_files)

        if all_have_dates and len(parsed_files) > 0:
            try:
                files_list = []
                fallback_dates = []
                for pf in parsed_files:
                    pf['file'].seek(0)
                    files_list.append(pf['file'])
                    fallback_dates.append(pf['date'])

                schedule = loader.load_multi_positions(files_list, fallback_dates)
                st.session_state.position_schedule = schedule
                st.session_state.positions = schedule[0][1]  # 第一期部位

                if len(schedule) > 1:
                    st.session_state.multi_mode = True
                    diffs = _DL.compute_position_diffs(schedule)
                    st.session_state.position_diffs = diffs
                    st.sidebar.success(f"已載入 {len(schedule)} 期部位")
                    for s_date, s_df in schedule:
                        st.sidebar.caption(f"  {s_date.strftime('%Y-%m-%d')}: {len(s_df)} 筆")
                else:
                    st.session_state.multi_mode = False
                    st.session_state.position_diffs = None
                    st.sidebar.success(f"已載入 {len(schedule[0][1])} 筆部位（建倉日 {schedule[0][0].strftime('%Y-%m-%d')}）")

            except Exception as e:
                st.sidebar.error(f"部位載入失敗：{e}")
                import traceback
                st.sidebar.code(traceback.format_exc())

    # 使用 session state 中的部位
    if st.session_state.positions is not None:
        positions = st.session_state.positions

    # 日期自動決定：建倉日 = schedule 第一個快照日期，結束日 = 股價最後一天
    trading_dates = st.session_state.trading_dates

    if st.session_state.position_schedule and len(st.session_state.position_schedule) > 0:
        start_date = st.session_state.position_schedule[0][0].date()
    elif len(trading_dates) > 0:
        default_start_idx = max(0, len(trading_dates) - 60)
        start_date = trading_dates[default_start_idx].date()
    else:
        start_date = None

    end_date = trading_dates[-1].date() if len(trading_dates) > 0 else None

    # 顯示選項
    st.sidebar.subheader("2️⃣ 顯示選項")

    show_etf_lookthrough = st.sidebar.checkbox("顯示 ETF look-through 明細", value=False)
    show_reduction_detail = st.sidebar.checkbox("顯示折減來源分解", value=True)

    # ==========================================================================
    # 執行回測
    # ==========================================================================
    st.sidebar.subheader("3️⃣ 執行")

    if st.sidebar.button("🚀 開始模擬", type="primary", use_container_width=True):
        if positions is None or len(positions) == 0:
            st.error("請先上傳部位檔案")
        elif start_date is None or end_date is None:
            st.error("股價數據尚未載入，無法決定模擬期間")
        else:
            try:
                engine = BacktestEngine(loader)

                # 取得交易日數以顯示進度
                calc_dates = engine.get_trading_dates_range(
                    pd.Timestamp(start_date), pd.Timestamp(end_date)
                )
                total_days = len(calc_dates)

                # 建立進度條
                progress_bar = st.progress(0, text="正在初始化...")
                status_text = st.empty()

                # 使用進度回調執行回測
                def progress_callback(current, total, date_str):
                    pct = current / total if total > 0 else 0
                    progress_bar.progress(pct, text=f"計算中... {current}/{total} ({pct:.0%})")
                    status_text.text(f"正在計算 {date_str}")

                # 統一走 position_schedule 路徑
                if st.session_state.position_schedule:
                    results = engine.run(
                        position_schedule=st.session_state.position_schedule,
                        start_date=pd.Timestamp(start_date),
                        end_date=pd.Timestamp(end_date),
                        progress_callback=progress_callback
                    )
                else:
                    results = engine.run(
                        positions=positions,
                        start_date=pd.Timestamp(start_date),
                        end_date=pd.Timestamp(end_date),
                        progress_callback=progress_callback
                    )

                progress_bar.progress(1.0, text="計算完成！")
                status_text.empty()
                st.session_state.backtest_results = results
                st.success(f"計算完成！共處理 {len(results.daily_results)} 個交易日")
            except Exception as e:
                st.error(f"計算失敗：{e}")
                import traceback
                st.code(traceback.format_exc())

    # ==========================================================================
    # 顯示部位
    # ==========================================================================
    # 取得股價（用於估算市值）
    try:
        _prices_for_mv = loader.load_prices()
    except Exception:
        _prices_for_mv = None

    if positions is not None and len(positions) > 0:
        st.header("📋 部位清單")

        # 多期模式
        if st.session_state.multi_mode and st.session_state.position_schedule and len(st.session_state.position_schedule) > 1:
            schedule = st.session_state.position_schedule

            # --- 部位變動摘要 ---
            diffs = st.session_state.position_diffs

            # 建立 date → snapshot df 的快速查找
            schedule_map = {pd.Timestamp(s_date): s_df for s_date, s_df in schedule}

            st.subheader("部位變動摘要")
            summary_records = []

            # 先加建倉日（schedule[0]）
            s0_date, s0_df = schedule[0]
            mv0 = estimate_mv(s0_df, _prices_for_mv, as_of_date=s0_date)
            summary_records.append({
                '日期': s0_date.strftime('%Y-%m-%d'),
                '類型': '建倉',
                '多方市值': f"{mv0['long_mv']:,.0f}" if mv0 else '-',
                '空方市值': f"{mv0['short_mv']:,.0f}" if mv0 else '-',
                '淨市值': f"{mv0['long_mv'] - mv0['short_mv']:+,.0f}" if mv0 else '-',
                '實現損益': '-', '出金': '-',
                '新增': '-', '平倉': '-', '加倉': '-',
                '減倉': '-', '翻倉': '-', '變動筆數': '-',
            })

            # 若有回測結果，建立 date → event 查找（含實現損益/出金）
            _bt_results = st.session_state.backtest_results
            _evt_map = {}
            if _bt_results and _bt_results.position_change_events:
                for evt in _bt_results.position_change_events:
                    _evt_map[pd.Timestamp(evt['date'])] = evt

            # 各次變動
            if diffs:
                for d in diffs:
                    type_counts = d['diff_df']['變動類型'].value_counts().to_dict() if len(d['diff_df']) > 0 else {}
                    # 找該日期對應的 snapshot 算市值
                    snap_df = schedule_map.get(pd.Timestamp(d['date']))
                    mv = estimate_mv(snap_df, _prices_for_mv, as_of_date=d['date']) if snap_df is not None else None
                    # 從回測結果取實現損益/出金
                    evt = _evt_map.get(pd.Timestamp(d['date']))
                    summary_records.append({
                        '日期': d['date'].strftime('%Y-%m-%d'),
                        '類型': '加減倉',
                        '多方市值': f"{mv['long_mv']:,.0f}" if mv else '-',
                        '空方市值': f"{mv['short_mv']:,.0f}" if mv else '-',
                        '淨市值': f"{mv['long_mv'] - mv['short_mv']:+,.0f}" if mv else '-',
                        '實現損益': f"{evt['realized_pnl']:+,.0f}" if evt else '-',
                        '出金': f"{evt['withdrawal']:,.0f}" if evt else '-',
                        '新增': type_counts.get('新增', 0),
                        '平倉': type_counts.get('平倉', 0),
                        '加倉': type_counts.get('加倉', 0),
                        '減倉': type_counts.get('減倉', 0),
                        '翻倉': type_counts.get('翻倉', 0),
                        '變動筆數': len(d['diff_df']),
                    })

            st.dataframe(pd.DataFrame(summary_records), use_container_width=True, hide_index=True)

            # --- 各期部位 tabs ---
            tab_labels = []
            for t_idx, (s_date, s_df) in enumerate(schedule):
                label = f"{s_date.strftime('%Y-%m-%d')}"
                if t_idx == 0:
                    label += " (建倉)"
                tab_labels.append(label)

            pos_tabs = st.tabs(tab_labels)

            for t_idx, (s_date, s_df) in enumerate(schedule):
                with pos_tabs[t_idx]:
                    mv_info = estimate_mv(s_df, _prices_for_mv, as_of_date=s_date)
                    if mv_info:
                        c1, c2, c3, c4, c5 = st.columns(5)
                        with c1:
                            st.metric("多方", f"{mv_info['long_count']} 檔")
                        with c2:
                            st.metric("多方市值", f"{mv_info['long_mv']:,.0f}")
                        with c3:
                            st.metric("空方", f"{mv_info['short_count']} 檔")
                        with c4:
                            st.metric("空方市值", f"{mv_info['short_mv']:,.0f}")
                        with c5:
                            net = mv_info['long_mv'] - mv_info['short_mv']
                            st.metric("淨市值", f"{net:+,.0f}")
                    else:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("多方", f"{len(s_df[s_df['side']=='LONG'])} 檔")
                        with c2:
                            st.metric("空方", f"{len(s_df[s_df['side']=='SHORT'])} 檔")
                        with c3:
                            st.metric("總部位", f"{len(s_df)} 筆")

                    # 若有對應的差異資料，顯示該期變動
                    if diffs and t_idx > 0 and t_idx - 1 < len(diffs):
                        diff_df = diffs[t_idx - 1]['diff_df']
                        if len(diff_df) > 0:
                            st.caption("與前期差異：")
                            st.dataframe(diff_df, use_container_width=True, height=180, hide_index=True)

                    st.dataframe(s_df, use_container_width=True, height=250)
        else:
            # 單期模式
            as_of = start_date if start_date else None
            mv_info = estimate_mv(positions, _prices_for_mv, as_of_date=as_of)
            if mv_info:
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.metric("多方", f"{mv_info['long_count']} 檔")
                with c2:
                    st.metric("多方市值", f"{mv_info['long_mv']:,.0f}")
                with c3:
                    st.metric("空方", f"{mv_info['short_count']} 檔")
                with c4:
                    st.metric("空方市值", f"{mv_info['short_mv']:,.0f}")
                with c5:
                    net = mv_info['long_mv'] - mv_info['short_mv']
                    st.metric("淨市值", f"{net:+,.0f}")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("多方", f"{len(positions[positions['side']=='LONG'])} 檔")
                with c2:
                    st.metric("空方", f"{len(positions[positions['side']=='SHORT'])} 檔")
                with c3:
                    st.metric("總部位", f"{len(positions)} 筆")

            with st.expander("查看部位明細", expanded=False):
                st.dataframe(positions, use_container_width=True)

    # ==========================================================================
    # 顯示結果
    # ==========================================================================
    if st.session_state.backtest_results is not None:
        results = st.session_state.backtest_results

        st.header("📊 模擬結果")

        # 驗證狀態
        if not results.verification_passed:
            st.markdown('<div class="error-card">', unsafe_allow_html=True)
            st.error("⚠️ 驗證發現問題")
            for err in results.verification_errors:
                st.write(f"- {err}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.success("✅ 所有驗證通過")

        # 核心指標
        ts = results.timeseries_df

        if len(ts) > 0:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "最新 IM",
                    f"{ts['IM_today'].iloc[-1]:,.0f}",
                    delta=f"{ts['IM_today'].iloc[-1] - ts['IM_today'].iloc[0]:,.0f}"
                )

            with col2:
                st.metric(
                    "最新 MM (70%)",
                    f"{ts['MM_today'].iloc[-1]:,.0f}"
                )

            with col3:
                margin_call_count = ts['margin_call_flag'].sum()
                st.metric(
                    "追繳次數",
                    f"{margin_call_count}",
                    delta="需注意" if margin_call_count > 0 else None,
                    delta_color="inverse"
                )

            with col4:
                st.metric(
                    "平均 Gross 槓桿",
                    f"{ts['Gross_Lev'].mean():.2f}x"
                )

            # 圖表
            st.subheader("📈 時序圖表")

            # 決定是否顯示部位變動頁籤
            has_pos_changes = (
                st.session_state.position_diffs is not None
                and len(st.session_state.position_diffs) > 0
            )

            tab_names = ["IM/MM/Equity", "市值變化", "折減分解"]
            if has_pos_changes:
                tab_names.append("部位變動")
                tab_names.append("資金流向")

            chart_tabs = st.tabs(tab_names)

            with chart_tabs[0]:
                fig = create_timeseries_chart(ts)
                st.plotly_chart(fig, use_container_width=True)

            with chart_tabs[1]:
                fig = create_mv_chart(ts)
                st.plotly_chart(fig, use_container_width=True)

            with chart_tabs[2]:
                if show_reduction_detail:
                    fig = create_reduction_chart(ts)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("請在側邊欄勾選「顯示折減來源分解」")

            if has_pos_changes:
                with chart_tabs[3]:
                    st.markdown("### 部位變動時間線")

                    # 變動時間線表格
                    schedule = st.session_state.position_schedule
                    # 建立 event 查找
                    _change_evt_map = {}
                    if results.position_change_events:
                        for evt in results.position_change_events:
                            _change_evt_map[pd.Timestamp(evt['date'])] = evt

                    timeline_records = []
                    for t_idx, (s_date, s_df) in enumerate(schedule):
                        long_count = len(s_df[s_df['side'] == 'LONG'])
                        short_count = len(s_df[s_df['side'] == 'SHORT'])
                        evt = _change_evt_map.get(pd.Timestamp(s_date))
                        rec = {
                            '快照日期': s_date.strftime('%Y-%m-%d'),
                            '多方數量': long_count,
                            '空方數量': short_count,
                            '總部位數': len(s_df),
                        }
                        if t_idx == 0:
                            rec['新 IM'] = f"{results.timeseries_df.iloc[0]['IM_today']:,.0f}" if len(results.timeseries_df) > 0 else '-'
                            rec['變動時權益'] = '-'
                            rec['實現損益'] = '-'
                            rec['出金'] = '-'
                        elif evt:
                            rec['新 IM'] = f"{evt['new_im']:,.0f}"
                            rec['變動時權益'] = f"{evt['equity_at_change']:,.0f}"
                            rec['實現損益'] = f"{evt['realized_pnl']:+,.0f}"
                            rec['出金'] = f"{evt['withdrawal']:,.0f}"
                        else:
                            rec['新 IM'] = '-'
                            rec['變動時權益'] = '-'
                            rec['實現損益'] = '-'
                            rec['出金'] = '-'
                        timeline_records.append(rec)

                    timeline_df = pd.DataFrame(timeline_records)
                    st.dataframe(timeline_df, use_container_width=True, hide_index=True)

                    # 逐期差異表
                    st.markdown("### 逐期差異明細")
                    diffs = st.session_state.position_diffs
                    diff_tab_names = [
                        f"{d['prev_date'].strftime('%m/%d')} → {d['date'].strftime('%m/%d')}"
                        for d in diffs
                    ]
                    diff_tabs = st.tabs(diff_tab_names)

                    for d_idx, diff_info in enumerate(diffs):
                        with diff_tabs[d_idx]:
                            diff_df = diff_info['diff_df']
                            st.caption(
                                f"從 {diff_info['prev_date'].strftime('%Y-%m-%d')} "
                                f"到 {diff_info['date'].strftime('%Y-%m-%d')} 的變動"
                            )
                            if len(diff_df) > 0:
                                # 變動類型摘要
                                type_counts = diff_df['變動類型'].value_counts()
                                summary_cols = st.columns(min(len(type_counts), 6))
                                for ci, (change_type, count) in enumerate(type_counts.items()):
                                    with summary_cols[ci % len(summary_cols)]:
                                        st.metric(change_type, f"{count} 筆")

                                st.dataframe(diff_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("無部位變動")

                with chart_tabs[4]:
                    st.markdown("### 資金流向分析")

                    if results.position_change_events:
                        # 建立子頁籤：每次變動一個
                        flow_tab_names = []
                        for evt_idx, evt in enumerate(results.position_change_events):
                            evt_date_str = evt['date'].strftime('%m/%d')
                            # 找前一個日期
                            if evt_idx == 0:
                                schedule = st.session_state.position_schedule
                                prev_date_str = schedule[0][0].strftime('%m/%d')
                            else:
                                prev_date_str = results.position_change_events[evt_idx - 1]['date'].strftime('%m/%d')
                            flow_tab_names.append(f"{prev_date_str} → {evt_date_str}")

                        flow_tabs = st.tabs(flow_tab_names)

                        for evt_idx, evt in enumerate(results.position_change_events):
                            with flow_tabs[evt_idx]:
                                evt_date_str = evt['date'].strftime('%Y-%m-%d')

                                # --- 區塊 A：變動前後比較表 ---
                                st.markdown(f"#### 變動前後比較（{evt_date_str}）")

                                old_long_mv = evt.get('old_long_mv', 0)
                                old_short_mv = evt.get('old_short_mv', 0)
                                new_long_mv = evt.get('long_mv', 0)
                                new_short_mv = evt.get('short_mv', 0)
                                old_im = evt.get('old_im', 0)
                                new_im = evt.get('new_im', 0)
                                old_mm = evt.get('old_mm', 0)
                                new_mm = evt.get('new_mm', 0)

                                compare_data = {
                                    '項目': ['多方 MV', '空方 MV', 'IM', 'MM'],
                                    '變動前': [
                                        f"{old_long_mv:,.0f}",
                                        f"{old_short_mv:,.0f}",
                                        f"{old_im:,.0f}",
                                        f"{old_mm:,.0f}",
                                    ],
                                    '變動後': [
                                        f"{new_long_mv:,.0f}",
                                        f"{new_short_mv:,.0f}",
                                        f"{new_im:,.0f}",
                                        f"{new_mm:,.0f}",
                                    ],
                                    '差異': [
                                        f"{new_long_mv - old_long_mv:+,.0f}",
                                        f"{new_short_mv - old_short_mv:+,.0f}",
                                        f"{new_im - old_im:+,.0f}",
                                        f"{new_mm - old_mm:+,.0f}",
                                    ],
                                }
                                st.dataframe(
                                    pd.DataFrame(compare_data),
                                    use_container_width=True,
                                    hide_index=True
                                )

                                # --- 區塊 B：資金流向瀑布 ---
                                st.markdown("#### 資金流向")

                                equity_before = evt.get('equity_before_change', 0)
                                realized_pnl = evt.get('realized_pnl', 0)
                                withdrawal = evt.get('withdrawal', 0)
                                equity_after = evt.get('equity_at_change', 0)

                                # 可出金金額計算公式
                                excess_over_im = max(0, equity_before - new_im)
                                max_withdrawal = min(realized_pnl, excess_over_im) if realized_pnl > 0 else 0

                                flow_md = f"""
| 步驟 | 說明 | 金額 |
|------|------|-----:|
| 1 | 變動前權益（舊部位以當日價格結算） | **{equity_before:,.0f}** |
| 2 | 實現損益（平/減倉部位按基準價差計算） | **{realized_pnl:+,.0f}** |
| 3 | 可出金金額 = min(實現損益, max(0, 權益 - 新IM)) | **{max_withdrawal:,.0f}** |
| 4 | 實際出金 | **-{withdrawal:,.0f}** |
| 5 | 出金後權益 | **{equity_after:,.0f}** |
"""
                                st.markdown(flow_md)

                                # --- 區塊 B2：出金來源 — 逐部位實現損益明細 ---
                                pnl_details = evt.get('realized_pnl_details', [])
                                if pnl_details:
                                    st.markdown("#### 出金來源 — 逐部位實現損益明細")
                                    detail_records = []
                                    for d in pnl_details:
                                        side_label = '多' if d['side'] == 'LONG' else '空'
                                        detail_records.append({
                                            '代號': d['code'],
                                            '方向': side_label,
                                            '變動': d['change_type'],
                                            '原數量': f"{d['old_qty']:,}",
                                            '新數量': f"{d['new_qty']:,}",
                                            '平/減量': f"{d['closed_qty']:,}",
                                            '基準價': f"{d['base_price']:.2f}",
                                            '當日價': f"{d['current_price']:.2f}",
                                            '價差': f"{d['current_price'] - d['base_price']:+.2f}" if d['side'] == 'LONG' else f"{d['base_price'] - d['current_price']:+.2f}",
                                            '實現損益': f"{d['pnl']:+,.0f}",
                                        })
                                    detail_df = pd.DataFrame(detail_records)
                                    st.dataframe(detail_df, use_container_width=True, hide_index=True)

                                    # 小計
                                    total_pnl = sum(d['pnl'] for d in pnl_details)
                                    profit_count = sum(1 for d in pnl_details if d['pnl'] > 0)
                                    loss_count = sum(1 for d in pnl_details if d['pnl'] < 0)
                                    profit_sum = sum(d['pnl'] for d in pnl_details if d['pnl'] > 0)
                                    loss_sum = sum(d['pnl'] for d in pnl_details if d['pnl'] < 0)

                                    sc1, sc2, sc3, sc4 = st.columns(4)
                                    with sc1:
                                        st.metric("獲利部位", f"{profit_count} 檔", delta=f"{profit_sum:+,.0f}")
                                    with sc2:
                                        st.metric("虧損部位", f"{loss_count} 檔", delta=f"{loss_sum:+,.0f}", delta_color="inverse")
                                    with sc3:
                                        st.metric("實現損益合計", f"{total_pnl:+,.0f}")
                                    with sc4:
                                        st.metric("實際出金", f"{withdrawal:,.0f}")
                                elif realized_pnl == 0:
                                    st.caption("本次變動無平倉/減倉部位，無實現損益。")

                                # --- 區塊 C：融資金額變化 ---
                                st.markdown("#### 融資金額變化")

                                old_long_fin = evt.get('old_long_financing', 0)
                                old_short_fin = evt.get('old_short_financing', 0)
                                new_long_fin = evt.get('new_long_financing', 0)
                                new_short_fin = evt.get('new_short_financing', 0)
                                old_total_fin = old_long_fin + old_short_fin
                                new_total_fin = new_long_fin + new_short_fin

                                fin_data = {
                                    '項目': ['多方融資', '空方融資', '總融資'],
                                    '變動前': [
                                        f"{old_long_fin:,.0f}",
                                        f"{old_short_fin:,.0f}",
                                        f"{old_total_fin:,.0f}",
                                    ],
                                    '變動後': [
                                        f"{new_long_fin:,.0f}",
                                        f"{new_short_fin:,.0f}",
                                        f"{new_total_fin:,.0f}",
                                    ],
                                    '差異': [
                                        f"{new_long_fin - old_long_fin:+,.0f}",
                                        f"{new_short_fin - old_short_fin:+,.0f}",
                                        f"{new_total_fin - old_total_fin:+,.0f}",
                                    ],
                                    '說明': [
                                        '多方MV - 帳上資金',
                                        '借券全額',
                                        '',
                                    ],
                                }
                                st.dataframe(
                                    pd.DataFrame(fin_data),
                                    use_container_width=True,
                                    hide_index=True
                                )

                                st.info(
                                    "出金使帳上資金減少，券商需多融出資金以維持多方市值。"
                                    "空方融資 = 借券市值，隨部位MV變動。"
                                )

                                # --- 區塊 D：規則說明 ---
                                with st.expander("計算規則說明"):
                                    st.markdown("""
- **IM 重算邏輯**：部位變動後，以新部位重新計算 IM（含分邊、大小邊、對沖折減）
- **MM = 新 IM x 70%**：部位變動後 MM 重置為新 IM 的 70%
- **出金規則**：僅就「已平倉/減倉部位的實現損益」可出金，且出金後權益不低於新 IM
  - 可出金金額 = min(實現損益, max(0, 權益 - 新IM))
- **融資公式**：
  - 多方融資 = max(0, 多方MV - 帳上資金)
  - 空方融資 = 空方MV（借券賣出全額融資）
- **追繳**：若出金後權益 < MM → 立即追繳至新 IM
""")
                    else:
                        st.info("無部位變動事件")

            # 逐日明細表 - 拆成兩個表
            st.subheader("📋 逐日明細")

            # 格式化顯示
            display_df = ts.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

            # ===== 表1: 權益與損益追蹤 =====
            st.markdown("**表1：權益與損益追蹤**")
            equity_cols = ['date', 'Long_MV', 'Short_MV',
                          'Daily_PnL_Long', 'Daily_PnL_Short', 'Daily_PnL',
                          'Cum_PnL_Long', 'Cum_PnL_Short', 'Cumulative_PnL',
                          'Equity_Before', 'MM_At_Call', 'IM_today',
                          'margin_call_flag', 'Required_Deposit', 'Withdrawal', 'Equity', 'MM_today']
            equity_df = display_df[[c for c in equity_cols if c in display_df.columns]].copy()

            # 格式化金額
            money_cols_1 = ['Long_MV', 'Short_MV',
                           'Daily_PnL_Long', 'Daily_PnL_Short', 'Daily_PnL',
                           'Cum_PnL_Long', 'Cum_PnL_Short', 'Cumulative_PnL',
                           'Equity_Before', 'MM_At_Call', 'IM_today', 'Required_Deposit',
                           'Withdrawal', 'Equity', 'MM_today']
            for col in money_cols_1:
                if col in equity_df.columns:
                    equity_df[col] = equity_df[col].apply(lambda x: f"{x:,.0f}")

            # 重命名欄位
            equity_df = equity_df.rename(columns={
                'date': '日期', 'Long_MV': '多方MV', 'Short_MV': '空方MV',
                'Daily_PnL_Long': '多方日損益', 'Daily_PnL_Short': '空方日損益', 'Daily_PnL': '合計日損益',
                'Cum_PnL_Long': '多方累計', 'Cum_PnL_Short': '空方累計', 'Cumulative_PnL': '合計累計',
                'Equity_Before': '權益(判定)', 'MM_At_Call': 'MM(判定)', 'IM_today': 'IM',
                'margin_call_flag': '追繳', 'Required_Deposit': '追繳金額',
                'Withdrawal': '出金', 'Equity': '權益(補後)', 'MM_today': 'MM(補後)'
            })

            st.dataframe(equity_df, use_container_width=True, height=300)

            # ===== 表2: 保證金計算明細 =====
            st.markdown("**表2：保證金計算明細**")
            margin_cols = ['date', 'Base_IM_long', 'Base_IM_short', 'IM_big', 'IM_small_before',
                          'reduction_etf_100', 'reduction_same_bucket', 'reduction_cross_bucket',
                          'total_reduction', 'IM_small_after', 'IM_today', 'Gross_Lev', 'Raw_Lev']
            margin_df = display_df[[c for c in margin_cols if c in display_df.columns]].copy()

            # 格式化金額
            money_cols_2 = ['Base_IM_long', 'Base_IM_short', 'IM_big', 'IM_small_before',
                           'reduction_etf_100', 'reduction_same_bucket', 'reduction_cross_bucket',
                           'total_reduction', 'IM_small_after', 'IM_today']
            for col in money_cols_2:
                if col in margin_df.columns:
                    margin_df[col] = margin_df[col].apply(lambda x: f"{x:,.0f}")

            # 重命名欄位
            margin_df = margin_df.rename(columns={
                'date': '日期', 'Base_IM_long': '多方Base_IM', 'Base_IM_short': '空方Base_IM',
                'IM_big': 'IM大邊', 'IM_small_before': 'IM小邊(折前)',
                'reduction_etf_100': 'ETF折減', 'reduction_same_bucket': '同桶折減',
                'reduction_cross_bucket': '跨桶折減', 'total_reduction': '總折減',
                'IM_small_after': 'IM小邊(折後)', 'IM_today': 'IM_today',
                'Gross_Lev': 'Gross槓桿', 'Raw_Lev': '無折減槓桿'
            })

            st.dataframe(margin_df, use_container_width=True, height=300)

            # ===== 表3: 融資費用明細 =====
            st.markdown("**表3：融資費用明細**")
            st.caption("多方融資 = 多方MV - IM｜空方融資 = 空方MV（借券賣出不繳保證金）｜客戶利率 3%｜券商收益 = 多方融資×1.2% + 空方融資×3%｜利息按日曆日計算")

            financing_cols = ['date', 'Long_MV', 'Short_MV', 'IM_today',
                             'Long_Financing', 'Short_Financing', 'Financing_Amount',
                             'Daily_Interest', 'Cumulative_Interest',
                             'Daily_Broker_Profit', 'Cumulative_Broker_Profit']
            financing_df = display_df[[c for c in financing_cols if c in display_df.columns]].copy()

            # 格式化金額
            money_cols_3 = ['Long_MV', 'Short_MV', 'IM_today', 'Long_Financing', 'Short_Financing',
                           'Financing_Amount', 'Daily_Interest', 'Cumulative_Interest',
                           'Daily_Broker_Profit', 'Cumulative_Broker_Profit']
            for col in money_cols_3:
                if col in financing_df.columns:
                    financing_df[col] = financing_df[col].apply(lambda x: f"{x:,.0f}")

            # 重命名欄位
            financing_df = financing_df.rename(columns={
                'date': '日期', 'Long_MV': '多方MV', 'Short_MV': '空方MV', 'IM_today': 'IM',
                'Long_Financing': '多方融資', 'Short_Financing': '空方融資',
                'Financing_Amount': '總融資',
                'Daily_Interest': '當日利息', 'Cumulative_Interest': '累計利息',
                'Daily_Broker_Profit': '當日券商收益', 'Cumulative_Broker_Profit': '累計券商收益'
            })

            st.dataframe(financing_df, use_container_width=True, height=300)

            # 融資摘要
            if len(ts) > 0:
                last_day = ts.iloc[-1]
                fin_col1, fin_col2, fin_col3, fin_col4 = st.columns(4)
                with fin_col1:
                    st.metric("最新融資金額", f"{last_day['Financing_Amount']:,.0f}")
                with fin_col2:
                    st.metric("累計利息支出", f"{last_day['Cumulative_Interest']:,.0f}")
                with fin_col3:
                    st.metric("券商累計收益", f"{last_day['Cumulative_Broker_Profit']:,.0f}")
                with fin_col4:
                    # 計算年化報酬率
                    days = len(ts)
                    if days > 0 and last_day['Financing_Amount'] > 0:
                        annualized_cost = (last_day['Cumulative_Interest'] / last_day['Financing_Amount']) * (365 / days) * 100
                        st.metric("年化融資成本", f"{annualized_cost:.2f}%")

            # 建倉日及追繳日多空配對與減收明細
            if results.daily_results:
                st.subheader("🔄 多空配對與減收明細")

                # 找出需要顯示的日期：建倉日 + 所有追繳日
                display_dates = []

                # 建倉日
                first_result = results.daily_results[0]
                display_dates.append(('建倉日', first_result))

                # 追繳日
                for dr in results.daily_results[1:]:
                    if dr.margin_result.margin_call:
                        date_str = dr.date.strftime('%Y-%m-%d')
                        display_dates.append((f'追繳日 {date_str}', dr))

                # 使用 tabs 顯示各日期的明細
                if len(display_dates) > 1:
                    tab_names = [d[0] for d in display_dates]
                    tabs = st.tabs(tab_names)

                    for i, (label, dr) in enumerate(display_dates):
                        with tabs[i]:
                            hedge_df = dr.margin_result.hedge_pairing_df
                            mr = dr.margin_result

                            st.markdown(f"**{label}** - {dr.date.strftime('%Y-%m-%d')}")

                            if len(hedge_df) > 0:
                                # 顯示摘要統計
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    paired_count = len(hedge_df[hedge_df['配對MV'] > 0])
                                    st.metric("配對標的數", f"{paired_count} 檔")
                                with col2:
                                    total_hedged = hedge_df['配對MV'].sum()
                                    st.metric("總配對MV", f"{total_hedged:,.0f}")
                                with col3:
                                    total_reduction = hedge_df['減收IM'].sum()
                                    st.metric("總減收IM", f"{total_reduction:,.0f}")
                                with col4:
                                    st.metric("當日IM", f"{mr.im_today:,.0f}")

                                # 顯示明細表
                                st.dataframe(hedge_df, use_container_width=True, height=250)

                                # 顯示折減來源分解
                                st.markdown("**折減來源分解：**")
                                breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
                                with breakdown_col1:
                                    st.metric("ETF完全對沖(100%)", f"{mr.reduction_etf_100:,.0f}")
                                with breakdown_col2:
                                    st.metric("同桶對沖", f"{mr.reduction_same_bucket:,.0f}")
                                with breakdown_col3:
                                    st.metric("跨桶對沖", f"{mr.reduction_cross_bucket:,.0f}")
                            else:
                                st.info("無多空配對")
                else:
                    # 只有建倉日，不需要 tabs
                    hedge_df = first_result.margin_result.hedge_pairing_df
                    mr = first_result.margin_result

                    st.markdown(f"**建倉日** - {first_result.date.strftime('%Y-%m-%d')}")

                    if len(hedge_df) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            paired_count = len(hedge_df[hedge_df['配對MV'] > 0])
                            st.metric("配對標的數", f"{paired_count} 檔")
                        with col2:
                            total_hedged = hedge_df['配對MV'].sum()
                            st.metric("總配對MV", f"{total_hedged:,.0f}")
                        with col3:
                            total_reduction = hedge_df['減收IM'].sum()
                            st.metric("總減收IM", f"{total_reduction:,.0f}")

                        st.dataframe(hedge_df, use_container_width=True, height=300)

                        st.markdown("**折減來源分解：**")
                        breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
                        with breakdown_col1:
                            st.metric("ETF完全對沖(100%)", f"{mr.reduction_etf_100:,.0f}")
                        with breakdown_col2:
                            st.metric("同桶對沖", f"{mr.reduction_same_bucket:,.0f}")
                        with breakdown_col3:
                            st.metric("跨桶對沖", f"{mr.reduction_cross_bucket:,.0f}")
                    else:
                        st.info("無多空配對")

            # ETF Look-through 明細
            if show_etf_lookthrough and results.daily_results:
                st.subheader("🔍 ETF Look-through 明細")
                first_result = results.daily_results[0]
                atom_df = first_result.margin_result.atom_detail_df

                if len(atom_df[atom_df['is_from_etf'] == True]) > 0:
                    etf_atoms = atom_df[atom_df['is_from_etf'] == True][
                        ['origin', 'parent', 'code', 'side', 'mv', 'leverage', 'base_im', 'bucket']
                    ].copy()
                    etf_atoms['mv'] = etf_atoms['mv'].apply(lambda x: f"{x:,.0f}")
                    etf_atoms['base_im'] = etf_atoms['base_im'].apply(lambda x: f"{x:,.0f}")
                    st.dataframe(etf_atoms, use_container_width=True)
                else:
                    st.info("目前部位無 ETF look-through 曝險")

            # 追繳事件
            if results.margin_call_events:
                st.subheader("⚠️ 追繳事件")
                events_df = pd.DataFrame(results.margin_call_events)
                events_df['date'] = pd.to_datetime(events_df['date']).dt.strftime('%Y-%m-%d')
                for col in ['im_today', 'mm_today', 'equity', 'required_deposit']:
                    events_df[col] = events_df[col].apply(lambda x: f"{x:,.0f}")
                st.dataframe(events_df, use_container_width=True)

            # 下載區
            st.subheader("📥 一鍵存檔")

            # HTML 報告（主要）
            html_report = create_html_report(results, positions)
            st.download_button(
                label="🌐 下載完整報告 (HTML)",
                data=html_report.encode('utf-8'),
                file_name=f"保證金模擬報告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True,
                type="primary"
            )
            st.caption("💡 HTML 報告可直接用瀏覽器開啟，包含所有圖表與數據，方便傳給他人檢視")

            # 其他格式
            with st.expander("其他匯出格式"):
                col1, col2 = st.columns(2)

                with col1:
                    csv_buffer = io.StringIO()
                    ts.to_csv(csv_buffer, index=False, encoding='utf-8-sig')

                    st.download_button(
                        label="📄 下載逐日明細 CSV",
                        data=csv_buffer.getvalue().encode('utf-8-sig'),
                        file_name=f"margin_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    zip_data = create_audit_zip(results, positions)

                    st.download_button(
                        label="📦 下載稽核包 ZIP",
                        data=zip_data,
                        file_name=f"audit_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

            # 假設說明
            with st.expander("📝 假設與保守口徑說明"):
                for assumption in results.assumptions:
                    st.write(f"- {assumption}")

                if results.missing_codes:
                    st.warning(f"⚠️ 發現 {len(results.missing_codes)} 檔缺碼，以保守口徑處理")
                    st.write("缺碼清單（前 20 檔）：")
                    st.code(", ".join(results.missing_codes[:20]))


if __name__ == "__main__":
    main()
