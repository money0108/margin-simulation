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
def load_sample_positions(_loader):
    """載入示範部位（快取）"""
    try:
        positions = _loader.load_positions()
        return positions, None
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

        # 8. 缺碼清單
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
        | 股票期貨標的 | 5x |
        | 0050/0056 成份股 | 4x |
        | 其他股票 | 3x |
        | 0050/0056 ETF | 7x |

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
        st.error(f"資料載入失敗：{error}")
        st.stop()

    st.session_state.data_loader = loader

    # ==========================================================================
    # 側邊欄設定
    # ==========================================================================
    st.sidebar.header("⚙️ 參數設定")

    # 部位上傳
    st.sidebar.subheader("1️⃣ 部位資料")

    upload_option = st.sidebar.radio(
        "選擇部位來源",
        ["載入示範部位", "上傳 Excel 檔案"]
    )

    positions = None

    if upload_option == "載入示範部位":
        if st.sidebar.button("載入示範部位"):
            positions, error = load_sample_positions(loader)
            if error:
                st.sidebar.error(f"載入失敗：{error}")
            else:
                st.session_state.positions = positions
                st.sidebar.success(f"已載入 {len(positions)} 筆部位")

    else:
        uploaded_file = st.sidebar.file_uploader(
            "上傳部位 Excel",
            type=['xlsx', 'xls'],
            help="欄位：代號、買進張數、賣出張數"
        )

        if uploaded_file is not None:
            try:
                positions = loader.load_positions(uploaded_file)
                st.session_state.positions = positions
                st.sidebar.success(f"已載入 {len(positions)} 筆部位")
            except Exception as e:
                st.sidebar.error(f"檔案解析失敗：{e}")

    # 使用 session state 中的部位
    if st.session_state.positions is not None:
        positions = st.session_state.positions

    # 日期選擇
    st.sidebar.subheader("2️⃣ 日期設定")

    trading_dates, error = get_trading_dates(loader)
    if error:
        st.sidebar.error(f"取得交易日失敗：{error}")
        trading_dates = []

    if trading_dates:
        # 預設建倉日：最近 60 個交易日前
        default_start_idx = max(0, len(trading_dates) - 60)
        default_end_idx = len(trading_dates) - 1

        start_date = st.sidebar.date_input(
            "建倉日期",
            value=trading_dates[default_start_idx].date(),
            min_value=trading_dates[0].date(),
            max_value=trading_dates[-1].date()
        )

        end_date = st.sidebar.date_input(
            "結束日期",
            value=trading_dates[default_end_idx].date(),
            min_value=trading_dates[0].date(),
            max_value=trading_dates[-1].date()
        )

    # 顯示選項
    st.sidebar.subheader("3️⃣ 顯示選項")

    show_etf_lookthrough = st.sidebar.checkbox("顯示 ETF look-through 明細", value=False)
    show_reduction_detail = st.sidebar.checkbox("顯示折減來源分解", value=True)

    # ==========================================================================
    # 執行回測
    # ==========================================================================
    st.sidebar.subheader("4️⃣ 執行")

    if st.sidebar.button("🚀 開始模擬", type="primary", use_container_width=True):
        if positions is None or len(positions) == 0:
            st.error("請先載入部位資料")
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
    if positions is not None and len(positions) > 0:
        st.header("📋 部位清單")

        col1, col2, col3 = st.columns(3)
        with col1:
            long_count = len(positions[positions['side'] == 'LONG'])
            st.metric("多方部位", f"{long_count} 檔")
        with col2:
            short_count = len(positions[positions['side'] == 'SHORT'])
            st.metric("空方部位", f"{short_count} 檔")
        with col3:
            st.metric("總部位數", f"{len(positions)} 筆")

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

            tab1, tab2, tab3 = st.tabs(["IM/MM/Equity", "市值變化", "折減分解"])

            with tab1:
                fig = create_timeseries_chart(ts)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig = create_mv_chart(ts)
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                if show_reduction_detail:
                    fig = create_reduction_chart(ts)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("請在側邊欄勾選「顯示折減來源分解」")

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
                          'margin_call_flag', 'Required_Deposit', 'Equity', 'MM_today']
            equity_df = display_df[[c for c in equity_cols if c in display_df.columns]].copy()

            # 格式化金額
            money_cols_1 = ['Long_MV', 'Short_MV',
                           'Daily_PnL_Long', 'Daily_PnL_Short', 'Daily_PnL',
                           'Cum_PnL_Long', 'Cum_PnL_Short', 'Cumulative_PnL',
                           'Equity_Before', 'MM_At_Call', 'IM_today', 'Required_Deposit',
                           'Equity', 'MM_today']
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
                'Equity': '權益(補後)', 'MM_today': 'MM(補後)'
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
