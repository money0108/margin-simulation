# =============================================================================
# 手動建倉模擬平台 - Streamlit 主程式
# 功能：手動選日期、輸入代號建倉（自動帶入收盤價），支援加倉/減倉
# 保證金計算規則 100% 與原平台相同
# =============================================================================

import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile

# 載入核心模組（共用原平台）
from core.data_loader import DataLoader
from core.engine import BacktestEngine, BacktestResults
from core.reporting import verify

# =============================================================================
# 頁面設定
# =============================================================================
st.set_page_config(
    page_title="手動建倉模擬平台",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .warning-card { background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; }
    .error-card { background-color: #f8d7da; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #dc3545; }
    .success-card { background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #28a745; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 初始化 Session State
# =============================================================================
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'prices_df' not in st.session_state:
    st.session_state.prices_df = None
if 'trading_dates' not in st.session_state:
    st.session_state.trading_dates = []
if 'schedule_entries' not in st.session_state:
    st.session_state.schedule_entries = []

# =============================================================================
# 輔助函式
# =============================================================================
@st.cache_resource
def init_data_loader():
    """初始化資料載入器（快取）"""
    try:
        config_path = str(Path(__file__).resolve().parent / "config" / "settings.yaml")
        loader = DataLoader(config_path=config_path)
        return loader, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def get_trading_dates_and_prices(_loader):
    """取得交易日列表與股價（快取）"""
    try:
        prices = _loader.load_prices()
        dates = sorted(prices['date'].unique())
        return prices, dates, None
    except Exception as e:
        return None, [], str(e)


def load_prices_from_upload(uploaded_file):
    """從上傳的文件載入股價數據"""
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        col_map = {'日期': 'date', '股票代號': 'code', '收盤價': 'close'}
        for orig_name, std_name in col_map.items():
            if orig_name in df.columns:
                df = df.rename(columns={orig_name: std_name})
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
        df['code'] = df['code'].astype(str).str.strip()
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['date', 'code', 'close'])
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, str(e)


def get_price_on_date(prices_df, code, date):
    """取得特定日期的收盤價"""
    ts = pd.Timestamp(date)
    mask = (prices_df['code'] == code) & (prices_df['date'] <= ts)
    subset = prices_df.loc[mask]
    if len(subset) == 0:
        return None
    latest = subset.loc[subset['date'].idxmax()]
    return float(latest['close'])


def classify_instrument(code, etf_codes=None):
    """判斷代號是 ETF 還是 STK"""
    if etf_codes is None:
        etf_codes = {'0050', '0056', '50', '56'}
    return 'ETF' if code in etf_codes else 'STK'


def schedule_entries_to_position_schedule(entries, contract_multiplier=1000):
    """將 session_state 的 schedule_entries 轉為 engine 需要的 position_schedule

    同一代碼若同時有 LONG + SHORT，自動淨額化（如 LONG 1000 + SHORT 2000 → SHORT 1000）。
    """
    schedule = []
    for entry in entries:
        rows = []
        for pos in entry['positions']:
            rows.append({
                'code': pos['code'],
                'side': pos['side'],
                'qty': pos['qty_lots'] * contract_multiplier,
                'instrument': pos['instrument'],
            })
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # --- 淨額化：同一代碼 LONG+SHORT 互抵 ---
        agg = df.groupby(['code', 'side'], as_index=False)['qty'].sum()
        inst_map = df.drop_duplicates('code').set_index('code')['instrument'].to_dict()
        netted = []
        for code in agg['code'].unique():
            cr = agg[agg['code'] == code]
            long_q = float(cr.loc[cr['side'] == 'LONG', 'qty'].sum())
            short_q = float(cr.loc[cr['side'] == 'SHORT', 'qty'].sum())
            net = long_q - short_q
            inst = inst_map.get(code, 'STK')
            if net > 0:
                netted.append({'code': code, 'side': 'LONG', 'qty': net, 'instrument': inst})
            elif net < 0:
                netted.append({'code': code, 'side': 'SHORT', 'qty': -net, 'instrument': inst})
        if netted:
            schedule.append((pd.Timestamp(entry['date']), pd.DataFrame(netted)))
    return schedule


# =============================================================================
# 圖表函式（與原平台相同）
# =============================================================================
def create_timeseries_chart(df):
    """建立 Equity/MM 時序圖 + 出入金與追繳標記"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=('Equity / MM 時序', '出入金 與 追繳標記'),
        row_heights=[0.6, 0.4]
    )

    # --- Row 1：Equity + MM ---
    fig.add_trace(go.Scatter(x=df['date'], y=df['MM_today'], name='MM (70%)',
                             line=dict(color='#ff7f0e', width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['Equity'], name='Equity',
                             line=dict(color='#2ca02c', width=2)), row=1, col=1)

    margin_call_df = df[df['margin_call_flag'] == 1]
    if len(margin_call_df) > 0:
        fig.add_trace(go.Scatter(x=margin_call_df['date'], y=margin_call_df['Equity'],
                                 mode='markers', name='追繳觸發',
                                 marker=dict(color='red', size=12, symbol='x')), row=1, col=1)

    # --- Row 2：出入金柱狀 + 追繳標記 ---
    if 'Initial_Deposit' in df.columns:
        mask_init = df['Initial_Deposit'] > 0
        if mask_init.any():
            fig.add_trace(go.Bar(x=df.loc[mask_init, 'date'], y=df.loc[mask_init, 'Initial_Deposit'],
                                 name='入金', marker_color='#2ca02c'), row=2, col=1)

    if 'Pos_Change_Deposit' in df.columns:
        mask_pos = df['Pos_Change_Deposit'] > 0
        if mask_pos.any():
            fig.add_trace(go.Bar(x=df.loc[mask_pos, 'date'], y=df.loc[mask_pos, 'Pos_Change_Deposit'],
                                 name='加減倉入金', marker_color='#1f77b4'), row=2, col=1)

    if 'Required_Deposit' in df.columns:
        mask_req = df['Required_Deposit'] > 0
        if mask_req.any():
            fig.add_trace(go.Bar(x=df.loc[mask_req, 'date'], y=df.loc[mask_req, 'Required_Deposit'],
                                 name='追繳入金', marker_color='#d62728'), row=2, col=1)

    if 'Withdrawal' in df.columns:
        mask_wdl = df['Withdrawal'] > 0
        if mask_wdl.any():
            fig.add_trace(go.Bar(x=df.loc[mask_wdl, 'date'], y=-df.loc[mask_wdl, 'Withdrawal'],
                                 name='出金', marker_color='#ff7f0e'), row=2, col=1)

    # 加減倉垂直線
    if 'position_change_flag' in df.columns:
        change_dates = df[df['position_change_flag'] == 1]['date']
        for cd in change_dates:
            cd_str = cd.isoformat() if hasattr(cd, 'isoformat') else str(cd)
            for row_idx in [1, 2]:
                fig.add_shape(type="line", x0=cd_str, x1=cd_str, y0=0, y1=1,
                              yref="paper" if row_idx == 1 else f"y{row_idx} domain",
                              line=dict(dash="dash", color="purple", width=1.5),
                              row=row_idx, col=1)
            fig.add_annotation(x=cd_str, y=1, yref="paper", text="加減倉",
                               showarrow=False, font=dict(color="purple", size=10),
                               xanchor="left", yanchor="bottom")

    fig.update_layout(height=600, showlegend=True, barmode='relative',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_yaxes(title_text='金額 (TWD)', row=1, col=1)
    fig.update_yaxes(title_text='金額 (TWD)', row=2, col=1)
    fig.update_xaxes(title_text='日期', row=2, col=1)
    return fig


def create_reduction_chart(df):
    """建立折減來源堆疊圖"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['date'], y=df['reduction_etf_100'],
                         name='ETF 100% 折減', marker_color='#1f77b4'))
    fig.add_trace(go.Bar(x=df['date'], y=df['reduction_same_bucket'],
                         name='同桶折減', marker_color='#ff7f0e'))
    fig.add_trace(go.Bar(x=df['date'], y=df['reduction_cross_bucket'],
                         name='跨桶折減', marker_color='#2ca02c'))
    fig.update_layout(barmode='stack', title='折減來源分解',
                      xaxis_title='日期', yaxis_title='折減金額 (TWD)', height=400)
    return fig


def create_mv_chart(df):
    """建立 MV 時序圖"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['Long_MV'], name='Long MV',
                             fill='tozeroy', line=dict(color='#2ca02c')))
    fig.add_trace(go.Scatter(x=df['date'], y=-df['Short_MV'], name='Short MV',
                             fill='tozeroy', line=dict(color='#d62728')))
    fig.update_layout(title='多空市值時序', xaxis_title='日期',
                      yaxis_title='市值 (TWD)', height=400)
    return fig


# =============================================================================
# HTML 報告生成
# =============================================================================
def _generate_hedge_sections(results):
    """生成多空配對與減收明細的 HTML 區塊"""
    if not results.daily_results:
        return '<p>無配對明細</p>'

    pos_change_dates = set()
    if results.position_change_events:
        for evt in results.position_change_events:
            pos_change_dates.add(pd.Timestamp(evt['date']))

    sections = []
    display_dates = []
    first_result = results.daily_results[0]
    display_dates.append(('建倉日', first_result))
    for dr in results.daily_results[1:]:
        date_str = dr.date.strftime('%Y-%m-%d')
        is_pos_change = dr.date in pos_change_dates
        is_margin_call = dr.margin_result.margin_call
        if is_pos_change and is_margin_call:
            display_dates.append((f'加減倉+追繳 {date_str}', dr))
        elif is_pos_change:
            display_dates.append((f'加減倉 {date_str}', dr))
        elif is_margin_call:
            display_dates.append((f'追繳日 {date_str}', dr))

    for label, dr in display_dates:
        hedge_df = dr.margin_result.hedge_pairing_df
        mr = dr.margin_result
        date_str = dr.date.strftime('%Y-%m-%d')
        section_html = f'<div style="margin-bottom:30px;padding:15px;background:#f9f9f9;border-radius:8px;"><h3 style="color:#1f77b4;margin-top:0;">{label} - {date_str}</h3>'
        if len(hedge_df) > 0:
            reduced_count = len(hedge_df[hedge_df['總折減'] > 0])
            total_reduction = hedge_df['總折減'].sum()
            section_html += f'''
            <div style="margin-bottom:15px;">
                <div class="metric-card" style="display:inline-block;margin-right:15px;min-width:120px;"><div class="metric-value">{reduced_count}</div><div class="metric-label">折減標的數</div></div>
                <div class="metric-card" style="display:inline-block;margin-right:15px;min-width:120px;"><div class="metric-value">{total_reduction:,.0f}</div><div class="metric-label">總折減IM</div></div>
                <div class="metric-card" style="display:inline-block;margin-right:15px;min-width:120px;"><div class="metric-value">{mr.im_today:,.0f}</div><div class="metric-label">當日IM</div></div>
            </div>'''
            hedge_display = hedge_df.copy()
            for col in hedge_display.columns:
                if col == '槓桿':
                    hedge_display[col] = hedge_display[col].apply(
                        lambda x: f'{x:.2f}' if pd.notna(x) and isinstance(x, (int, float)) else x)
                elif col not in ['代碼', '產業桶', '減收類型']:
                    hedge_display[col] = hedge_display[col].apply(
                        lambda x: f'{x:,.0f}' if pd.notna(x) and isinstance(x, (int, float)) else x)
            section_html += f'<div style="max-height:300px;overflow-y:auto;">{hedge_display.to_html(index=False, classes="data-table", escape=False)}</div>'
            section_html += f'''<p style="margin-top:15px;"><strong>折減來源分解：</strong></p>
            <div style="display:flex;gap:20px;">
                <div>ETF完全對沖(100%): <strong>{mr.reduction_etf_100:,.0f}</strong></div>
                <div>同桶對沖: <strong>{mr.reduction_same_bucket:,.0f}</strong></div>
                <div>跨桶對沖: <strong>{mr.reduction_cross_bucket:,.0f}</strong></div>
            </div>'''
            bkt_df = mr.bucket_hedge_df
            if bkt_df is not None and len(bkt_df) > 0:
                bkt_show = bkt_df[['產業桶', '同桶折減率', '3M報酬差', '可對沖比例', '折減來源']].copy()
                section_html += '<p style="margin-top:10px;"><strong>各桶折減率判定：</strong></p>'
                section_html += bkt_show.to_html(index=False, classes="data-table", escape=False)
        else:
            section_html += '<p style="color:#666;">無多空配對</p>'
        section_html += '</div>'
        sections.append(section_html)
    return '\n'.join(sections)


def _compute_deposit_column(ts, results):
    """計算入金欄位（共用邏輯）

    三種入金分開：
    - Initial_Deposit:    入金（建倉日的初始 IM）
    - Pos_Change_Deposit: 加減倉入金（加減倉時補足保證金差額）
    - Required_Deposit:   追繳入金（權益因市場波動跌破 MM）

    平倉現金流（加減倉日，賣出多方部位時）：
    - Sale_Net:           平倉淨額（賣出價金 - 證交稅）
    - Customer_Deposit:   客戶實際入金 = max(0, 加減倉入金 - 平倉淨額)
    - Sale_Surplus:       平倉餘額出金 = max(0, 平倉淨額 - 加減倉入金)
    """
    TAX_RATE = 0.003
    ts['Initial_Deposit'] = 0.0
    ts['Pos_Change_Deposit'] = 0.0
    ts['Sale_Net'] = 0.0
    ts['Customer_Deposit'] = 0.0
    ts['Sale_Surplus'] = 0.0
    # Required_Deposit 由 engine 提供，只在追繳時有值

    # 建倉日：入金 = 初始 IM
    ts.loc[ts.index[0], 'Initial_Deposit'] = ts.iloc[0]['IM_today']
    ts.loc[ts.index[0], 'Customer_Deposit'] = ts.iloc[0]['IM_today']

    # 加減倉日：補足至新 IM 的差額
    if results.position_change_events:
        margin_call_dates = set()
        if results.margin_call_events:
            for mc in results.margin_call_events:
                margin_call_dates.add(pd.Timestamp(mc['date']))

        for evt in results.position_change_events:
            evt_date = pd.Timestamp(evt['date'])
            new_im = evt.get('new_im', 0)
            equity_after = evt.get('equity_at_change', 0)
            deposit_needed = max(0, new_im - equity_after)
            mask = ts['date'] == evt_date
            if mask.any():
                # 加減倉日的補繳一律歸入加減倉入金，非市場追繳
                ts.loc[mask, 'Pos_Change_Deposit'] = deposit_needed
                ts.loc[mask, 'Required_Deposit'] = 0.0
                ts.loc[mask, 'margin_call_flag'] = 0
                # 加減倉日：判定 = 補後（入金與 IM 同日生效，保持一致）
                ts.loc[mask, 'Equity_Before'] = ts.loc[mask, 'Equity']
                ts.loc[mask, 'MM_At_Call'] = ts.loc[mask, 'MM_today']

                # 平倉現金流
                pnl_details = evt.get('realized_pnl_details', [])
                sp = sum(d['current_price'] * d['closed_qty']
                         for d in pnl_details if d['side'] == 'LONG')
                sp_tax = round(sp * TAX_RATE)
                sp_net = sp - sp_tax
                cust_dep = max(0, deposit_needed - sp_net)
                surplus = max(0, sp_net - deposit_needed) if deposit_needed > 0 else 0
                ts.loc[mask, 'Sale_Net'] = sp_net
                ts.loc[mask, 'Customer_Deposit'] = cust_dep
                ts.loc[mask, 'Sale_Surplus'] = surplus

    return ts


def _generate_cashflow_section(results, position_schedule):
    """生成出入金計算明細的 HTML 區塊"""
    if not results.daily_results:
        return '<p>無出入金明細</p>'

    ts = results.timeseries_df
    first_im = ts.iloc[0]['IM_today'] if len(ts) > 0 else 0
    build_date = results.daily_results[0].date

    # --- 收集所有資金事件 ---
    flow_events = []

    # 建倉入金
    flow_events.append({
        'date': build_date, 'type': '建倉入金',
        'in': first_im, 'out': 0,
        'desc': f'建倉日 IM = {first_im:,.0f}',
    })

    # 加減倉事件
    TAX_RATE = 0.003  # 證交稅 0.3%（賣出時課徵）
    _change_evt_map = {}
    for evt in (results.position_change_events or []):
        _change_evt_map[pd.Timestamp(evt['date'])] = evt
        equity_before = evt.get('equity_before_change', 0)
        equity_after_wdl = evt.get('equity_at_change', 0)
        new_im = evt.get('new_im', 0)
        realized_pnl = evt.get('realized_pnl', 0)
        withdrawal = evt.get('withdrawal', 0)
        deposit = max(0, new_im - equity_after_wdl)
        # 計算平倉賣出價金（僅平倉多方 = 賣出，產生現金）
        _pnl_details = evt.get('realized_pnl_details', [])
        sale_proceeds = sum(d['current_price'] * d['closed_qty'] for d in _pnl_details if d['side'] == 'LONG')
        sale_tax = round(sale_proceeds * TAX_RATE)
        sale_net = sale_proceeds - sale_tax
        customer_cash = max(0, deposit - sale_net)
        # sale_surplus 僅在需入金時有意義；出金時已含在 withdrawal
        sale_surplus = max(0, sale_net - deposit) if deposit > 0 else 0
        total_event_out = withdrawal + sale_surplus
        flow_events.append({
            'date': evt['date'], 'type': '加減倉',
            'in': deposit, 'out': total_event_out,
            'sale_net': sale_net, 'customer_cash': customer_cash,
            'sale_surplus': sale_surplus,
            'desc': f'實現損益 {realized_pnl:+,.0f} / 出金 {total_event_out:,.0f} / 入金 {deposit:,.0f}',
        })

    # 追繳事件
    for evt in (results.margin_call_events or []):
        dep = evt.get('required_deposit', 0)
        flow_events.append({
            'date': evt['date'], 'type': '追繳入金',
            'in': dep, 'out': 0,
            'desc': f'追繳金額 = 新IM - 追繳前權益 = {dep:,.0f}',
        })

    total_in = sum(e['in'] for e in flow_events)
    total_out = sum(e['out'] for e in flow_events)
    total_sale_net = sum(e.get('sale_net', 0) for e in flow_events)
    total_sale_surplus = sum(e.get('sale_surplus', 0) for e in flow_events)
    total_customer_cash = sum(e.get('customer_cash', e['in']) for e in flow_events)
    net_flow = total_in - total_out

    # --- HTML: 摘要 ---
    html = f'''
    <div style="margin-bottom:20px;">
        <div class="summary-grid">
            <div class="metric-card"><div class="metric-value">{total_in:,.0f}</div><div class="metric-label">總入金（帳面）</div></div>
            <div class="metric-card"><div class="metric-value">{total_sale_net:,.0f}</div><div class="metric-label">平倉淨額</div></div>
            <div class="metric-card"><div class="metric-value">{total_customer_cash:,.0f}</div><div class="metric-label">客戶實際入金</div></div>
            <div class="metric-card"><div class="metric-value">{total_out:,.0f}</div><div class="metric-label">總出金</div></div>
        </div>
    </div>
    '''

    # --- HTML: 摘要表 ---
    html += '<h3>資金流向總覽</h3><table class="data-table"><thead><tr>'
    html += '<th>日期</th><th>事件類型</th><th>入金(帳面)</th><th>平倉淨額</th><th>客戶實際入金</th><th>出金</th><th>說明</th>'
    html += '</tr></thead><tbody>'
    for fe in flow_events:
        d = fe['date'].strftime('%Y-%m-%d') if hasattr(fe['date'], 'strftime') else str(fe['date'])
        sn = fe.get('sale_net', 0)
        cc = fe.get('customer_cash', fe['in'])
        html += f'''<tr>
            <td style="text-align:left">{d}</td><td style="text-align:left">{fe['type']}</td>
            <td>{fe['in']:,.0f}</td><td>{sn:,.0f}</td><td>{cc:,.0f}</td><td>{fe['out']:,.0f}</td>
            <td style="text-align:left">{fe['desc']}</td></tr>'''
    html += f'''<tr style="font-weight:bold;background:#e8f4fc;">
        <td style="text-align:left" colspan="2">合計</td>
        <td>{total_in:,.0f}</td><td>{total_sale_net:,.0f}</td><td>{total_customer_cash:,.0f}</td><td>{total_out:,.0f}</td>
        <td style="text-align:left">淨客戶現金流 {total_customer_cash - total_out:,.0f}</td></tr>'''
    html += '</tbody></table>'

    # --- HTML: 逐事件計算明細 ---
    html += '<h3>逐事件計算明細</h3>'

    # 建倉
    html += f'''
    <div style="margin-bottom:25px;padding:15px;background:#f9f9f9;border-radius:8px;">
        <h4 style="color:#1f77b4;margin-top:0;">建倉入金 — {build_date.strftime('%Y-%m-%d')}</h4>
        <table class="data-table" style="max-width:600px;">
            <tr><td style="text-align:left">建倉日 IM（Base_IM 大邊 + 小邊折減後）</td><td><strong>{first_im:,.0f}</strong></td></tr>
            <tr><td style="text-align:left">入金 = IM</td><td><strong>{first_im:,.0f}</strong></td></tr>
            <tr><td style="text-align:left">MM = IM × 70%</td><td><strong>{first_im * 0.7:,.0f}</strong></td></tr>
        </table>
    </div>'''

    # 加減倉事件
    for evt in (results.position_change_events or []):
        evt_date = evt['date'].strftime('%Y-%m-%d')
        equity_before = evt.get('equity_before_change', 0)
        new_im = evt.get('new_im', 0)
        new_mm = evt.get('new_mm', 0)
        realized_pnl = evt.get('realized_pnl', 0)
        withdrawal = evt.get('withdrawal', 0)
        equity_after_wdl = evt.get('equity_at_change', 0)
        deposit = max(0, new_im - equity_after_wdl)
        equity_final = equity_after_wdl + deposit
        cash_base = evt.get('cash_base', equity_before)
        max_withdrawal = max(0, min(cash_base - new_im, equity_before - new_im))

        old_im = evt.get('old_im', 0)
        old_long_mv = evt.get('old_long_mv', 0)
        old_short_mv = evt.get('old_short_mv', 0)
        new_long_mv = evt.get('long_mv', 0)
        new_short_mv = evt.get('short_mv', 0)

        html += f'''
    <div style="margin-bottom:25px;padding:15px;background:#f9f9f9;border-radius:8px;">
        <h4 style="color:#1f77b4;margin-top:0;">加減倉 — {evt_date}</h4>
        <table class="data-table" style="max-width:700px;">
            <tr style="background:#e8f4fc;"><td style="text-align:left" colspan="2"><strong>變動前（舊部位以當日價格結算）</strong></td></tr>
            <tr><td style="text-align:left">舊部位 多方MV / 空方MV</td><td>{old_long_mv:,.0f} / {old_short_mv:,.0f}</td></tr>
            <tr><td style="text-align:left">舊 IM</td><td>{old_im:,.0f}</td></tr>
            <tr><td style="text-align:left">① 變動前權益</td><td><strong>{equity_before:,.0f}</strong></td></tr>
            <tr style="background:#e8f4fc;"><td style="text-align:left" colspan="2"><strong>變動後（新部位 IM 計算）</strong></td></tr>
            <tr><td style="text-align:left">新部位 多方MV / 空方MV</td><td>{new_long_mv:,.0f} / {new_short_mv:,.0f}</td></tr>
            <tr><td style="text-align:left">② 新 IM</td><td><strong>{new_im:,.0f}</strong></td></tr>
            <tr style="background:#e8f4fc;"><td style="text-align:left" colspan="2"><strong>實現損益與出金</strong></td></tr>
            <tr><td style="text-align:left">③ 實現損益（平/減倉部位 × 基準價差）</td><td><strong>{realized_pnl:+,.0f}</strong></td></tr>
            <tr><td style="text-align:left">③½ 現金基底（期初入金 + ③實現損益）</td><td>{cash_base:,.0f}</td></tr>
            <tr><td style="text-align:left">④ 可出金（浮盈不可出金）</td><td>max(0, min({cash_base:,.0f}, {equity_before:,.0f}) - {new_im:,.0f}) = <strong>{max_withdrawal:,.0f}</strong></td></tr>
            <tr><td style="text-align:left">⑤ 實際出金</td><td><strong style="color:#dc3545;">-{withdrawal:,.0f}</strong></td></tr>
            <tr><td style="text-align:left">⑥ 出金後權益</td><td>{equity_after_wdl:,.0f}</td></tr>
            <tr style="background:#e8f4fc;"><td style="text-align:left" colspan="2"><strong>入金（補足至新 IM）</strong></td></tr>
            <tr><td style="text-align:left">⑦ 加倉入金 = max(0, ②新IM - ⑥出金後權益)</td><td>max(0, {new_im:,.0f} - {equity_after_wdl:,.0f}) = <strong style="color:#28a745;">+{deposit:,.0f}</strong></td></tr>
            <tr><td style="text-align:left">⑧ 最終權益</td><td><strong>{equity_final:,.0f}</strong></td></tr>
            <tr><td style="text-align:left">⑨ 新 MM = 新IM × 70%</td><td>{new_mm:,.0f}</td></tr>
        </table>'''

        # 逐部位實現損益明細
        pnl_details = evt.get('realized_pnl_details', [])
        if pnl_details:
            html += '<p style="margin-top:10px;"><strong>逐部位實現損益：</strong></p>'
            html += '<table class="data-table" style="max-width:800px;"><thead><tr>'
            html += '<th>代號</th><th>方向</th><th>變動</th><th>原數量</th><th>新數量</th><th>平/減量</th><th>基準價</th><th>當日價</th><th>實現損益</th>'
            html += '</tr></thead><tbody>'
            for d in pnl_details:
                side_label = '多' if d['side'] == 'LONG' else '空'
                pnl_class = 'positive' if d['pnl'] >= 0 else 'negative'
                html += f'''<tr>
                    <td style="text-align:left">{d['code']}</td><td>{side_label}</td><td>{d['change_type']}</td>
                    <td>{d['old_qty']:,}</td><td>{d['new_qty']:,}</td><td>{d['closed_qty']:,}</td>
                    <td>{d['base_price']:.2f}</td><td>{d['current_price']:.2f}</td>
                    <td class="{pnl_class}"><strong>{d['pnl']:+,.0f}</strong></td></tr>'''
            html += '</tbody></table>'

            # 平倉價金分析（僅賣出多方產生現金）
            sell_details = [d for d in pnl_details if d['side'] == 'LONG']
            if sell_details:
                sp_total = sum(d['current_price'] * d['closed_qty'] for d in sell_details)
                sp_tax = round(sp_total * TAX_RATE)
                sp_net = sp_total - sp_tax
                cust_cash = max(0, deposit - sp_net)
                # 僅在需入金時計算餘額出金；出金時已含在 ④ 出金中
                sp_surplus = max(0, sp_net - deposit) if deposit > 0 else 0
                html += '<p style="margin-top:10px;"><strong>平倉價金與實際現金流分析：</strong></p>'
                html += '<table class="data-table" style="max-width:700px;">'
                html += '<tr style="background:#e8f4fc;"><td style="text-align:left" colspan="2"><strong>賣出價金（平倉多方部位）</strong></td></tr>'
                for d in sell_details:
                    sp = d['current_price'] * d['closed_qty']
                    html += f'<tr><td style="text-align:left">⑩ {d["code"]} 賣出價金 = {d["current_price"]:,.2f} × {d["closed_qty"]:,}</td><td>{sp:,.0f}</td></tr>'
                if len(sell_details) > 1:
                    html += f'<tr><td style="text-align:left">　 賣出價金合計</td><td><strong>{sp_total:,.0f}</strong></td></tr>'
                html += f'''<tr><td style="text-align:left">⑪ 證交稅（{TAX_RATE:.1%}）</td><td style="color:#dc3545;">-{sp_tax:,.0f}</td></tr>
                <tr><td style="text-align:left">⑫ 淨賣出價金 = ⑩ - ⑪</td><td><strong>{sp_net:,.0f}</strong></td></tr>
                <tr style="background:#e8f4fc;"><td style="text-align:left" colspan="2"><strong>客戶現金流</strong></td></tr>
                <tr><td style="text-align:left">⑬ 客戶實際入金 = max(0, ⑦加倉入金 - ⑫淨賣出價金)</td><td>max(0, {deposit:,.0f} - {sp_net:,.0f}) = <strong style="color:#ff6600;">{cust_cash:,.0f}</strong></td></tr>
                <tr><td style="text-align:left">　 其中來自賣出價金</td><td style="color:#28a745;">{min(deposit, sp_net):,.0f}</td></tr>
                <tr><td style="text-align:left">⑭ 賣出價金餘額可出金 = max(0, ⑫淨賣出價金 - ⑦加倉入金)</td><td>max(0, {sp_net:,.0f} - {deposit:,.0f}) = <strong style="color:#28a745;">{sp_surplus:,.0f}</strong></td></tr>'''
                html += '</table>'
        elif realized_pnl == 0:
            html += '<p style="color:#666;">無平/減倉部位（純加倉）</p>'

        html += '</div>'

    # 追繳事件
    for evt in (results.margin_call_events or []):
        mc_date = evt['date'].strftime('%Y-%m-%d') if hasattr(evt['date'], 'strftime') else str(evt['date'])
        mc_im = evt.get('im_today', 0)
        mc_mm = evt.get('mm_today', 0)
        mc_eq = evt.get('equity', 0)
        mc_dep = evt.get('required_deposit', 0)
        mc_eq_before = mc_eq - mc_dep  # 追繳前權益 ≈ equity - deposit

        # 從 timeseries 找追繳前權益
        mc_ts_mask = ts['date'] == pd.Timestamp(evt['date'])
        if mc_ts_mask.any():
            mc_row = ts[mc_ts_mask].iloc[0]
            mc_eq_before = mc_row.get('Equity_Before', mc_eq_before)
            mc_mm = mc_row.get('MM_At_Call', mc_mm)

        html += f'''
    <div style="margin-bottom:25px;padding:15px;background:#fff3cd;border-radius:8px;">
        <h4 style="color:#dc3545;margin-top:0;">追繳 — {mc_date}</h4>
        <table class="data-table" style="max-width:600px;">
            <tr><td style="text-align:left">① 追繳前權益</td><td><strong>{mc_eq_before:,.0f}</strong></td></tr>
            <tr><td style="text-align:left">② 維持保證金(MM)</td><td>{mc_mm:,.0f}</td></tr>
            <tr><td style="text-align:left">觸發條件：①權益 &lt; ②MM</td><td style="color:#dc3545;"><strong>{mc_eq_before:,.0f} &lt; {mc_mm:,.0f} → 觸發追繳</strong></td></tr>
            <tr><td style="text-align:left">③ 當日新 IM</td><td>{mc_im:,.0f}</td></tr>
            <tr><td style="text-align:left">④ 追繳入金 = ③新IM - ①權益</td><td><strong style="color:#28a745;">+{mc_dep:,.0f}</strong></td></tr>
            <tr><td style="text-align:left">⑤ 追繳後權益 = 新IM</td><td><strong>{mc_im:,.0f}</strong></td></tr>
            <tr><td style="text-align:left">⑥ 新 MM = 新IM × 70%</td><td>{mc_im * 0.7:,.0f}</td></tr>
        </table>
    </div>'''

    return html


def create_html_report(results, position_schedule):
    """建立完整 HTML 報告"""
    ts = results.timeseries_df.copy()

    # 計算入金欄位
    if len(ts) > 0:
        ts = _compute_deposit_column(ts, results)

    fig1 = create_timeseries_chart(ts)
    fig2 = create_mv_chart(ts)
    fig3 = create_reduction_chart(ts)
    chart1_html = fig1.to_html(full_html=False, include_plotlyjs='cdn')
    chart2_html = fig2.to_html(full_html=False, include_plotlyjs=False)
    chart3_html = fig3.to_html(full_html=False, include_plotlyjs=False)

    first_day = ts.iloc[0]
    last_day = ts.iloc[-1]

    # 權益表
    equity_cols = ['date', 'Long_MV', 'Short_MV',
                   'Daily_PnL_Long', 'Daily_PnL_Short', 'Daily_PnL',
                   'Cum_PnL_Long', 'Cum_PnL_Short', 'Cumulative_PnL',
                   'Equity_Before', 'MM_At_Call', 'IM_today',
                   'margin_call_flag', 'Required_Deposit',
                   'Withdrawal', 'Equity', 'MM_today']
    equity_df = ts[[c for c in equity_cols if c in ts.columns]].copy()
    equity_df['date'] = equity_df['date'].dt.strftime('%Y-%m-%d')
    for col in equity_df.columns:
        if col not in ('date', 'margin_call_flag'):
            equity_df[col] = equity_df[col].apply(lambda x: f'{x:,.0f}' if pd.notna(x) else '')
    equity_df.columns = ['日期', '多方MV', '空方MV', '多方日損益', '空方日損益', '合計日損益',
                          '多方累計', '空方累計', '合計累計', '權益(判定)', 'MM(判定)', 'IM',
                          '追繳', '追繳入金', '出金',
                          '權益(補後)', 'MM(補後)'][:len(equity_df.columns)]

    # 出入金追蹤表（僅事件日）
    cashflow_rows = []
    cum_customer_net = 0.0
    # 建倉
    first_im = ts.iloc[0]['IM_today']
    cum_customer_net += first_im
    cashflow_rows.append({
        '日期': ts.iloc[0]['date'].strftime('%Y-%m-%d') if hasattr(ts.iloc[0]['date'], 'strftime') else str(ts.iloc[0]['date']),
        '事件': '建倉', '帳面入金': first_im, '平倉淨額': 0,
        '客戶實際入金': first_im, '追繳入金': 0,
        '出金(損益)': 0, '平倉餘額出金': 0,
        '累計客戶淨現金流': cum_customer_net,
    })
    # 加減倉
    TAX_RATE_CF = 0.003
    for evt in (results.position_change_events or []):
        evt_date = evt['date'].strftime('%Y-%m-%d') if hasattr(evt['date'], 'strftime') else str(evt['date'])
        new_im = evt.get('new_im', 0)
        eq_after = evt.get('equity_at_change', 0)
        dep = max(0, new_im - eq_after)
        wdl = evt.get('withdrawal', 0)
        pnl_dets = evt.get('realized_pnl_details', [])
        sp = sum(d['current_price'] * d['closed_qty'] for d in pnl_dets if d['side'] == 'LONG')
        sp_net = sp - round(sp * TAX_RATE_CF)
        cust_dep = max(0, dep - sp_net)
        surplus = max(0, sp_net - dep) if dep > 0 else 0
        cum_customer_net += cust_dep - wdl - surplus
        cashflow_rows.append({
            '日期': evt_date, '事件': '加減倉', '帳面入金': dep,
            '平倉淨額': sp_net, '客戶實際入金': cust_dep,
            '追繳入金': 0, '出金(損益)': wdl, '平倉餘額出金': surplus,
            '累計客戶淨現金流': cum_customer_net,
        })
    # 追繳
    for mc in (results.margin_call_events or []):
        mc_date = mc['date'].strftime('%Y-%m-%d') if hasattr(mc['date'], 'strftime') else str(mc['date'])
        mc_dep = mc.get('required_deposit', 0)
        cum_customer_net += mc_dep
        cashflow_rows.append({
            '日期': mc_date, '事件': '追繳', '帳面入金': 0,
            '平倉淨額': 0, '客戶實際入金': 0,
            '追繳入金': mc_dep, '出金(損益)': 0, '平倉餘額出金': 0,
            '累計客戶淨現金流': cum_customer_net,
        })
    cashflow_df = pd.DataFrame(cashflow_rows)
    if len(cashflow_df) > 0:
        cashflow_df = cashflow_df.sort_values('日期').reset_index(drop=True)
        # 重算累計
        cashflow_df['累計客戶淨現金流'] = (
            cashflow_df['客戶實際入金'] + cashflow_df['追繳入金']
            - cashflow_df['出金(損益)'] - cashflow_df['平倉餘額出金']
        ).cumsum()
        for col in cashflow_df.columns:
            if col not in ('日期', '事件'):
                cashflow_df[col] = cashflow_df[col].apply(lambda x: f'{x:,.0f}' if pd.notna(x) else '')
    cashflow_table_html = cashflow_df.to_html(index=False, classes='data-table', escape=False) if len(cashflow_df) > 0 else ''

    # 保證金表
    margin_cols = ['date', 'Base_IM_long', 'Base_IM_short', 'IM_big', 'IM_small_before',
                   'reduction_etf_100', 'reduction_same_bucket', 'reduction_cross_bucket',
                   'total_reduction', 'IM_small_after', 'IM_today', 'Gross_Lev', 'Raw_Lev']
    margin_df = ts[[c for c in margin_cols if c in ts.columns]].copy()
    margin_df['date'] = margin_df['date'].dt.strftime('%Y-%m-%d')
    for col in margin_df.columns:
        if col == 'date':
            continue
        elif col in ('Gross_Lev', 'Raw_Lev'):
            margin_df[col] = margin_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '')
        else:
            margin_df[col] = margin_df[col].apply(lambda x: f'{x:,.0f}' if pd.notna(x) else '')
    margin_df.columns = ['日期', '多方Base_IM', '空方Base_IM', 'IM大邊', 'IM小邊(折前)',
                          'ETF折減', '同桶折減', '跨桶折減', '總折減',
                          'IM小邊(折後)', 'IM_today', 'Gross槓桿', '無折減槓桿'][:len(margin_df.columns)]

    # 融資表
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

    html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>手動建倉模擬報告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft JhengHei", sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #1f77b4; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; border-left: 4px solid #1f77b4; padding-left: 10px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1f77b4; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: right; border: 1px solid #ddd; font-size: 13px; }}
        th {{ background: #1f77b4; color: white; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #e8f4fc; }}
        td:first-child, th:first-child {{ text-align: left; }}
        .table-wrapper {{ max-height: 400px; overflow-y: auto; margin: 20px 0; }}
        .info {{ background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 15px; margin: 10px 0; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; }}
        .footer {{ text-align: center; color: #666; margin-top: 40px; padding: 20px; border-top: 1px solid #ddd; }}
        .negative {{ color: #dc3545; }}
        .positive {{ color: #28a745; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>手動建倉模擬報告</h1>
        <p style="color:#666;">產出時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>模擬摘要</h2>
        <div class="summary-grid">
            <div class="metric-card"><div class="metric-value">{first_day['date'].strftime('%Y-%m-%d') if hasattr(first_day['date'], 'strftime') else first_day['date']}</div><div class="metric-label">建倉日期</div></div>
            <div class="metric-card"><div class="metric-value">{last_day['date'].strftime('%Y-%m-%d') if hasattr(last_day['date'], 'strftime') else last_day['date']}</div><div class="metric-label">結束日期</div></div>
            <div class="metric-card"><div class="metric-value">{len(ts)}</div><div class="metric-label">交易日數</div></div>
            <div class="metric-card"><div class="metric-value">{first_day['IM_today']:,.0f}</div><div class="metric-label">建倉日 IM</div></div>
            <div class="metric-card"><div class="metric-value">{last_day['IM_today']:,.0f}</div><div class="metric-label">最新 IM</div></div>
            <div class="metric-card"><div class="metric-value">{last_day['Equity']:,.0f}</div><div class="metric-label">最新權益</div></div>
            <div class="metric-card"><div class="metric-value {'negative' if last_day['Cumulative_PnL'] < 0 else 'positive'}">{last_day['Cumulative_PnL']:+,.0f}</div><div class="metric-label">累計損益</div></div>
            <div class="metric-card"><div class="metric-value">{int(ts['margin_call_flag'].sum())}</div><div class="metric-label">追繳次數</div></div>
            <div class="metric-card"><div class="metric-value">{last_day.get('Financing_Amount', 0):,.0f}</div><div class="metric-label">融資金額</div></div>
            <div class="metric-card"><div class="metric-value">{last_day.get('Cumulative_Interest', 0):,.0f}</div><div class="metric-label">累計利息支出</div></div>
        </div>

        <h2>IM / MM / 權益走勢</h2>
        <div class="chart-container">{chart1_html}</div>

        <h2>多空市值走勢</h2>
        <div class="chart-container">{chart2_html}</div>

        <h2>折減來源分解</h2>
        <div class="chart-container">{chart3_html}</div>

        <h2>權益與損益追蹤</h2>
        <div class="table-wrapper">{equity_df.to_html(index=False, classes='data-table', escape=False)}</div>

        <h2>保證金計算明細</h2>
        <div class="table-wrapper">{margin_df.to_html(index=False, classes='data-table', escape=False)}</div>

        <h2>融資費用明細</h2>
        <div class="table-wrapper">{financing_df.to_html(index=False, classes='data-table', escape=False)}</div>

        <h2>出入金追蹤</h2>
        <div class="table-wrapper">{cashflow_table_html}</div>

        <h2>出入金計算明細</h2>
        {_generate_cashflow_section(results, position_schedule)}

        <h2>多空配對與減收明細</h2>
        {_generate_hedge_sections(results)}

        <h2>假設與說明</h2>
        <div class="info"><ul>{''.join(f'<li>{a}</li>' for a in results.assumptions)}</ul></div>

        {f'<div class="warning"><strong>追繳事件：</strong>共 {len(results.margin_call_events)} 次追繳</div>' if results.margin_call_events else ''}

        <div class="footer"><p>手動建倉模擬平台</p><p>此報告由系統自動產生，僅供參考</p></div>
    </div>
</body>
</html>'''
    return html


def create_full_report_excel(results, position_schedule):
    """建立完整報告 Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        ts = results.timeseries_df.copy()
        if len(ts) > 0:
            ts = _compute_deposit_column(ts, results)
        if len(ts) > 0:
            first_day = ts.iloc[0]
            last_day = ts.iloc[-1]
            summary_data = {
                '項目': ['模擬期間', '交易日數', '建倉日IM', '最新IM',
                        '建倉日MM', '最新MM', '最新權益', '累計損益',
                        '追繳次數', '平均Gross槓桿', '平均無折減槓桿'],
                '數值': [
                    f"{first_day['date'].strftime('%Y-%m-%d') if hasattr(first_day['date'], 'strftime') else first_day['date']} ~ {last_day['date'].strftime('%Y-%m-%d') if hasattr(last_day['date'], 'strftime') else last_day['date']}",
                    len(ts), f"{first_day['IM_today']:,.0f}", f"{last_day['IM_today']:,.0f}",
                    f"{first_day['MM_today']:,.0f}", f"{last_day['MM_today']:,.0f}",
                    f"{last_day['Equity']:,.0f}", f"{last_day['Cumulative_PnL']:,.0f}",
                    int(ts['margin_call_flag'].sum()),
                    f"{ts['Gross_Lev'].mean():.2f}x", f"{ts['Raw_Lev'].mean():.2f}x"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='摘要', index=False)

        if len(ts) > 0:
            equity_cols = ['date', 'Long_MV', 'Short_MV',
                           'Daily_PnL_Long', 'Daily_PnL_Short', 'Daily_PnL',
                           'Cum_PnL_Long', 'Cum_PnL_Short', 'Cumulative_PnL',
                           'Equity_Before', 'MM_At_Call', 'IM_today',
                           'margin_call_flag', 'Required_Deposit',
                           'Withdrawal', 'Equity', 'MM_today']
            equity_df = ts[[c for c in equity_cols if c in ts.columns]].copy()
            equity_df.columns = ['日期', '多方MV', '空方MV', '多方日損益', '空方日損益', '合計日損益',
                                 '多方累計', '空方累計', '合計累計', '權益(判定)', 'MM(判定)', 'IM',
                                 '追繳', '追繳入金', '出金',
                                 '權益(補後)', 'MM(補後)'][:len(equity_df.columns)]
            equity_df.to_excel(writer, sheet_name='權益損益追蹤', index=False)

        # 出入金追蹤（獨立 sheet，僅事件日）
        if len(ts) > 0:
            cf_rows = []
            _first_im = ts.iloc[0]['IM_today']
            _cum = 0.0
            _cum += _first_im
            cf_rows.append({'日期': ts.iloc[0]['date'], '事件': '建倉',
                            '帳面入金': _first_im, '平倉淨額': 0,
                            '客戶實際入金': _first_im, '追繳入金': 0,
                            '出金(損益)': 0, '平倉餘額出金': 0})
            _TR = 0.003
            for _e in (results.position_change_events or []):
                _nim = _e.get('new_im', 0)
                _ea = _e.get('equity_at_change', 0)
                _d = max(0, _nim - _ea)
                _w = _e.get('withdrawal', 0)
                _pd = _e.get('realized_pnl_details', [])
                _sp = sum(x['current_price'] * x['closed_qty'] for x in _pd if x['side'] == 'LONG')
                _sn = _sp - round(_sp * _TR)
                _cd = max(0, _d - _sn)
                _su = max(0, _sn - _d) if _d > 0 else 0
                cf_rows.append({'日期': _e['date'], '事件': '加減倉',
                                '帳面入金': _d, '平倉淨額': _sn,
                                '客戶實際入金': _cd, '追繳入金': 0,
                                '出金(損益)': _w, '平倉餘額出金': _su})
            for _m in (results.margin_call_events or []):
                _md = _m.get('required_deposit', 0)
                cf_rows.append({'日期': _m['date'], '事件': '追繳',
                                '帳面入金': 0, '平倉淨額': 0,
                                '客戶實際入金': 0, '追繳入金': _md,
                                '出金(損益)': 0, '平倉餘額出金': 0})
            cf_df = pd.DataFrame(cf_rows)
            cf_df = cf_df.sort_values('日期').reset_index(drop=True)
            cf_df['累計客戶淨現金流'] = (
                cf_df['客戶實際入金'] + cf_df['追繳入金']
                - cf_df['出金(損益)'] - cf_df['平倉餘額出金']
            ).cumsum()
            cf_df.to_excel(writer, sheet_name='出入金追蹤', index=False)

        if len(ts) > 0:
            margin_cols = ['date', 'Base_IM_long', 'Base_IM_short', 'IM_big', 'IM_small_before',
                           'reduction_etf_100', 'reduction_same_bucket', 'reduction_cross_bucket',
                           'total_reduction', 'IM_small_after', 'IM_today', 'Gross_Lev', 'Raw_Lev']
            margin_df = ts[[c for c in margin_cols if c in ts.columns]].copy()
            margin_df.columns = ['日期', '多方Base_IM', '空方Base_IM', 'IM大邊', 'IM小邊(折前)',
                                 'ETF折減', '同桶折減', '跨桶折減', '總折減',
                                 'IM小邊(折後)', 'IM_today', 'Gross槓桿', '無折減槓桿'][:len(margin_df.columns)]
            margin_df.to_excel(writer, sheet_name='保證金計算明細', index=False)

        if results.daily_results:
            hedge_df = results.daily_results[0].margin_result.hedge_pairing_df
            if len(hedge_df) > 0:
                hedge_df.to_excel(writer, sheet_name='多空配對明細', index=False)

        # 部位清單（所有期）
        for idx, (s_date, s_df) in enumerate(position_schedule):
            sheet_name = f"部位_{s_date.strftime('%Y%m%d')}" if len(position_schedule) > 1 else '部位清單'
            s_df.to_excel(writer, sheet_name=sheet_name, index=False)

        if results.margin_call_events:
            pd.DataFrame(results.margin_call_events).to_excel(writer, sheet_name='追繳事件', index=False)

        pd.DataFrame({'假設說明': results.assumptions}).to_excel(writer, sheet_name='假設說明', index=False)

        if results.position_change_events:
            pd.DataFrame(results.position_change_events).to_excel(writer, sheet_name='部位變動事件', index=False)

        if results.missing_codes:
            pd.DataFrame({'缺碼代號': results.missing_codes}).to_excel(writer, sheet_name='缺碼清單', index=False)

    output.seek(0)
    return output.getvalue()


def create_audit_zip(results, position_schedule):
    """建立稽核包 ZIP"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        csv_buf = io.StringIO()
        results.timeseries_df.to_csv(csv_buf, index=False, encoding='utf-8-sig')
        zf.writestr('final_timeseries.csv', csv_buf.getvalue().encode('utf-8-sig'))

        for idx, (s_date, s_df) in enumerate(position_schedule):
            csv_buf = io.StringIO()
            s_df.to_csv(csv_buf, index=False, encoding='utf-8-sig')
            zf.writestr(f'inputs_snapshot/positions_{s_date.strftime("%Y%m%d")}.csv',
                        csv_buf.getvalue().encode('utf-8-sig'))

        if results.margin_call_events:
            csv_buf = io.StringIO()
            pd.DataFrame(results.margin_call_events).to_csv(csv_buf, index=False)
            zf.writestr('margin_call_events.csv', csv_buf.getvalue().encode('utf-8-sig'))

        assumptions_content = "# 假設與保守口徑說明\n\n"
        for a in results.assumptions:
            assumptions_content += f"- {a}\n"
        if results.missing_codes:
            assumptions_content += f"\n## 缺碼清單（共 {len(results.missing_codes)} 檔）\n"
            for code in results.missing_codes[:50]:
                assumptions_content += f"- {code}\n"
        zf.writestr('assumptions.md', assumptions_content.encode('utf-8'))

        verification = verify(results)
        import json
        zf.writestr('verification.json', json.dumps(verification, ensure_ascii=False, indent=2).encode('utf-8'))

        if results.position_change_events:
            csv_buf = io.StringIO()
            pd.DataFrame(results.position_change_events).to_csv(csv_buf, index=False)
            zf.writestr('position_change_events.csv', csv_buf.getvalue().encode('utf-8-sig'))

        if results.daily_results:
            n = len(results.daily_results)
            sample_indices = [0, n // 2, n - 1] if n >= 3 else list(range(n))
            for idx in sample_indices:
                dr = results.daily_results[idx]
                date_str = dr.date.strftime('%Y%m%d')
                mr = dr.margin_result
                for attr, suffix in [('summary_df', 'summary'), ('bucket_hedge_df', 'bucket_hedge'),
                                     ('reduction_breakdown_df', 'reduction_breakdown')]:
                    csv_buf = io.StringIO()
                    getattr(mr, attr).to_csv(csv_buf, index=False)
                    zf.writestr(f'calc_steps/{date_str}_{suffix}.csv', csv_buf.getvalue().encode('utf-8-sig'))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# =============================================================================
# 主程式
# =============================================================================
def main():
    st.markdown('<p class="main-header">手動建倉模擬平台</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Manual Position Building Simulation Platform</p>', unsafe_allow_html=True)

    with st.expander("制度口徑一句話摘要", expanded=False):
        st.info("""
        **本制度以固定槓桿計算分邊 Base IM，並以 Base IM 判定大小邊；**
        對沖折減僅適用於小邊，依三產業桶與 3M 加權累積報酬率決定折減率（50% 或 20%）；
        0050/0056 ETF 採 look-through，成份股完全對沖部分可 100% 減收；
        維持保證金為當日 IM 的 70%，跌破維持保證金時需追繳回補至當日 IM（100%）。
        """)

    # ----- 區塊 1：初始化 -----
    loader, error = init_data_loader()
    if error:
        st.error(f"資料載入器初始化失敗：{error}")
        st.stop()
    st.session_state.data_loader = loader

    # ----- 區塊 2：Sidebar - 股價來源 -----
    st.sidebar.header("參數設定")
    st.sidebar.subheader("0. 股價數據")

    prices_df, trading_dates, price_error = get_trading_dates_and_prices(loader)

    if price_error:
        st.sidebar.warning("雲端數據載入失敗，請上傳股價檔案")
        price_file = st.sidebar.file_uploader("上傳股價 CSV", type=['csv'],
                                               help="必須包含欄位：日期、股票代號、收盤價",
                                               key="price_upload")
        if price_file is not None:
            prices_df, err = load_prices_from_upload(price_file)
            if err:
                st.sidebar.error(f"解析失敗：{err}")
            else:
                st.session_state.prices_df = prices_df
                st.session_state.trading_dates = sorted(prices_df['date'].unique())
                loader.set_prices_df(prices_df)
                st.sidebar.success(f"已載入 {len(prices_df)} 筆股價數據")
                trading_dates = st.session_state.trading_dates
    else:
        st.sidebar.success("股價數據已載入")
        st.session_state.prices_df = prices_df
        st.session_state.trading_dates = trading_dates

    if st.session_state.prices_df is not None:
        loader.set_prices_df(st.session_state.prices_df)

    prices_df = st.session_state.prices_df
    trading_dates = st.session_state.trading_dates

    if prices_df is None or len(trading_dates) == 0:
        st.warning("請先載入股價數據")
        st.stop()

    # 建立代號集合與日期→價格映射
    available_codes = set(prices_df['code'].unique())
    etf_codes = {'0050', '0056', '50', '56'}

    # 載入槓桿對照表（股期標的 → 遠期槓桿倍數）
    try:
        futures_leverage_map = loader.get_futures_leverage_mapping()
    except Exception:
        futures_leverage_map = {}
    leverage_rules = loader.config.get('leverage_rules', {})

    # ----- 區塊 3：Sidebar - 部位建構器 -----
    st.sidebar.subheader("1. 部位建構器")

    # 3a. 選擇日期
    min_date = pd.Timestamp(trading_dates[0]).date()
    max_date = pd.Timestamp(trading_dates[-1]).date()

    selected_date = st.sidebar.date_input(
        "建倉/加減倉日期",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        help="選擇建倉或加減倉的日期"
    )
    selected_ts = pd.Timestamp(selected_date)

    # 找到最近交易日
    td_array = np.array([pd.Timestamp(d) for d in trading_dates])
    nearest_idx = np.searchsorted(td_array, selected_ts, side='right') - 1
    nearest_idx = max(0, min(nearest_idx, len(td_array) - 1))
    actual_date = td_array[nearest_idx]
    if actual_date != selected_ts:
        st.sidebar.caption(f"最近交易日：{actual_date.strftime('%Y-%m-%d')}")

    # 3b. 輸入代號
    code_input = st.sidebar.text_input("股票代號", placeholder="例：2330",
                                        help="輸入股票代號，按 Enter 確認")

    # 驗證代號並顯示收盤價與槓桿
    code_valid = False
    code_price = None
    code_instrument = None
    code_leverage = None

    if code_input:
        code_clean = code_input.strip()
        if code_clean in available_codes:
            code_valid = True
            code_price = get_price_on_date(prices_df, code_clean, actual_date)
            code_instrument = classify_instrument(code_clean, etf_codes)

            # 查詢槓桿倍數
            if code_instrument == 'ETF':
                code_leverage = leverage_rules.get('etf_high_volume', 7.0)
            elif code_clean in futures_leverage_map:
                code_leverage = futures_leverage_map[code_clean]
            else:
                code_leverage = leverage_rules.get('default', 3.0)

            if code_price is not None:
                st.sidebar.success(
                    f"{code_clean} | 收盤價：{code_price:,.2f} | "
                    f"{code_instrument} | 槓桿：{code_leverage:.0f}x"
                )
            else:
                st.sidebar.warning(f"{code_clean} 在 {actual_date.strftime('%Y-%m-%d')} 無收盤價")
                code_valid = False
        else:
            st.sidebar.error(f"代號 {code_clean} 不存在於股價資料中")

    # 3c. 方向與張數
    col_side, col_qty = st.sidebar.columns(2)
    with col_side:
        side = st.selectbox("方向", ["LONG", "SHORT"], key="side_select")
    with col_qty:
        qty_lots = st.number_input("張數", min_value=1, value=1, step=1, key="qty_input")

    # 顯示市值預估
    if code_valid and code_price is not None:
        contract_mult = loader.config.get('contract_multiplier', 1000)
        mv = code_price * qty_lots * contract_mult
        if mv >= 1e8:
            mv_str = f"{mv / 1e8:,.2f} 億"
        elif mv >= 1e7:
            mv_str = f"{mv / 1e7:,.2f} 千萬"
        elif mv >= 1e4:
            mv_str = f"{mv / 1e4:,.2f} 萬"
        else:
            mv_str = f"{mv:,.0f}"
        st.sidebar.info(f"預估市值：{mv_str} TWD")

    # 3d. 加入部位按鈕
    if st.sidebar.button("加入部位", type="primary", use_container_width=True):
        if not code_valid:
            st.sidebar.error("請先輸入有效代號")
        else:
            code_clean = code_input.strip()
            # 找到或建立對應日期的 entry
            entry_idx = None
            for i, entry in enumerate(st.session_state.schedule_entries):
                if pd.Timestamp(entry['date']) == actual_date:
                    entry_idx = i
                    break

            if entry_idx is None:
                # 新日期：自動繼承前期部位（以當日收盤價更新）
                carried_positions = []
                if st.session_state.schedule_entries:
                    # 找最近一期（日期 <= actual_date 中最晚的）
                    prev_entries = [e for e in st.session_state.schedule_entries
                                    if pd.Timestamp(e['date']) < actual_date]
                    if prev_entries:
                        prev_entry = max(prev_entries, key=lambda e: e['date'])
                        for pos in prev_entry['positions']:
                            new_price = get_price_on_date(prices_df, pos['code'], actual_date)
                            carried_positions.append({
                                'code': pos['code'],
                                'side': pos['side'],
                                'qty_lots': pos['qty_lots'],
                                'instrument': pos['instrument'],
                                'price': new_price if new_price else pos['price'],
                            })
                st.session_state.schedule_entries.append({
                    'date': actual_date,
                    'positions': carried_positions,
                })
                entry_idx = len(st.session_state.schedule_entries) - 1

            # 檢查是否已有相同代號+方向 → 加碼
            # 或有反方向 → 淨額化（反向平倉）
            positions = st.session_state.schedule_entries[entry_idx]['positions']
            same_side_idx = None
            opp_side_idx = None
            opp_side = 'SHORT' if side == 'LONG' else 'LONG'
            for pi, pos in enumerate(positions):
                if pos['code'] == code_clean:
                    if pos['side'] == side:
                        same_side_idx = pi
                    elif pos['side'] == opp_side:
                        opp_side_idx = pi

            if same_side_idx is not None:
                # 同方向 → 累加張數
                positions[same_side_idx]['qty_lots'] += qty_lots
            elif opp_side_idx is not None:
                # 反方向 → 淨額化
                opp_pos = positions[opp_side_idx]
                if qty_lots < opp_pos['qty_lots']:
                    # 部分平倉：減少反方向數量
                    opp_pos['qty_lots'] -= qty_lots
                elif qty_lots == opp_pos['qty_lots']:
                    # 完全平倉：移除反方向部位
                    positions.pop(opp_side_idx)
                else:
                    # 超過反方向 → 移除反方向，新增正方向餘額
                    positions.pop(opp_side_idx)
                    positions.append({
                        'code': code_clean,
                        'side': side,
                        'qty_lots': qty_lots - opp_pos['qty_lots'],
                        'instrument': code_instrument,
                        'price': code_price,
                    })
            else:
                # 全新部位
                positions.append({
                    'code': code_clean,
                    'side': side,
                    'qty_lots': qty_lots,
                    'instrument': code_instrument,
                    'price': code_price,
                })

            # 依日期排序
            st.session_state.schedule_entries.sort(key=lambda x: x['date'])
            st.rerun()

    # 3e. 從前期複製部位（新增日期點）
    if st.session_state.schedule_entries:
        st.sidebar.divider()
        if st.sidebar.button("新增日期點（複製前期部位）", use_container_width=True):
            last_entry = st.session_state.schedule_entries[-1]
            # 使用前期部位複製到新日期（用新的價格）
            new_positions = []
            for pos in last_entry['positions']:
                new_price = get_price_on_date(prices_df, pos['code'], actual_date)
                new_positions.append({
                    'code': pos['code'],
                    'side': pos['side'],
                    'qty_lots': pos['qty_lots'],
                    'instrument': pos['instrument'],
                    'price': new_price if new_price else pos['price'],
                })

            # 檢查是否已存在該日期
            exists = False
            for entry in st.session_state.schedule_entries:
                if pd.Timestamp(entry['date']) == actual_date:
                    exists = True
                    break

            if exists:
                st.sidebar.warning("該日期已有部位")
            elif actual_date <= pd.Timestamp(last_entry['date']):
                st.sidebar.warning("新日期必須晚於前期日期")
            else:
                st.session_state.schedule_entries.append({
                    'date': actual_date,
                    'positions': new_positions,
                })
                st.session_state.schedule_entries.sort(key=lambda x: x['date'])
                st.rerun()

        # 清除所有部位
        if st.sidebar.button("清除所有部位", use_container_width=True):
            st.session_state.schedule_entries = []
            st.session_state.backtest_results = None
            st.rerun()

    # 3f. 執行模擬
    st.sidebar.divider()
    st.sidebar.subheader("2. 執行")

    if st.sidebar.button("開始模擬", type="primary", use_container_width=True):
        if not st.session_state.schedule_entries:
            st.error("請先建立部位")
        else:
            position_schedule = schedule_entries_to_position_schedule(
                st.session_state.schedule_entries
            )
            if not position_schedule:
                st.error("部位資料為空")
            else:
                try:
                    engine = BacktestEngine(loader)
                    start_date = position_schedule[0][0]
                    end_date = pd.Timestamp(trading_dates[-1])

                    calc_dates = engine.get_trading_dates_range(start_date, end_date)
                    progress_bar = st.progress(0, text="正在初始化...")
                    status_text = st.empty()

                    def progress_callback(current, total, date_str):
                        pct = current / total if total > 0 else 0
                        progress_bar.progress(pct, text=f"計算中... {current}/{total} ({pct:.0%})")
                        status_text.text(f"正在計算 {date_str}")

                    results = engine.run(
                        position_schedule=position_schedule,
                        start_date=start_date,
                        end_date=end_date,
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

    # ----- 區塊 4：Main - 部位預覽 -----
    if st.session_state.schedule_entries:
        st.header("部位預覽")

        for entry_idx, entry in enumerate(st.session_state.schedule_entries):
            entry_date = pd.Timestamp(entry['date'])
            is_first = entry_idx == 0
            label = f"{entry_date.strftime('%Y-%m-%d')}" + (" (建倉)" if is_first else " (加減倉)")

            with st.expander(label, expanded=True):
                if not entry['positions']:
                    st.info("無部位")
                    continue

                # 建立顯示用 DataFrame
                pos_records = []
                for pos in entry['positions']:
                    mv = (pos['price'] or 0) * pos['qty_lots'] * 1000
                    pos_records.append({
                        '代號': pos['code'],
                        '方向': pos['side'],
                        '張數': pos['qty_lots'],
                        '類型': pos['instrument'],
                        '收盤價': pos['price'] or 0,
                        '市值': mv,
                    })

                pos_df = pd.DataFrame(pos_records)
                st.dataframe(pos_df, use_container_width=True, hide_index=True)

                # 逐筆刪除
                del_cols = st.columns(min(len(entry['positions']), 6))
                deleted = False
                for pi, pos in enumerate(entry['positions']):
                    with del_cols[pi % len(del_cols)]:
                        if st.button(f"刪除 {pos['code']}", key=f"del_{entry_idx}_{pi}"):
                            st.session_state.schedule_entries[entry_idx]['positions'].pop(pi)
                            deleted = True
                if deleted:
                    st.rerun()

                # 摘要指標
                if pos_records:
                    long_mv = sum(r['市值'] for r in pos_records if r['方向'] == 'LONG')
                    short_mv = sum(r['市值'] for r in pos_records if r['方向'] == 'SHORT')
                    net_mv = long_mv - short_mv
                    long_count = sum(1 for r in pos_records if r['方向'] == 'LONG')
                    short_count = sum(1 for r in pos_records if r['方向'] == 'SHORT')

                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.metric("多方", f"{long_count} 檔")
                    with c2:
                        st.metric("多方MV", f"{long_mv:,.0f}")
                    with c3:
                        st.metric("空方", f"{short_count} 檔")
                    with c4:
                        st.metric("空方MV", f"{short_mv:,.0f}")
                    with c5:
                        st.metric("淨MV", f"{net_mv:+,.0f}")

                # 與前期差異
                if entry_idx > 0 and st.session_state.schedule_entries[entry_idx - 1]['positions']:
                    prev_positions = st.session_state.schedule_entries[entry_idx - 1]['positions']
                    prev_map = {}
                    for p in prev_positions:
                        key = (p['code'], p['side'])
                        prev_map[key] = p['qty_lots']

                    curr_map = {}
                    for p in entry['positions']:
                        key = (p['code'], p['side'])
                        curr_map[key] = p['qty_lots']

                    diff_records = []
                    all_keys = set(prev_map.keys()) | set(curr_map.keys())
                    for key in sorted(all_keys):
                        code, side = key
                        prev_qty = prev_map.get(key, 0)
                        curr_qty = curr_map.get(key, 0)
                        if prev_qty != curr_qty:
                            if prev_qty == 0:
                                change_type = '新增'
                            elif curr_qty == 0:
                                change_type = '平倉'
                            elif curr_qty > prev_qty:
                                change_type = '加倉'
                            else:
                                change_type = '減倉'
                            diff_records.append({
                                '代號': code, '方向': side, '變動類型': change_type,
                                '前期張數': prev_qty, '本期張數': curr_qty,
                                '差異': curr_qty - prev_qty,
                            })

                    if diff_records:
                        st.caption("與前期差異：")
                        st.dataframe(pd.DataFrame(diff_records), use_container_width=True, hide_index=True)

    # ----- 區塊 5：Main - 模擬執行與結果 -----
    if st.session_state.backtest_results is not None:
        results = st.session_state.backtest_results
        position_schedule = schedule_entries_to_position_schedule(
            st.session_state.schedule_entries
        )

        st.header("模擬結果")

        # 驗證狀態
        if not results.verification_passed:
            st.error("驗證發現問題")
            for err in results.verification_errors:
                st.write(f"- {err}")
        else:
            st.success("所有驗證通過")

        # 核心指標
        ts = results.timeseries_df

        # 計算入金欄位
        if len(ts) > 0:
            ts = ts.copy()
            ts = _compute_deposit_column(ts, results)

        if len(ts) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最新 IM", f"{ts['IM_today'].iloc[-1]:,.0f}",
                          delta=f"{ts['IM_today'].iloc[-1] - ts['IM_today'].iloc[0]:,.0f}")
            with col2:
                st.metric("最新 MM (70%)", f"{ts['MM_today'].iloc[-1]:,.0f}")
            with col3:
                margin_call_count = int(ts['margin_call_flag'].sum())
                st.metric("追繳次數", f"{margin_call_count}",
                          delta="需注意" if margin_call_count > 0 else None,
                          delta_color="inverse")
            with col4:
                st.metric("平均 Gross 槓桿", f"{ts['Gross_Lev'].mean():.2f}x")

            # 圖表
            st.subheader("時序圖表")

            has_pos_changes = (
                results.position_change_events
                and len(results.position_change_events) > 0
            )

            tab_names = ["IM/MM/Equity", "市值變化", "折減分解"]
            if has_pos_changes:
                tab_names.append("部位變動")
                tab_names.append("資金流向")

            chart_tabs = st.tabs(tab_names)

            with chart_tabs[0]:
                st.plotly_chart(create_timeseries_chart(ts), use_container_width=True)

            with chart_tabs[1]:
                st.plotly_chart(create_mv_chart(ts), use_container_width=True)

            with chart_tabs[2]:
                st.plotly_chart(create_reduction_chart(ts), use_container_width=True)

            if has_pos_changes:
                with chart_tabs[3]:
                    st.markdown("### 部位變動時間線")
                    _change_evt_map = {}
                    if results.position_change_events:
                        for evt in results.position_change_events:
                            _change_evt_map[pd.Timestamp(evt['date'])] = evt

                    # 建立 date → Required_Deposit 查找
                    _deposit_map = {}
                    if len(ts) > 0:
                        for _, row in ts.iterrows():
                            if row.get('Deposit', 0) > 0:
                                _deposit_map[pd.Timestamp(row['date'])] = row['Deposit']

                    timeline_records = []
                    for t_idx, (s_date, s_df) in enumerate(position_schedule):
                        long_count = len(s_df[s_df['side'] == 'LONG'])
                        short_count = len(s_df[s_df['side'] == 'SHORT'])
                        evt = _change_evt_map.get(pd.Timestamp(s_date))
                        rec = {
                            '快照日期': s_date.strftime('%Y-%m-%d'),
                            '多方數量': long_count, '空方數量': short_count,
                            '總部位數': len(s_df),
                        }
                        if t_idx == 0:
                            init_im = ts.iloc[0]['IM_today'] if len(ts) > 0 else 0
                            rec['IM'] = f"{init_im:,.0f}" if len(ts) > 0 else '-'
                            rec['入金'] = f"{init_im:,.0f}"
                            rec['加減倉入金'] = '-'
                            rec['變動時權益'] = '-'
                            rec['實現損益'] = '-'
                            rec['出金'] = '-'
                        elif evt:
                            equity_after_wdl = evt.get('equity_at_change', 0)
                            new_im = evt.get('new_im', 0)
                            deposit = max(0, new_im - equity_after_wdl)
                            rec['IM'] = f"{new_im:,.0f}"
                            rec['入金'] = '-'
                            rec['加減倉入金'] = f"{deposit:,.0f}"
                            rec['變動時權益'] = f"{equity_after_wdl:,.0f}"
                            rec['實現損益'] = f"{evt['realized_pnl']:+,.0f}"
                            rec['出金'] = f"{evt['withdrawal']:,.0f}"
                        else:
                            rec['IM'] = '-'
                            rec['入金'] = '-'
                            rec['加減倉入金'] = '-'
                            rec['變動時權益'] = '-'
                            rec['實現損益'] = '-'
                            rec['出金'] = '-'
                        timeline_records.append(rec)

                    st.dataframe(pd.DataFrame(timeline_records), use_container_width=True, hide_index=True)

                with chart_tabs[4]:
                    st.markdown("### 資金流向分析")
                    if results.position_change_events:
                        flow_tab_names = []
                        for evt_idx, evt in enumerate(results.position_change_events):
                            evt_date_str = evt['date'].strftime('%m/%d')
                            if evt_idx == 0:
                                prev_date_str = position_schedule[0][0].strftime('%m/%d')
                            else:
                                prev_date_str = results.position_change_events[evt_idx - 1]['date'].strftime('%m/%d')
                            flow_tab_names.append(f"{prev_date_str} -> {evt_date_str}")

                        flow_tabs = st.tabs(flow_tab_names)
                        for evt_idx, evt in enumerate(results.position_change_events):
                            with flow_tabs[evt_idx]:
                                evt_date_str = evt['date'].strftime('%Y-%m-%d')
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
                                    '變動前': [f"{old_long_mv:,.0f}", f"{old_short_mv:,.0f}",
                                             f"{old_im:,.0f}", f"{old_mm:,.0f}"],
                                    '變動後': [f"{new_long_mv:,.0f}", f"{new_short_mv:,.0f}",
                                             f"{new_im:,.0f}", f"{new_mm:,.0f}"],
                                    '差異': [f"{new_long_mv - old_long_mv:+,.0f}",
                                            f"{new_short_mv - old_short_mv:+,.0f}",
                                            f"{new_im - old_im:+,.0f}",
                                            f"{new_mm - old_mm:+,.0f}"],
                                }
                                st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

                                st.markdown("#### 資金流向")
                                equity_before = evt.get('equity_before_change', 0)
                                realized_pnl = evt.get('realized_pnl', 0)
                                withdrawal = evt.get('withdrawal', 0)
                                equity_after = evt.get('equity_at_change', 0)
                                cash_base = evt.get('cash_base', equity_before)
                                max_withdrawal = max(0, min(cash_base - new_im, equity_before - new_im))

                                # 加倉入金：補足至新 IM 的差額（非追繳）
                                deposit = max(0, new_im - equity_after)
                                equity_final = equity_after + deposit

                                flow_md = f"""
| 步驟 | 說明 | 金額 |
|------|------|-----:|
| 1 | 變動前權益（舊部位以當日價格結算） | **{equity_before:,.0f}** |
| 2 | 實現損益（平/減倉部位按基準價差計算） | **{realized_pnl:+,.0f}** |
| 2½ | 現金基底（期初入金 + 實現損益） | **{cash_base:,.0f}** |
| 3 | 可出金（浮盈不可出金） | max(0, min({cash_base:,.0f}, {equity_before:,.0f}) - {new_im:,.0f}) = **{max_withdrawal:,.0f}** |
| 4 | 實際出金 | **-{withdrawal:,.0f}** |
| 5 | 出金後權益 | **{equity_after:,.0f}** |
| 6 | 加倉入金（補足至新 IM，非追繳） | **+{deposit:,.0f}** |
| 7 | 最終權益 | **{equity_final:,.0f}** |
"""
                                st.markdown(flow_md)

                                # 逐部位實現損益明細
                                pnl_details = evt.get('realized_pnl_details', [])
                                if pnl_details:
                                    st.markdown("#### 逐部位實現損益明細")
                                    detail_records = []
                                    for d in pnl_details:
                                        side_label = '多' if d['side'] == 'LONG' else '空'
                                        detail_records.append({
                                            '代號': d['code'], '方向': side_label,
                                            '變動': d['change_type'],
                                            '原數量': f"{d['old_qty']:,}",
                                            '新數量': f"{d['new_qty']:,}",
                                            '平/減量': f"{d['closed_qty']:,}",
                                            '基準價': f"{d['base_price']:.2f}",
                                            '當日價': f"{d['current_price']:.2f}",
                                            '實現損益': f"{d['pnl']:+,.0f}",
                                        })
                                    st.dataframe(pd.DataFrame(detail_records),
                                                 use_container_width=True, hide_index=True)
                    else:
                        st.info("無部位變動事件")

            # 逐日明細表
            st.subheader("逐日明細")
            display_df = ts.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

            # 表1：權益與損益追蹤
            st.markdown("**表1：權益與損益追蹤**")
            equity_cols = ['date', 'Long_MV', 'Short_MV',
                           'Daily_PnL_Long', 'Daily_PnL_Short', 'Daily_PnL',
                           'Cum_PnL_Long', 'Cum_PnL_Short', 'Cumulative_PnL',
                           'Equity_Before', 'MM_At_Call', 'IM_today',
                           'Initial_Deposit', 'Pos_Change_Deposit',
                           'margin_call_flag', 'Required_Deposit',
                           'Withdrawal', 'Equity', 'MM_today']
            equity_df = display_df[[c for c in equity_cols if c in display_df.columns]].copy()
            money_cols_1 = ['Long_MV', 'Short_MV', 'Daily_PnL_Long', 'Daily_PnL_Short',
                            'Daily_PnL', 'Cum_PnL_Long', 'Cum_PnL_Short', 'Cumulative_PnL',
                            'Equity_Before', 'MM_At_Call', 'IM_today', 'Initial_Deposit',
                            'Pos_Change_Deposit', 'Required_Deposit',
                            'Withdrawal', 'Equity', 'MM_today']
            for col in money_cols_1:
                if col in equity_df.columns:
                    equity_df[col] = equity_df[col].apply(lambda x: f"{x:,.0f}")
            equity_df = equity_df.rename(columns={
                'date': '日期', 'Long_MV': '多方MV', 'Short_MV': '空方MV',
                'Daily_PnL_Long': '多方日損益', 'Daily_PnL_Short': '空方日損益',
                'Daily_PnL': '合計日損益', 'Cum_PnL_Long': '多方累計',
                'Cum_PnL_Short': '空方累計', 'Cumulative_PnL': '合計累計',
                'Equity_Before': '權益(判定)', 'MM_At_Call': 'MM(判定)', 'IM_today': 'IM',
                'Initial_Deposit': '入金', 'Pos_Change_Deposit': '加減倉入金',
                'margin_call_flag': '追繳', 'Required_Deposit': '追繳入金',
                'Withdrawal': '出金', 'Equity': '權益(補後)', 'MM_today': 'MM(補後)'
            })
            st.dataframe(equity_df, use_container_width=True, height=300)

            # 表2：保證金計算明細
            st.markdown("**表2：保證金計算明細**")
            margin_cols = ['date', 'Base_IM_long', 'Base_IM_short', 'IM_big', 'IM_small_before',
                           'reduction_etf_100', 'reduction_same_bucket', 'reduction_cross_bucket',
                           'total_reduction', 'IM_small_after', 'IM_today', 'Gross_Lev', 'Raw_Lev']
            margin_df = display_df[[c for c in margin_cols if c in display_df.columns]].copy()
            money_cols_2 = ['Base_IM_long', 'Base_IM_short', 'IM_big', 'IM_small_before',
                            'reduction_etf_100', 'reduction_same_bucket', 'reduction_cross_bucket',
                            'total_reduction', 'IM_small_after', 'IM_today']
            for col in money_cols_2:
                if col in margin_df.columns:
                    margin_df[col] = margin_df[col].apply(lambda x: f"{x:,.0f}")
            for col in ['Gross_Lev', 'Raw_Lev']:
                if col in margin_df.columns:
                    margin_df[col] = margin_df[col].apply(lambda x: f"{x:.2f}")
            margin_df = margin_df.rename(columns={
                'date': '日期', 'Base_IM_long': '多方Base_IM', 'Base_IM_short': '空方Base_IM',
                'IM_big': 'IM大邊', 'IM_small_before': 'IM小邊(折前)',
                'reduction_etf_100': 'ETF折減', 'reduction_same_bucket': '同桶折減',
                'reduction_cross_bucket': '跨桶折減', 'total_reduction': '總折減',
                'IM_small_after': 'IM小邊(折後)', 'IM_today': 'IM_today',
                'Gross_Lev': 'Gross槓桿', 'Raw_Lev': '無折減槓桿'
            })
            st.dataframe(margin_df, use_container_width=True, height=300)

            # 表3：融資費用明細
            st.markdown("**表3：融資費用明細**")
            st.caption("多方融資 = 多方MV - IM | 空方融資 = 空方MV | 客戶利率 3% | 券商收益 = 多方融資x1.2% + 空方融資x3% | 利息按日曆日計算")
            financing_cols = ['date', 'Long_MV', 'Short_MV', 'IM_today',
                              'Long_Financing', 'Short_Financing', 'Financing_Amount',
                              'Daily_Interest', 'Cumulative_Interest',
                              'Daily_Broker_Profit', 'Cumulative_Broker_Profit']
            financing_df = display_df[[c for c in financing_cols if c in display_df.columns]].copy()
            money_cols_3 = ['Long_MV', 'Short_MV', 'IM_today', 'Long_Financing', 'Short_Financing',
                            'Financing_Amount', 'Daily_Interest', 'Cumulative_Interest',
                            'Daily_Broker_Profit', 'Cumulative_Broker_Profit']
            for col in money_cols_3:
                if col in financing_df.columns:
                    financing_df[col] = financing_df[col].apply(lambda x: f"{x:,.0f}")
            financing_df = financing_df.rename(columns={
                'date': '日期', 'Long_MV': '多方MV', 'Short_MV': '空方MV', 'IM_today': 'IM',
                'Long_Financing': '多方融資', 'Short_Financing': '空方融資',
                'Financing_Amount': '總融資', 'Daily_Interest': '當日利息',
                'Cumulative_Interest': '累計利息', 'Daily_Broker_Profit': '當日券商收益',
                'Cumulative_Broker_Profit': '累計券商收益'
            })
            st.dataframe(financing_df, use_container_width=True, height=300)

            # 融資摘要
            if len(ts) > 0:
                last_day = ts.iloc[-1]
                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    st.metric("最新融資金額", f"{last_day.get('Financing_Amount', 0):,.0f}")
                with fc2:
                    st.metric("累計利息支出", f"{last_day.get('Cumulative_Interest', 0):,.0f}")
                with fc3:
                    st.metric("券商累計收益", f"{last_day.get('Cumulative_Broker_Profit', 0):,.0f}")
                with fc4:
                    days = len(ts)
                    fin_amt = last_day.get('Financing_Amount', 0)
                    cum_int = last_day.get('Cumulative_Interest', 0)
                    if days > 0 and fin_amt > 0:
                        annualized_cost = (cum_int / fin_amt) * (365 / days) * 100
                        st.metric("年化融資成本", f"{annualized_cost:.2f}%")

            # 表4：出入金追蹤（僅事件日）
            st.markdown("**表4：出入金追蹤**")
            st.caption("僅列出有資金異動的事件日（建倉 / 加減倉 / 追繳）")
            _cf_rows_st = []
            _first_im_st = ts.iloc[0]['IM_today']
            _cf_rows_st.append({'日期': ts.iloc[0]['date'].strftime('%Y-%m-%d') if hasattr(ts.iloc[0]['date'], 'strftime') else str(ts.iloc[0]['date']),
                                '事件': '建倉', '帳面入金': _first_im_st, '平倉淨額': 0,
                                '客戶實際入金': _first_im_st, '追繳入金': 0,
                                '出金(損益)': 0, '平倉餘額出金': 0})
            _TR_ST = 0.003
            for _e_st in (results.position_change_events or []):
                _nim_st = _e_st.get('new_im', 0)
                _ea_st = _e_st.get('equity_at_change', 0)
                _d_st = max(0, _nim_st - _ea_st)
                _w_st = _e_st.get('withdrawal', 0)
                _pd_st = _e_st.get('realized_pnl_details', [])
                _sp_st = sum(x['current_price'] * x['closed_qty'] for x in _pd_st if x['side'] == 'LONG')
                _sn_st = _sp_st - round(_sp_st * _TR_ST)
                _cd_st = max(0, _d_st - _sn_st)
                _su_st = max(0, _sn_st - _d_st) if _d_st > 0 else 0
                _cf_rows_st.append({
                    '日期': _e_st['date'].strftime('%Y-%m-%d') if hasattr(_e_st['date'], 'strftime') else str(_e_st['date']),
                    '事件': '加減倉', '帳面入金': _d_st, '平倉淨額': _sn_st,
                    '客戶實際入金': _cd_st, '追繳入金': 0,
                    '出金(損益)': _w_st, '平倉餘額出金': _su_st})
            for _m_st in (results.margin_call_events or []):
                _md_st = _m_st.get('required_deposit', 0)
                _cf_rows_st.append({
                    '日期': _m_st['date'].strftime('%Y-%m-%d') if hasattr(_m_st['date'], 'strftime') else str(_m_st['date']),
                    '事件': '追繳', '帳面入金': 0, '平倉淨額': 0,
                    '客戶實際入金': 0, '追繳入金': _md_st,
                    '出金(損益)': 0, '平倉餘額出金': 0})
            _cf_df_st = pd.DataFrame(_cf_rows_st)
            if len(_cf_df_st) > 0:
                _cf_df_st = _cf_df_st.sort_values('日期').reset_index(drop=True)
                _cf_df_st['累計客戶淨現金流'] = (
                    _cf_df_st['客戶實際入金'] + _cf_df_st['追繳入金']
                    - _cf_df_st['出金(損益)'] - _cf_df_st['平倉餘額出金']
                ).cumsum()
                _cf_display = _cf_df_st.copy()
                for _cc_st in _cf_display.columns:
                    if _cc_st not in ('日期', '事件'):
                        _cf_display[_cc_st] = _cf_display[_cc_st].apply(lambda x: f"{x:,.0f}")
                st.dataframe(_cf_display, use_container_width=True, hide_index=True)

            # 出入金計算明細
            if results.daily_results:
                st.subheader("出入金計算明細")

                first_im = ts.iloc[0]['IM_today'] if len(ts) > 0 else 0
                build_date = results.daily_results[0].date

                # 摘要指標
                _TAX_RATE = 0.003  # 證交稅 0.3%
                total_in = first_im
                total_out = 0
                total_sale_net = 0
                total_customer_cash = first_im  # 建倉入金 = 客戶實際入金
                for _evt in (results.position_change_events or []):
                    _eq_after = _evt.get('equity_at_change', 0)
                    _new_im = _evt.get('new_im', 0)
                    _dep = max(0, _new_im - _eq_after)
                    total_in += _dep
                    _pnl_dets = _evt.get('realized_pnl_details', [])
                    _sp = sum(d['current_price'] * d['closed_qty'] for d in _pnl_dets if d['side'] == 'LONG')
                    _sp_tax = round(_sp * _TAX_RATE)
                    _sp_net = _sp - _sp_tax
                    _sp_surplus = max(0, _sp_net - _dep) if _dep > 0 else 0
                    total_sale_net += _sp_net
                    total_customer_cash += max(0, _dep - _sp_net)
                    total_out += _evt.get('withdrawal', 0) + _sp_surplus
                for _mc in (results.margin_call_events or []):
                    _mc_dep = _mc.get('required_deposit', 0)
                    total_in += _mc_dep
                    total_customer_cash += _mc_dep

                cf1, cf2, cf3, cf4 = st.columns(4)
                with cf1:
                    st.metric("總入金（帳面）", f"{total_in:,.0f}")
                with cf2:
                    st.metric("平倉淨額", f"{total_sale_net:,.0f}")
                with cf3:
                    st.metric("客戶實際入金", f"{total_customer_cash:,.0f}")
                with cf4:
                    st.metric("總出金", f"{total_out:,.0f}")

                # 資金流向總覽表
                flow_records = []
                flow_records.append({
                    '日期': build_date.strftime('%Y-%m-%d'),
                    '事件類型': '建倉入金',
                    '入金(帳面)': f"{first_im:,.0f}",
                    '平倉淨額': '0',
                    '客戶實際入金': f"{first_im:,.0f}",
                    '出金': '0',
                    '說明': f'建倉日 IM = {first_im:,.0f}',
                })
                for _evt in (results.position_change_events or []):
                    _eq_after = _evt.get('equity_at_change', 0)
                    _new_im = _evt.get('new_im', 0)
                    _dep = max(0, _new_im - _eq_after)
                    _wdl = _evt.get('withdrawal', 0)
                    _rpnl = _evt.get('realized_pnl', 0)
                    _pnl_dets = _evt.get('realized_pnl_details', [])
                    _sp = sum(d['current_price'] * d['closed_qty'] for d in _pnl_dets if d['side'] == 'LONG')
                    _sp_tax = round(_sp * _TAX_RATE)
                    _sp_net = _sp - _sp_tax
                    _cc = max(0, _dep - _sp_net)
                    _sp_surplus = max(0, _sp_net - _dep) if _dep > 0 else 0
                    _total_out = _wdl + _sp_surplus
                    flow_records.append({
                        '日期': _evt['date'].strftime('%Y-%m-%d'),
                        '事件類型': '加減倉',
                        '入金(帳面)': f"{_dep:,.0f}",
                        '平倉淨額': f"{_sp_net:,.0f}",
                        '客戶實際入金': f"{_cc:,.0f}",
                        '出金': f"{_total_out:,.0f}",
                        '說明': f'實現損益 {_rpnl:+,.0f} / 出金 {_total_out:,.0f} / 入金 {_dep:,.0f}',
                    })
                for _mc in (results.margin_call_events or []):
                    _dep = _mc.get('required_deposit', 0)
                    flow_records.append({
                        '日期': _mc['date'].strftime('%Y-%m-%d') if hasattr(_mc['date'], 'strftime') else str(_mc['date']),
                        '事件類型': '追繳入金',
                        '入金(帳面)': f"{_dep:,.0f}",
                        '平倉淨額': '0',
                        '客戶實際入金': f"{_dep:,.0f}",
                        '出金': '0',
                        '說明': f'追繳金額 = 新IM - 追繳前權益 = {_dep:,.0f}',
                    })
                st.dataframe(pd.DataFrame(flow_records), use_container_width=True, hide_index=True)

                # 逐事件計算明細
                st.markdown("#### 逐事件計算明細")

                # 建倉
                with st.expander(f"建倉入金 — {build_date.strftime('%Y-%m-%d')}", expanded=False):
                    st.markdown(f"""
| 項目 | 金額 |
|------|-----:|
| 建倉日 IM（Base_IM 大邊 + 小邊折減後） | **{first_im:,.0f}** |
| 入金 = IM | **{first_im:,.0f}** |
| MM = IM × 70% | **{first_im * 0.7:,.0f}** |
""")

                # 加減倉
                for _evt in (results.position_change_events or []):
                    _d = _evt['date'].strftime('%Y-%m-%d')
                    _eq_before = _evt.get('equity_before_change', 0)
                    _new_im = _evt.get('new_im', 0)
                    _new_mm = _evt.get('new_mm', 0)
                    _rpnl = _evt.get('realized_pnl', 0)
                    _wdl = _evt.get('withdrawal', 0)
                    _eq_after = _evt.get('equity_at_change', 0)
                    _dep = max(0, _new_im - _eq_after)
                    _eq_final = _eq_after + _dep
                    _cash_base = _evt.get('cash_base', _eq_before)
                    _max_wdl = max(0, min(_cash_base - _new_im, _eq_before - _new_im))

                    _old_im = _evt.get('old_im', 0)
                    _old_lmv = _evt.get('old_long_mv', 0)
                    _old_smv = _evt.get('old_short_mv', 0)
                    _new_lmv = _evt.get('long_mv', 0)
                    _new_smv = _evt.get('short_mv', 0)

                    with st.expander(f"加減倉 — {_d}", expanded=False):
                        st.markdown(f"""
**變動前（舊部位以當日價格結算）**

| 項目 | 金額 |
|------|-----:|
| 舊部位 多方MV / 空方MV | {_old_lmv:,.0f} / {_old_smv:,.0f} |
| 舊 IM | {_old_im:,.0f} |
| ① 變動前權益 | **{_eq_before:,.0f}** |

**變動後（新部位 IM 計算）**

| 項目 | 金額 |
|------|-----:|
| 新部位 多方MV / 空方MV | {_new_lmv:,.0f} / {_new_smv:,.0f} |
| ② 新 IM | **{_new_im:,.0f}** |

**實現損益與出金**

| 步驟 | 說明 | 金額 |
|------|------|-----:|
| ③ | 實現損益（平/減倉部位 × 基準價差） | **{_rpnl:+,.0f}** |
| ③½ | 現金基底（期初入金 + ③實現損益） | {_cash_base:,.0f} |
| ④ | 可出金（浮盈不可出金） | max(0, min({_cash_base:,.0f}, {_eq_before:,.0f}) - {_new_im:,.0f}) = **{_max_wdl:,.0f}** |
| ⑤ | 實際出金 | **-{_wdl:,.0f}** |
| ⑥ | 出金後權益 | {_eq_after:,.0f} |

**入金（補足至新 IM）**

| 步驟 | 說明 | 金額 |
|------|------|-----:|
| ⑦ | 加倉入金 = max(0, ②新IM - ⑥出金後權益) | max(0, {_new_im:,.0f}-{_eq_after:,.0f}) = **+{_dep:,.0f}** |
| ⑧ | 最終權益 | **{_eq_final:,.0f}** |
| ⑨ | 新 MM = 新IM × 70% | {_new_mm:,.0f} |
""")
                        # 逐部位實現損益明細
                        _pnl_details = _evt.get('realized_pnl_details', [])
                        if _pnl_details:
                            st.markdown("**逐部位實現損益：**")
                            _detail_recs = []
                            for _dd in _pnl_details:
                                _sl = '多' if _dd['side'] == 'LONG' else '空'
                                _detail_recs.append({
                                    '代號': _dd['code'], '方向': _sl,
                                    '變動': _dd['change_type'],
                                    '原數量': f"{_dd['old_qty']:,}",
                                    '新數量': f"{_dd['new_qty']:,}",
                                    '平/減量': f"{_dd['closed_qty']:,}",
                                    '基準價': f"{_dd['base_price']:.2f}",
                                    '當日價': f"{_dd['current_price']:.2f}",
                                    '實現損益': f"{_dd['pnl']:+,.0f}",
                                })
                            st.dataframe(pd.DataFrame(_detail_recs),
                                         use_container_width=True, hide_index=True)

                            # 平倉價金與實際現金流分析
                            _sell_dets = [_dd for _dd in _pnl_details if _dd['side'] == 'LONG']
                            if _sell_dets:
                                _sp_total = sum(_dd['current_price'] * _dd['closed_qty'] for _dd in _sell_dets)
                                _sp_tax = round(_sp_total * _TAX_RATE)
                                _sp_net = _sp_total - _sp_tax
                                _cust_cash = max(0, _dep - _sp_net)
                                _sp_surplus = max(0, _sp_net - _dep) if _dep > 0 else 0
                                st.markdown("**平倉價金與實際現金流分析：**")
                                _sp_lines = []
                                for _dd in _sell_dets:
                                    _sp_i = _dd['current_price'] * _dd['closed_qty']
                                    _sp_lines.append(f"| ⑩ {_dd['code']} 賣出價金 = {_dd['current_price']:,.2f} × {_dd['closed_qty']:,} | {_sp_i:,.0f} |")
                                if len(_sell_dets) > 1:
                                    _sp_lines.append(f"|   賣出價金合計 | **{_sp_total:,.0f}** |")
                                _sp_lines.append(f"| ⑪ 證交稅（{_TAX_RATE:.1%}） | -{_sp_tax:,.0f} |")
                                _sp_lines.append(f"| ⑫ 淨賣出價金 = ⑩ - ⑪ | **{_sp_net:,.0f}** |")
                                _sp_lines.append(f"| ⑬ 客戶實際入金 = max(0, ⑦加倉入金 - ⑫淨賣出價金) | max(0, {_dep:,.0f} - {_sp_net:,.0f}) = **{_cust_cash:,.0f}** |")
                                _sp_lines.append(f"|   其中來自賣出價金 | {min(_dep, _sp_net):,.0f} |")
                                _sp_lines.append(f"| ⑭ 賣出價金餘額可出金 = max(0, ⑫ - ⑦) | max(0, {_sp_net:,.0f} - {_dep:,.0f}) = **{_sp_surplus:,.0f}** |")
                                st.markdown("| 項目 | 金額 |\n|------|-----:|\n" + "\n".join(_sp_lines))
                        elif _rpnl == 0:
                            st.info("無平/減倉部位（純加倉）")

                # 追繳
                for _mc in (results.margin_call_events or []):
                    _mc_d = _mc['date'].strftime('%Y-%m-%d') if hasattr(_mc['date'], 'strftime') else str(_mc['date'])
                    _mc_im = _mc.get('im_today', 0)
                    _mc_dep = _mc.get('required_deposit', 0)

                    _mc_eq_before = 0
                    _mc_mm = 0
                    _mc_ts_mask = ts['date'] == pd.Timestamp(_mc['date'])
                    if _mc_ts_mask.any():
                        _mc_row = ts[_mc_ts_mask].iloc[0]
                        _mc_eq_before = _mc_row.get('Equity_Before', 0)
                        _mc_mm = _mc_row.get('MM_At_Call', 0)

                    with st.expander(f"追繳 — {_mc_d}", expanded=False):
                        st.markdown(f"""
| 項目 | 金額 |
|------|-----:|
| ① 追繳前權益 | **{_mc_eq_before:,.0f}** |
| ② 維持保證金(MM) | {_mc_mm:,.0f} |
| 觸發條件：①權益 < ②MM | **{_mc_eq_before:,.0f} < {_mc_mm:,.0f} → 觸發追繳** |
| ③ 當日新 IM | {_mc_im:,.0f} |
| ④ 追繳入金 = ③新IM - ①權益 | **+{_mc_dep:,.0f}** |
| ⑤ 追繳後權益 = 新IM | **{_mc_im:,.0f}** |
| ⑥ 新 MM = 新IM × 70% | {_mc_im * 0.7:,.0f} |
""")

            # 多空配對明細
            if results.daily_results:
                st.subheader("多空配對與減收明細")

                # 收集部位變動日期
                pos_change_dates = set()
                if results.position_change_events:
                    for evt in results.position_change_events:
                        pos_change_dates.add(pd.Timestamp(evt['date']))

                display_dates = []
                first_result = results.daily_results[0]
                display_dates.append(('建倉日', first_result))
                for dr in results.daily_results[1:]:
                    date_str = dr.date.strftime('%Y-%m-%d')
                    is_pos_change = dr.date in pos_change_dates
                    is_margin_call = dr.margin_result.margin_call
                    if is_pos_change and is_margin_call:
                        display_dates.append((f'加減倉+追繳 {date_str}', dr))
                    elif is_pos_change:
                        display_dates.append((f'加減倉 {date_str}', dr))
                    elif is_margin_call:
                        display_dates.append((f'追繳日 {date_str}', dr))

                if len(display_dates) > 1:
                    tab_names_h = [d[0] for d in display_dates]
                    tabs_h = st.tabs(tab_names_h)
                    for i, (label, dr) in enumerate(display_dates):
                        with tabs_h[i]:
                            hedge_df = dr.margin_result.hedge_pairing_df
                            mr = dr.margin_result
                            st.markdown(f"**{label}** - {dr.date.strftime('%Y-%m-%d')}")
                            if len(hedge_df) > 0:
                                hc1, hc2, hc3 = st.columns(3)
                                with hc1:
                                    st.metric("折減標的數", f"{len(hedge_df[hedge_df['總折減'] > 0])} 檔")
                                with hc2:
                                    st.metric("總折減IM", f"{hedge_df['總折減'].sum():,.0f}")
                                with hc3:
                                    st.metric("當日IM", f"{mr.im_today:,.0f}")
                                hedge_display = hedge_df.copy()
                                for col in hedge_display.columns:
                                    if col == '槓桿':
                                        hedge_display[col] = hedge_display[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) and isinstance(x, (int, float)) else x)
                                    elif col not in ['代碼', '產業桶', '減收類型']:
                                        hedge_display[col] = hedge_display[col].apply(lambda x: f'{x:,.0f}' if pd.notna(x) and isinstance(x, (int, float)) else x)
                                st.dataframe(hedge_display, use_container_width=True, height=250)
                                st.markdown("**折減來源分解：**")
                                bc1, bc2, bc3 = st.columns(3)
                                with bc1:
                                    st.metric("ETF完全對沖(100%)", f"{mr.reduction_etf_100:,.0f}")
                                with bc2:
                                    st.metric("同桶對沖", f"{mr.reduction_same_bucket:,.0f}")
                                with bc3:
                                    st.metric("跨桶對沖", f"{mr.reduction_cross_bucket:,.0f}")
                                bkt_df = mr.bucket_hedge_df
                                if bkt_df is not None and len(bkt_df) > 0:
                                    st.markdown("**各桶折減率判定：**")
                                    st.dataframe(bkt_df[['產業桶', '同桶折減率', '3M報酬差', '可對沖比例', '折減來源']],
                                                 use_container_width=True, height=140)
                            else:
                                st.info("無多空配對")
                else:
                    hedge_df = first_result.margin_result.hedge_pairing_df
                    mr = first_result.margin_result
                    st.markdown(f"**建倉日** - {first_result.date.strftime('%Y-%m-%d')}")
                    if len(hedge_df) > 0:
                        hc1, hc2, hc3 = st.columns(3)
                        with hc1:
                            st.metric("折減標的數", f"{len(hedge_df[hedge_df['總折減'] > 0])} 檔")
                        with hc2:
                            st.metric("總折減IM", f"{hedge_df['總折減'].sum():,.0f}")
                        with hc3:
                            st.metric("當日IM", f"{mr.im_today:,.0f}")
                        hedge_display = hedge_df.copy()
                        for col in hedge_display.columns:
                            if col == '槓桿':
                                hedge_display[col] = hedge_display[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) and isinstance(x, (int, float)) else x)
                            elif col not in ['代碼', '產業桶', '減收類型']:
                                hedge_display[col] = hedge_display[col].apply(lambda x: f'{x:,.0f}' if pd.notna(x) and isinstance(x, (int, float)) else x)
                        st.dataframe(hedge_display, use_container_width=True, height=300)
                        st.markdown("**折減來源分解：**")
                        bc1, bc2, bc3 = st.columns(3)
                        with bc1:
                            st.metric("ETF完全對沖(100%)", f"{mr.reduction_etf_100:,.0f}")
                        with bc2:
                            st.metric("同桶對沖", f"{mr.reduction_same_bucket:,.0f}")
                        with bc3:
                            st.metric("跨桶對沖", f"{mr.reduction_cross_bucket:,.0f}")
                        bkt_df = mr.bucket_hedge_df
                        if bkt_df is not None and len(bkt_df) > 0:
                            st.markdown("**各桶折減率判定：**")
                            st.dataframe(bkt_df[['產業桶', '同桶折減率', '3M報酬差', '可對沖比例', '折減來源']],
                                         use_container_width=True, height=140)
                    else:
                        st.info("無多空配對")

            # 追繳事件
            if results.margin_call_events:
                st.subheader("追繳事件")
                events_df = pd.DataFrame(results.margin_call_events)
                events_df['date'] = pd.to_datetime(events_df['date']).dt.strftime('%Y-%m-%d')
                for col in ['im_today', 'mm_today', 'equity', 'required_deposit']:
                    if col in events_df.columns:
                        events_df[col] = events_df[col].apply(lambda x: f"{x:,.0f}")
                st.dataframe(events_df, use_container_width=True)

            # 下載區
            st.subheader("一鍵存檔")

            html_report = create_html_report(results, position_schedule)
            st.download_button(
                label="下載完整報告 (HTML)",
                data=html_report.encode('utf-8'),
                file_name=f"手動建倉報告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True,
                type="primary"
            )
            st.caption("HTML 報告可直接用瀏覽器開啟，包含所有圖表與數據")

            with st.expander("其他匯出格式"):
                dl_col1, dl_col2, dl_col3 = st.columns(3)

                with dl_col1:
                    csv_buffer = io.StringIO()
                    ts.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="下載逐日明細 CSV",
                        data=csv_buffer.getvalue().encode('utf-8-sig'),
                        file_name=f"margin_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv", use_container_width=True
                    )

                with dl_col2:
                    excel_data = create_full_report_excel(results, position_schedule)
                    st.download_button(
                        label="下載完整報告 (Excel)",
                        data=excel_data,
                        file_name=f"手動建倉報告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                with dl_col3:
                    zip_data = create_audit_zip(results, position_schedule)
                    st.download_button(
                        label="下載稽核包 ZIP",
                        data=zip_data,
                        file_name=f"audit_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip", use_container_width=True
                    )

            # 假設說明
            with st.expander("假設與保守口徑說明"):
                for assumption in results.assumptions:
                    st.write(f"- {assumption}")
                if results.missing_codes:
                    st.warning(f"發現 {len(results.missing_codes)} 檔缺碼，以保守口徑處理")
                    st.write("缺碼清單（前 20 檔）：")
                    st.code(", ".join(results.missing_codes[:20]))


if __name__ == "__main__":
    main()
