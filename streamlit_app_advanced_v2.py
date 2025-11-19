# trading_bot_streamlit.py (atualizado)
# Versão com atualização não-bloqueante a cada 30s e botão manual

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import time
import base64
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
except:
    xgb = None
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except:
    Sequential = None
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ----------------- CONFIG -----------------
st.set_page_config(page_title="AI Trading Bot", layout="wide")

# Exchanges e símbolos
EXCHANGES = {
    'Binance': ccxt.binance({'enableRateLimit': True}),
    'KuCoin': ccxt.kucoin({'enableRateLimit': True}),
    'Kraken': ccxt.kraken({'enableRateLimit': True}),
    'Coinbase': ccxt.coinbase({'enableRateLimit': True})
}
SYMBOLS = ['BTC/USDT','ETH/USDT','SOL/USDT','ADA/USDT','XAU/USD','EUR/USD','GBP/USD']
TIMEFRAMES = ['1m','5m','15m','30m','1h','4h']

# CSS dark + animações
st.markdown("""
<style>
body { background-color: #000; color: #fff; }
.metric-card { background: linear-gradient(90deg, #0f172a, #001219); padding: 12px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); }
.signal-up { background: linear-gradient(90deg,#04290f,#058a1f); color: white; padding:14px; border-radius:10px; font-weight:700; animation: pulse 1.2s infinite;}
.signal-down { background: linear-gradient(90deg,#2b0606,#8b0000); color: white; padding:14px; border-radius:10px; font-weight:700; animation: pulse 1.2s infinite;}
@keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05);} 100% { transform: scale(1); } }
.small-muted { color: #94a3b8; font-size:12px }
</style>
""", unsafe_allow_html=True)

# ----------------- Helpers -----------------
@st.cache_data(ttl=30)
def safe_load_markets(_exchange):
    try:
        return _exchange.load_markets()
    except:
        return {}

@st.cache_data(ttl=15)
def fetch_ohlcv(_exchange, symbol, timeframe, limit=500):
    try:
        markets = safe_load_markets(_exchange)
        chosen = symbol
        if symbol not in markets and symbol.replace('/','') in markets:
            chosen = symbol.replace('/','')
        ohlcv = _exchange.fetch_ohlcv(chosen, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        for name, ex in EXCHANGES.items():
            if ex != _exchange:
                try:
                    df = fetch_ohlcv(ex, symbol, timeframe, limit)
                    st.info(f'Fallback: {name}')
                    return df
                except:
                    continue
        st.warning('Todas as exchanges falharam.')
        return pd.DataFrame()

# Features
def add_features(df):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['hl_range'] = (df['high'] - df['low']) / (df['open']+1e-9)
    df['oc_change'] = (df['close'] - df['open']) / (df['open']+1e-9)
    df['sma5'] = df['close'].rolling(5).mean()
    df['sma14'] = df['close'].rolling(14).mean()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + 1e-9)
    df['rsi14'] = 100 - (100 / (1 + rs))
    return df.dropna()

# Candle patterns
def is_hammer(o,h,l,c): return (min(o,c)-l > 2*abs(c-o)) and (h-max(o,c) < 0.25*abs(c-o))
def is_doji(o,h,l,c,tol=0.15): return abs(c-o) <= tol*(h-l+1e-9)
def bullish_engulf(prev,o,h,l,c,po,ph,pl,pc): return (pc<po) and (c>o) and (c>po) and (o<pc)
def bearish_engulf(prev,o,h,l,c,po,ph,pl,pc): return (pc>po) and (c<o) and (o>pc) and (c<po)

# ----------------- UI -----------------
st.title('🖤 AI Trading Bot - Responsivo')
col1, col2 = st.columns([1,2])
with col1:
    exchange_choice = st.selectbox('Exchange', list(EXCHANGES.keys()))
    symbol_choice = st.selectbox('Símbolo', SYMBOLS)
    timeframe_choice = st.selectbox('Timeframe', TIMEFRAMES, index=2)
    start_live = st.button('▶️ Start Live')
    stop_live = st.button('⏹ Stop')
    manual_run = st.button('⚡ Atualizar Manualmente')
with col2:
    model_choice = st.selectbox('Modelo IA', ['RandomForest','XGBoost','LSTM'])
    run_train = st.button('Treinar/Atualizar Modelo')

exchange = EXCHANGES[exchange_choice]
chart_placeholder = st.empty()
stats_placeholder = st.empty()
patterns_placeholder = st.empty()

if 'live' not in st.session_state: st.session_state.live = False
if start_live: st.session_state.live = True
if stop_live: st.session_state.live = False

# Função para atualizar dashboard
def update_dashboard():
    df_raw = fetch_ohlcv(exchange, symbol_choice, timeframe_choice, limit=300)
    if df_raw.empty:
        st.warning('Sem dados.')
        return
    df_feat = add_features(df_raw)
    last = df_feat.iloc[-1]
    prev = df_feat.iloc[-2]
    signal = 'SUBIR' if last['close']>prev['close'] else 'DESCER'
    color_class = 'signal-up' if signal=='SUBIR' else 'signal-down'
    stats_placeholder.markdown(f"<div class='{color_class}'>Último: {last['close']:.2f} | Sinal: {signal}</div>", unsafe_allow_html=True)
    patterns=[]
    if is_hammer(last['open'],last['high'],last['low'],last['close']): patterns.append('Hammer')
    if is_doji(last['open'],last['high'],last['low'],last['close']): patterns.append('Doji')
    if bullish_engulf(None,last['open'],last['high'],last['low'],last['close'],prev['open'],prev['high'],prev['low'],prev['close']): patterns.append('Bull Engulf')
    if bearish_engulf(None,last['open'],last['high'],last['low'],last['close'],prev['open'],prev['high'],prev['low'],prev['close']): patterns.append('Bear Engulf')
    patterns_placeholder.markdown(f"<div class='metric-card'><b>Padrões:</b> {', '.join(patterns) if patterns else 'Nenhum'}</div>", unsafe_allow_html=True)
    fig = go.Figure(data=[go.Candlestick(x=df_feat.index, open=df_feat['open'], high=df_feat['high'], low=df_feat['low'], close=df_feat['close'])])
    fig.update_layout(template='plotly_dark', height=600)
    chart_placeholder.plotly_chart(fig, use_container_width=True)

# Atualização automática a cada 30s
if st.session_state.live:
    count = st_autorefresh(interval=30000, limit=None, key='datarefresh')
    update_dashboard()

# Atualização manual
if manual_run:
    update_dashboard()

if not st.session_state.live and not manual_run:
    st.info('Clique ▶️ Start Live ou ⚡ Atualizar Manualmente para carregar dados.')
