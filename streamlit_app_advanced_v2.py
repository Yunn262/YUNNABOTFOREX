# streamlit_app_all_in_one.py
# Single-file Streamlit app: Multi-exchange data, signals, alerts, optional ML models and SHAP.
# Requirements (example): pip install streamlit pandas numpy ccxt joblib requests scikit-learn ta tensorflow shap streamlit-autorefresh
# If you don't use models or shap, those packages are optional.

import os
import io
import base64
from datetime import datetime
import time
import traceback

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import joblib
import requests

# Try to import optional libs
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False

# Optional tensorflow/shap imports guarded
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------------------
# UI / Page config and CSS
# ---------------------------
st.set_page_config(page_title="🚀 Bot Trading Pro — All-in-One", layout="wide")

st.markdown(
    """
    <style>
    body { background-color: #0b0b0b; color: #e6e6e6; }
    .stApp { background-color: #0b0b0b; }
    .block-container { padding: 1rem 2rem; }
    .signal-box { padding: 1rem; border-radius: 0.75rem; color: #fff; text-align: center; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 Bot Trading Pro — Multi-Exchange (All-in-One)")

# ---------------------------
# Sounds (local fallback -> base64)
# ---------------------------
# Short base64 WAV placeholders (very small). Replace with better sounds if wanted.
SOUND_UP_B64 = (
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA="
)
SOUND_DOWN_B64 = (
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA="
)


def play_sound(local_path=None, sound_b64=None):
    """
    Play a sound: prefer local_path if exists, else fallback to base64 bytes.
    """
    try:
        if local_path and os.path.exists(local_path):
            with open(local_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")
            return
    except Exception:
        pass
    try:
        if sound_b64:
            audio_bytes = base64.b64decode(sound_b64)
            st.audio(io.BytesIO(audio_bytes), format="audio/wav")
    except Exception:
        pass


# ---------------------------
# Utilities: fetch multi-exchange OHLCV
# ---------------------------
@st.cache_data(ttl=20)
def fetch_data_multi(symbol: str, timeframe: str = "1m", limit: int = 300, exchanges_to_try=None):
    """
    Attempts to fetch OHLCV from multiple exchanges via ccxt.
    Returns combined DataFrame (index = timestamp) and dict counts per source.
    """
    if exchanges_to_try is None:
        exchanges_to_try = ["binance", "coinbasepro", "kucoin", "kraken"]

    rows = []
    sources = {}
    # Normalize input symbol: user may input BTC/USDT or BTC-USDT or BTCUSDT
    sym_input = symbol.strip()
    if "/" not in sym_input and "-" not in sym_input:
        # try common formats: BTC/USDT
        sym_options = [sym_input, sym_input.replace("USDT", "/USDT"), sym_input.replace("USD", "/USD")]
    else:
        sym_options = [sym_input.replace("-", "/")]

    for ex_name in exchanges_to_try:
        count = 0
        for try_sym in sym_options:
            try:
                ex_class = getattr(ccxt, ex_name)
                ex = ex_class({"enableRateLimit": True})
                # Some exchanges may need load_markets
                try:
                    # try to load markets (some ccxt builds require this)
                    ex.load_markets()
                except Exception:
                    pass
                ohlcv = ex.fetch_ohlcv(try_sym, timeframe=timeframe, limit=limit)
                if not ohlcv:
                    continue
                df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                df.set_index("ts", inplace=True)
                df["exchange"] = ex_name
                rows.append(df)
                count = len(df)
                break  # stop trying other symbol formats for this exchange
            except Exception:
                # try alternative: replace USDT -> USD
                continue
        sources[ex_name] = count
    if len(rows) == 0:
        raise RuntimeError("No exchange returned data for symbol/timeframe. Check symbol and network.")
    combined = pd.concat(rows)
    # remove duplicated index keeping last (in case same timestamp present)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return combined, sources


# ---------------------------
# Technical features (same as training)
# ---------------------------
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    # compute simple indicators without external heavy libs to avoid extra deps
    df = df.copy()
    df["close"] = df["close"].astype(float)
    # EMA
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    # MACD simple
    df["macd"] = df["ema12"] - df["ema26"]
    # RSI (simple implementation)
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    # ATR approx
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["momentum"] = df["close"] - df["close"].shift(5)
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)
    df = df.fillna(method="bfill").fillna(0)
    return df


# ---------------------------
# Load artefacts (optional)
# ---------------------------
@st.cache_data(ttl=600)
def load_artefacts():
    arte = {"scaler": None, "models_list": None, "stack_clf": None}
    try:
        arte["scaler"] = joblib.load("scaler.gz")
    except Exception:
        arte["scaler"] = None
    try:
        arte["models_list"] = joblib.load("models_list.gz")
    except Exception:
        arte["models_list"] = None
    try:
        arte["stack_clf"] = joblib.load("stack_calibrator.gz")
    except Exception:
        arte["stack_clf"] = None
    return arte


arte = load_artefacts()

# ---------------------------
# Simple backtest util (expiration in candles)
# ---------------------------
def backtest_signals(df: pd.DataFrame, signals_df: pd.DataFrame, sl=0.002, tp=0.003):
    """
    signals_df should contain: entry_index (int position in df), signal_type ('CALL'/'PUT'), expiration_candles (int)
    returns stats dict
    """
    results = []
    for _, r in signals_df.iterrows():
        idx = int(r["entry_index"])
        exp = int(r["expiration_candles"])
        entry = df["close"].iloc[idx]
        exit_idx = min(idx + exp, len(df) - 1)
        exit_price = df["close"].iloc[exit_idx]
        if r["signal_type"] == "CALL":
            pnl = (exit_price - entry) / entry
        else:
            pnl = (entry - exit_price) / entry
        results.append(pnl)
    res = pd.Series(results)
    stats = {"trades": len(res), "avg_pnl": res.mean() if len(res) else 0.0, "winrate": (res > 0).mean() if len(res) else 0.0, "total": res.sum() if len(res) else 0.0}
    return stats


# ---------------------------
# Signal generator
# ---------------------------
def generate_signal(df_merged: pd.DataFrame, arte: dict, exp: str):
    """
    Returns (signal_str, confidence_float, details_dict).
    Tries to use models if present. Otherwise, use heuristic based on short momentum + RSI.
    """
    df = add_technical_features(df_merged.copy())

    # If models exist, try to do prediction using ensemble stacking
    try:
        if arte.get("models_list") and arte.get("stack_clf") and arte.get("scaler") and TF_AVAILABLE:
            features = ["open", "high", "low", "close", "volume", "ema12", "ema26", "macd", "rsi", "atr", "momentum", "vwap"]
            seq_len = 50
            if len(df) >= seq_len:
                vals = df[features].values[-seq_len:]
                resh = vals.reshape(-1, vals.shape[-1])
                scaled = arte["scaler"].transform(resh).reshape(1, seq_len, vals.shape[-1])
                preds = []
                from tensorflow.keras.models import load_model
                for lstm_path, trans_path in arte["models_list"]:
                    try:
                        m1 = load_model(lstm_path, compile=False)
                        m2 = load_model(trans_path, compile=False)
                        p1 = float(m1.predict(scaled, verbose=0)[0][0])
                        p2 = float(m2.predict(scaled, verbose=0)[0][0])
                        preds.append(p1)
                        preds.append(p2)
                    except Exception:
                        continue
                if len(preds) > 0:
                    stack_X = np.array(preds).reshape(1, -1)
                    prob = float(arte["stack_clf"].predict_proba(stack_X)[0][1])
                    signal = "SUBIDA 🔼" if prob > 0.5 else "DESCIDA 🔽"
                    return signal, prob * 100.0, {"method": "model"}
    except Exception:
        # if any model step fails, fall back to heuristic
        pass

    # Heuristic fallback
    try:
        df["mom"] = df["close"].pct_change().rolling(window=3).mean()
        mom = float(df["mom"].iloc[-1])
    except Exception:
        mom = 0.0
    rsi = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0

    # Build a confidence score (0-100)
    score = 50.0
    # momentum contribution (scaled)
    score += np.clip(mom * 1000.0, -30.0, 30.0)
    # RSI: push away from neutral 50
    score += np.clip((50.0 - rsi) * -0.2, -20.0, 20.0)
    prob = float(np.clip(score, 5.0, 99.0))

    if mom > 0.0008:
        signal = "SUBIDA 🔼"
    elif mom < -0.0008:
        signal = "DESCIDA 🔽"
    else:
        signal = "NEUTRAL ⚪"

    return signal, prob, {"method": "heuristic", "mom": mom, "rsi": rsi}


# ---------------------------
# SHAP explainability
# ---------------------------
@st.cache_data(ttl=300)
def compute_shap_local(arte: dict, df: pd.DataFrame):
    """
    Compute SHAP values using a lightweight path: load first model and do DeepExplainer.
    Only runs if shap + tensorflow available and artefacts exist.
    Returns dict feature -> importance or None.
    """
    if not SHAP_AVAILABLE or not TF_AVAILABLE:
        return None
    try:
        features = ["open", "high", "low", "close", "volume", "ema12", "ema26", "macd", "rsi", "atr", "momentum", "vwap"]
        seq_len = 50
        df2 = add_technical_features(df.copy())
        if len(df2) < seq_len:
            return None
        vals = df2[features].values[-seq_len:]
        resh = vals.reshape(-1, vals.shape[-1])
        scaled = arte["scaler"].transform(resh).reshape(1, seq_len, vals.shape[-1])
        lstm_path, trans_path = arte["models_list"][0]
        from tensorflow.keras.models import load_model
        model = load_model(lstm_path, compile=False)
        # Use small background
        background = scaled
        explainer = shap.DeepExplainer(model, background)
        shap_vals = explainer.shap_values(scaled)[0]  # shape (seq_len, nfeat)
        shap_by_feat = np.mean(np.abs(shap_vals), axis=0)  # average across timesteps
        return dict(zip(features, shap_by_feat.tolist()))
    except Exception:
        return None


# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Configurações")
symbol = st.sidebar.text_input("Símbolo (ex: BTC/USDT)", "BTC/USDT")
exchange_select = st.sidebar.selectbox("Fonte preferencial (All combina)", ["All", "binance", "coinbasepro", "kucoin", "kraken"])
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h"])
exp = st.sidebar.selectbox("Expiração do sinal", ["1m", "5m", "15m"])
confidence_threshold = st.sidebar.slider("Nível mínimo de confiança p/ alerta (%)", 50, 100, 75, 1)
auto_refresh = st.sidebar.checkbox("Atualizar automaticamente", value=False)
interval = st.sidebar.number_input("Intervalo (s) - autorefresh", min_value=5, max_value=600, value=30)

with st.sidebar.expander("Telegram (opcional)"):
    TG_TOKEN = st.text_input("Telegram bot token")
    TG_CHAT = st.text_input("Chat id")

# Autorefresh mechanism
if auto_refresh and AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=interval * 1000, key="auto-refresh-key")

# ---------------------------
# Main action
# ---------------------------
col1, col2 = st.columns([3, 1])

with col1:
    analyze_clicked = st.button("🔮 Analisar mercado agora")
    shap_button = st.button("🧠 Gerar explicação SHAP (se disponível)")
    simulate_backtest = st.button("⚙️ Simular Backtest Rápido (últimas candles)")

with col2:
    st.write("Artefatos:")
    st.write(
        {
            "scaler": bool(arte.get("scaler")),
            "models_list": bool(arte.get("models_list")),
            "stack_clf": bool(arte.get("stack_clf")),
            "tensorflow": TF_AVAILABLE,
            "shap": SHAP_AVAILABLE,
        }
    )
    st.write("Sons locais (./sounds/up.wav, ./sounds/down.wav):")
    st.write({"up_exists": os.path.exists("sounds/up.wav"), "down_exists": os.path.exists("sounds/down.wav")})

# Run analysis if requested or auto_refresh triggers
if analyze_clicked or (auto_refresh and AUTOREFRESH_AVAILABLE):
    with st.spinner("Recolhendo dados e gerando sinal..."):
        try:
            exchanges_try = ["binance", "coinbasepro", "kucoin", "kraken"] if exchange_select == "All" else [exchange_select]
            df_combined, sources = fetch_data_multi(symbol, timeframe, limit=400, exchanges_to_try=exchanges_try)

            if exchange_select != "All":
                df_use = df_combined[df_combined["exchange"] == exchange_select]
            else:
                # aggregate combined rows by index
                df_use = df_combined.groupby(df_combined.index).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                )

            if df_use.empty:
                st.error("Dados insuficientes para o símbolo/timeframe selecionado.")
            else:
                signal, confidence, details = generate_signal(df_use, arte, exp)
                last_price = df_use["close"].iloc[-1]
                st.metric("Preço atual", f"{last_price:.8f}")
                srcs = ", ".join([f"{k}:{v}" for k, v in sources.items() if v > 0])
                st.write(f"Fontes: {srcs}")
                st.markdown(f"### Sinal: **{signal}** — Confiança: **{confidence:.2f}%**")
                st.write("Detalhes:", details)

                # Visual box
                color = "#1db954" if "SUBIDA" in signal else "#e63946" if "DESCIDA" in signal else "#6c757d"
                st.markdown(
                    f"<div class='signal-box' style='background:{color};'>"
                    f"<div style='font-size:1.1rem'>{signal}</div>"
                    f"<div style='font-size:0.9rem'>Confiança: {confidence:.2f}%</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                # play sound if above threshold
                if confidence >= confidence_threshold:
                    if "SUBIDA" in signal:
                        play_sound(local_path="sounds/up.wav", sound_b64=SOUND_UP_B64)
                    elif "DESCIDA" in signal:
                        play_sound(local_path="sounds/down.wav", sound_b64=SOUND_DOWN_B64)

                # send telegram if configured
                if TG_TOKEN and TG_CHAT:
                    try:
                        txt = f"SINAL: {signal}\nAtivo: {symbol} — TF: {timeframe} — Exp: {exp}\nProb: {confidence:.2f}%\nHora: {datetime.utcnow().isoformat()}Z"
                        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", json={"chat_id": TG_CHAT, "text": txt}, timeout=8)
                    except Exception:
                        pass

                # show chart
                st.line_chart(df_use["close"].tail(300))

        except Exception as e:
            st.error(f"Erro ao buscar/processar dados: {e}")
            st.write(traceback.format_exc())

# SHAP
if shap_button:
    with st.spinner("Calculando SHAP (pode ser lento)..."):
        try:
            exchanges_try = ["binance", "coinbasepro", "kucoin", "kraken"] if exchange_select == "All" else [exchange_select]
            df_combined, sources = fetch_data_multi(symbol, timeframe, limit=500, exchanges_to_try=exchanges_try)
            if exchange_select != "All":
                df_use = df_combined[df_combined["exchange"] == exchange_select]
            else:
                df_use = df_combined.groupby(df_combined.index).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                )
            shap_vals = compute_shap_local(arte, df_use)
            if shap_vals is None:
                st.warning("SHAP não disponível (verifica se instalaste 'shap' e tens modelos tf salvos).")
            else:
                feat_df = pd.DataFrame(list(shap_vals.items()), columns=["feature", "importance"]).sort_values("importance", ascending=False).set_index("feature")
                st.subheader("SHAP — top features")
                st.bar_chart(feat_df.head(12))
        except Exception as e:
            st.error(f"Erro no cálculo SHAP: {e}")
            st.write(traceback.format_exc())

# Quick backtest simulation using current heuristic signals
if simulate_backtest:
    with st.spinner("Simulando backtest rápido..."):
        try:
            # fetch historical candles from a single exchange (preferred or first available)
            exchanges_try = [exchange_select] if exchange_select != "All" else ["binance", "coinbasepro", "kucoin", "kraken"]
            df_combined, sources = fetch_data_multi(symbol, timeframe, limit=1000, exchanges_to_try=exchanges_try)
            if exchange_select != "All":
                df_use = df_combined[df_combined["exchange"] == exchange_select]
            else:
                df_use = df_combined.groupby(df_combined.index).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                )
            df_use = add_technical_features(df_use)
            # produce signals by sliding window heuristic for demonstration
            signals = []
            for i in range(60, len(df_use) - 5):
                window = df_use.iloc[: i + 1]
                sig, conf, _ = generate_signal(window, arte, exp)
                entry_index = i
                signal_type = "CALL" if "SUBIDA" in sig else "PUT" if "DESCIDA" in sig else "NEUTRAL"
                expiration_candles = 1 if exp == "1m" else 5 if exp == "5m" else 15
                if signal_type != "NEUTRAL":
                    signals.append({"entry_index": entry_index, "signal_type": signal_type, "expiration_candles": expiration_candles})
            signals_df = pd.DataFrame(signals)
            stats = backtest_signals(df_use.reset_index(drop=True), signals_df)
            st.write("Backtest rápido — métricas:", stats)
        except Exception as e:
            st.error(f"Erro no backtest: {e}")
            st.write(traceback.format_exc())

st.markdown("---")
st.caption("Streamlit app — Multi-exchange. Guardar ficheiros opcionais: sounds/up.wav, sounds/down.wav. Use models (scaler.gz, models_list.gz, stack_calibrator.gz) para previsões ML.")
