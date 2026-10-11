"""
AI Trading Arena V6 — Order Book, Depth, & Interactive AI Tracking.

Features:
- Live limit order book generation and market depth visualization
- Interactive AI highlighting (click an AI to isolate its data on charts)
- Trade profit tracking (dotted lines drawn from entry to exit)
- Advanced metrics: Sharpe Ratio and Profit Factor
- Rank history sparkline in the leaderboard
"""
from __future__ import annotations
import sys
import math
import random
from dataclasses import dataclass, field
from collections import deque
from typing import List, Callable, Optional, Tuple

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, QElapsedTimer
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPainterPath,
    QLinearGradient, QRadialGradient, QPolygonF
)

SEED = 42
random.seed(SEED)

# ── Palette ────────────────────────────────────────────────────────────
COLORS = {
    'bg':          QColor("#050810"),
    'panel':       QColor("#0d1320"),
    'panel_alt':   QColor("#121a2b"),
    'grid':        QColor("#1a2238"),
    'text':        QColor("#e4e9f2"),
    'text_dim':    QColor("#8b95a8"),
    'text_dimmer': QColor("#4a5470"),
    'accent':      QColor("#00d4ff"),
    'green':       QColor("#00ff88"),
    'red':         QColor("#ff4757"),
    'yellow':      QColor("#ffd93d"),
    'purple':      QColor("#a78bfa"),
    'border':      QColor("#1f2942"),
}
AI_PALETTE = [
    QColor("#00d4ff"), QColor("#ff6b9d"), QColor("#ffd93d"),
    QColor("#00ff88"), QColor("#a78bfa"), QColor("#ff8c42"),
    QColor("#4ade80"), QColor("#e879f9")
]

# ── Data ────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    ai_index: int
    action: str
    qty: int
    price: float
    tick: int

@dataclass
class ClosedTrade:
    entry_tick: int
    entry_price: float
    exit_tick: int
    exit_price: float
    qty: int
    profit: float

@dataclass
class AITrader:
    name: str
    color: QColor
    strategy: Callable
    cash: float = 10000.0
    shares: int = 0
    starting_cash: float = 10000.0
    history: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    closed_trades: List[ClosedTrade] = field(default_factory=list)
    last_action: str = "INIT"
    state_text: str = "Booting..."
    pulse: float = 0.0
    peak_value: float = 10000.0
    max_drawdown: float = 0.0
    wins: int = 0
    losses: int = 0
    avg_entry: float = 0.0
    entry_tick: int = -1
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    rank_history: List[int] = field(default_factory=list)

    def value_at(self, price: float) -> float:
        return self.cash + self.shares * price

    def pct_at(self, price: float) -> float:
        return (self.value_at(price) / self.starting_cash - 1.0) * 100.0

    def reset(self) -> None:
        self.cash = self.starting_cash
        self.shares = 0
        self.history.clear()
        self.trades.clear()
        self.closed_trades.clear()
        self.last_action = "INIT"
        self.state_text = "Booting..."
        self.pulse = 0.0
        self.peak_value = self.starting_cash
        self.max_drawdown = 0.0
        self.wins = 0
        self.losses = 0
        self.avg_entry = 0.0
        self.entry_tick = -1
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.rank_history.clear()

# ── Indicators ─────────────────────────────────────────────────────────
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None, []
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [c if c > 0 else 0 for c in changes]
    losses = [-c if c < 0 else 0 for c in changes]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsis = []
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100.0 - 100.0 / (1.0 + rs))
    return rsis[-1], rsis

def ema(data, period):
    if len(data) < period: return None, []
    k = 2 / (period + 1)
    ema_val = sum(data[:period]) / period
    ema_hist = [ema_val]
    for p in data[period:]:
        ema_val = p * k + ema_val * (1 - k)
        ema_hist.append(ema_val)
    return ema_val, ema_hist

# ── Strategies ──────────────────────────────────────────────────────────
def strat_momentum(ai, prices, vols, tick):
    if len(prices) < 30: return None, "Need more data"
    ma_s = sum(prices[-10:]) / 10
    ma_l = sum(prices[-30:]) / 30
    diff_pct = (ma_s - ma_l) / ma_l * 100
    if diff_pct > 0.2 and ai.shares >= 0:
        q = int(ai.starting_cash * 0.2 / prices[-1])
        return ('BUY', q), f"MA Cross↑ {diff_pct:.1f}%"
    if diff_pct < -0.2 and ai.shares <= 0:
        q = int(ai.starting_cash * 0.2 / prices[-1])
        return ('SELL', q), f"MA Cross↓ {diff_pct:.1f}%"
    return None, f"Wait MA {diff_pct:.1f}%"

def strat_mean_rev(ai, prices, vols, tick):
    if len(prices) < 40: return None, "Need more data"
    w = prices[-40:]
    mean = sum(w) / len(w)
    var = sum((p - mean) ** 2 for p in w) / len(w)
    std = math.sqrt(var)
    if std < 1e-6: return None, "Vol too low"
    upper, lower = mean + 2 * std, mean - 2 * std
    p = prices[-1]
    dev = (p - mean) / std
    if p < lower and ai.shares >= 0:
        q = int(ai.starting_cash * 0.3 / p)
        return ('BUY', q), f"Dev {dev:.1f}σ Buy"
    if p > upper and ai.shares <= 0:
        q = int(ai.starting_cash * 0.3 / p)
        return ('SELL', q), f"Dev {dev:.1f}σ Short"
    return None, f"Dev {dev:.1f}σ Hold"

def strat_rsi(ai, prices, vols, tick):
    rsi, _ = calculate_rsi(prices, 14)
    if rsi is None: return None, "Need more data"
    p = prices[-1]
    if rsi < 35 and ai.shares >= 0:
        q = int(ai.starting_cash * 0.35 / p)
        return ('BUY', q), f"RSI {rsi:.0f} Oversold"
    if rsi > 65 and ai.shares <= 0:
        q = int(ai.starting_cash * 0.35 / p)
        return ('SELL', q), f"RSI {rsi:.0f} Overbought"
    return None, f"RSI {rsi:.0f} Neutral"

def strat_scalper(ai, prices, vols, tick):
    if len(prices) < 8: return None, "Need more data"
    p = prices[-1]
    avg = sum(prices[-8:]) / 8
    diff = (p - avg) / avg * 100
    if diff < -0.3 and ai.shares >= 0:
        q = int(ai.starting_cash * 0.15 / p)
        return ('BUY', q), f"Scalp {diff:.2f}%"
    if diff > 0.3 and ai.shares <= 0:
        q = int(ai.starting_cash * 0.15 / p)
        return ('SELL', q), f"Scalp {diff:.2f}%"
    return None, f"Wait {diff:.2f}%"

def strat_trend(ai, prices, vols, tick):
    if len(prices) < 50: return None, "Need more data"
    w = prices[-50:]
    n = len(w)
    x_avg = (n - 1) / 2.0
    y_avg = sum(w) / n
    num = sum((i - x_avg) * (w[i] - y_avg) for i in range(n))
    den = sum((i - x_avg) ** 2 for i in range(n))
    slope = num / den if den > 0 else 0
    p = prices[-1]
    if slope > 0.1 and ai.shares >= 0:
        q = int(ai.starting_cash * 0.2 / p)
        return ('BUY', q), f"Trend ↑ {slope:.2f}"
    if slope < -0.1 and ai.shares <= 0:
        q = int(ai.starting_cash * 0.2 / p)
        return ('SELL', q), f"Trend ↓ {slope:.2f}"
    return None, f"Flat {slope:.2f}"

def strat_contrarian(ai, prices, vols, tick):
    if len(prices) < 25: return None, "Need more data"
    avg = sum(prices[-25:]) / 25
    p = prices[-1]
    diff = (p - avg) / avg * 100
    if diff > 2.5 and ai.shares <= 0:
        q = int(ai.starting_cash * 0.25 / p)
        return ('SELL', q), f"Fade +{diff:.1f}%"
    if diff < -2.5 and ai.shares >= 0:
        q = int(ai.starting_cash * 0.25 / p)
        return ('BUY', q), f"Fade {diff:.1f}%"
    return None, f"Wait {diff:.1f}%"

def strat_breakout(ai, prices, vols, tick):
    if len(prices) < 50: return None, "Need data"
    w = prices[-50:]
    high = max(w)
    low = min(w)
    p = prices[-1]
    if p > high - 0.01 and ai.shares >= 0:
        q = int(ai.starting_cash * 0.2 / p)
        return ('BUY', q), f"Brk ↑ {high:.2f}"
    if p < low + 0.01 and ai.shares <= 0:
        q = int(ai.starting_cash * 0.2 / p)
        return ('SELL', q), f"Brk ↓ {low:.2f}"
    return None, f"Rng [{low:.2f}-{high:.2f}]"

def strat_macd(ai, prices, vols, tick):
    if len(prices) < 35: return None, "Need data"
    ema12_val, ema12_hist = ema(prices, 12)
    ema26_val, ema26_hist = ema(prices, 26)
    if not ema12_hist or not ema26_hist: return None, "Calc EMA"
    min_len = min(len(ema12_hist), len(ema26_hist))
    macd_hist = [ema12_hist[-min_len:][i] - ema26_hist[-min_len:][i] for i in range(min_len)]
    if len(macd_hist) < 9: return None, "Calc MACD"
    signal_val, signal_hist = ema(macd_hist, 9)
    macd_line = macd_hist[-1]
    prev_macd = macd_hist[-2]
    prev_signal = signal_hist[-2]
    p = prices[-1]
    if prev_macd < prev_signal and macd_line > signal_val and ai.shares >= 0:
        q = int(ai.starting_cash * 0.2 / p)
        return ('BUY', q), f"MACD Cross↑"
    if prev_macd > prev_signal and macd_line < signal_val and ai.shares <= 0:
        q = int(ai.starting_cash * 0.2 / p)
        return ('SELL', q), f"MACD Cross↓"
    return None, f"MACD {macd_line:.2f}"

# ── Market simulator ────────────────────────────────────────────────────
class StockSimulator:
    def __init__(self, start_price: float = 100.0):
        self.start_price = start_price
        self.price = start_price
        self.history: List[float] = [start_price]
        self.candles: List[dict] = [{'o': start_price, 'h': start_price, 'l': start_price, 'c': start_price, 'v': 5000}]
        self.trend = 0.0003
        self.volatility = 0.008
        self.target = 100.0
        self.ticks = 0
        self.news_text = ""
        self.news_timer = 0
        self.news_events = [
            ("Fed hikes interest rates", -0.04),
            ("Tech earnings beat estimates", 0.05),
            ("Inflation fears rise", -0.03),
            ("AI breakthrough announced", 0.06),
            ("Regulatory crackdown", -0.05),
            ("Strong jobs report", 0.02),
            ("Geopolitical tensions", -0.04),
        ]
        self.order_book = {'bids': [], 'asks': []}

    def inject_shock(self, shock_pct: float):
        self.trend += shock_pct
        self.news_text = f"{'PUMP' if shock_pct > 0 else 'CRASH'}: {shock_pct*100:.0f}%"
        self.news_timer = 4.0

    def generate_order_book(self):
        bids = []
        asks = []
        spread = self.price * 0.001
        for i in range(1, 11):
            bid_price = self.price - spread * i - random.uniform(0, spread * 0.5)
            ask_price = self.price + spread * i + random.uniform(0, spread * 0.5)
            # Volume walls
            bid_vol = random.randint(100, 2000) + (8000 if i in [3, 7] else 0)
            ask_vol = random.randint(100, 2000) + (8000 if i in [4, 8] else 0)
            bids.append({'price': bid_price, 'volume': bid_vol})
            asks.append({'price': ask_price, 'volume': ask_vol})
        self.order_book = {'bids': bids, 'asks': asks}

    def next(self) -> float:
        if random.random() < 0.01:
            text, shock = random.choice(self.news_events)
            self.trend += shock
            self.news_text = text
            self.news_timer = 3.0
            
        shock = random.gauss(0, self.volatility)
        reversion = (self.target - self.price) * 0.003
        new_price = self.price * (1 + self.trend + shock) + reversion * 0.5
        self.trend *= 0.95
        new_price = max(10.0, min(400.0, new_price))
        
        o = self.price
        c = new_price
        wick_range = abs(c - o) + self.price * self.volatility * random.uniform(0.5, 1.5)
        h = max(o, c) + random.uniform(0, wick_range * 0.5)
        l = min(o, c) - random.uniform(0, wick_range * 0.5)
        v = int(5000 + abs(c - o) * 100000 + random.uniform(0, 5000))
        
        self.candles.append({'o': o, 'h': h, 'l': l, 'c': c, 'v': v})
        self.history.append(new_price)
        self.price = new_price
        
        self.ticks += 1
        if len(self.history) > 3000:
            self.history = self.history[-3000:]
            self.candles = self.candles[-3000:]
            
        self.generate_order_book()
        return new_price

    def reset(self) -> None:
        self.price = self.start_price
        self.history = [self.start_price]
        self.candles = [{'o': self.start_price, 'h': self.start_price, 'l': self.start_price, 'c': self.start_price, 'v': 5000}]
        self.trend = 0.0003
        self.volatility = 0.008
        self.ticks = 0
        self.news_text = ""
        self.news_timer = 0
        self.generate_order_book()

# ── Price chart ─────────────────────────────────────────────────────────
class PriceChartWidget(QWidget):
    def __init__(self, sim: StockSimulator, traders: List[AITrader]):
        super().__init__()
        self.sim = sim
        self.traders = traders
        self.title = "MARKET  ·  TSLA"
        self.window_size = 180
        self.particles: List[dict] = []
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.setMouseTracking(True)
        self.hover_x = -1
        self.hover_y = -1
        self.selected_ai: Optional[str] = None

    def add_trade_effect(self, sx: float, sy: float, color: QColor):
        for _ in range(12):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(40, 120)
            self.particles.append({
                'x': sx, 'y': sy,
                'vx': math.cos(ang) * spd, 'vy': math.sin(ang) * spd,
                'life': 1.0, 'color': color,
                'size': random.uniform(2.5, 5.0),
            })

    def update_anim(self, dt: float):
        for p in self.particles:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['vy'] += 220 * dt
            p['life'] -= dt * 1.8
        self.particles = [p for p in self.particles if p['life'] > 0]

    def mouseMoveEvent(self, event):
        self.hover_x = event.position().x()
        self.hover_y = event.position().y()
        self.update()

    def leaveEvent(self, event):
        self.hover_x = -1
        self.hover_y = -1
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.window_size = max(60, self.window_size - 20)
        else:
            self.window_size = min(400, self.window_size + 20)
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor("#131a2a"))
        grad.setColorAt(1, QColor("#070b16"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(COLORS['border'], 1))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)

        painter.setPen(QPen(COLORS['text_dim']))
        f = QFont("Arial", 9, QFont.Bold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        painter.setFont(f)
        painter.drawText(QRectF(14, 8, 300, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        
        painter.setBrush(QBrush(COLORS['red']))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(140, 18), 3, 3)
        
        painter.setPen(QPen(COLORS['text_dimmer']))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(QRectF(w - 100, 8, 80, 20), Qt.AlignRight | Qt.AlignVCenter, f"Zoom: {self.window_size}")

        m_left, m_right, m_top, m_bot = 64, 100, 38, 100
        cw, ch = w - m_left - m_right, h - m_top - m_bot
        if cw < 20 or ch < 20: return
        if len(self.sim.candles) < 2:
            painter.setPen(QPen(COLORS['text_dim']))
            painter.drawText(rect, Qt.AlignCenter, "Initializing market data…")
            return

        vis_candles = self.sim.candles[-self.window_size:]
        closes = [c['c'] for c in vis_candles]
        p_min = min(c['l'] for c in vis_candles) * 0.995
        p_max = max(c['h'] for c in vis_candles) * 1.005
        p_rng = max(0.01, p_max - p_min)

        def xs(i): return m_left + (i / max(1, len(vis_candles) - 1)) * cw
        def ys(p): return m_top + (1 - (p - p_min) / p_rng) * ch

        for i in range(5):
            y = m_top + (i / 4) * ch
            painter.setPen(QPen(COLORS['grid'], 1, Qt.DotLine))
            painter.drawLine(QPointF(m_left, y), QPointF(m_left + cw, y))
            price = p_max - (i / 4) * p_rng
            painter.setPen(QPen(COLORS['text_dimmer']))
            painter.setFont(QFont("Consolas", 9))
            painter.drawText(QRectF(4, y - 9, m_left - 8, 18), Qt.AlignRight | Qt.AlignVCenter, f"${price:6.2f}")

        # Closed Trade Lines (Entry to Exit)
        for trader in self.traders:
            alpha = 255 if self.selected_ai is None or self.selected_ai == trader.name else 40
            for ct in trader.closed_trades[-15:]:
                if ct.exit_tick < len(self.sim.history) - self.window_size: continue
                start_idx = max(0, ct.entry_tick - (len(self.sim.history) - self.window_size))
                end_idx = max(0, ct.exit_tick - (len(self.sim.history) - self.window_size))
                
                x1 = xs(start_idx)
                y1 = ys(ct.entry_price)
                x2 = xs(end_idx)
                y2 = ys(ct.exit_price)
                
                line_color = COLORS['green'] if ct.profit >= 0 else COLORS['red']
                painter.setPen(QPen(QColor(line_color.red(), line_color.green(), line_color.blue(), alpha // 2), 1, Qt.DotLine))
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # AI Average Entry Lines
        for trader in self.traders:
            if trader.shares != 0 and trader.avg_entry > 0:
                alpha = 255 if self.selected_ai is None or self.selected_ai == trader.name else 40
                y_entry = ys(trader.avg_entry)
                c = trader.color
                painter.setPen(QPen(QColor(c.red(), c.green(), c.blue(), alpha // 2), 1, Qt.DashLine))
                painter.drawLine(QPointF(m_left, y_entry), QPointF(m_left + cw, y_entry))
                painter.setPen(QPen(QColor(c.red(), c.green(), c.blue(), alpha)))
                painter.setFont(QFont("Consolas", 8, QFont.Bold))
                painter.drawText(QRectF(m_left + cw - 70, y_entry - 14, 65, 12), Qt.AlignRight, f"{trader.name[:3]}: {trader.avg_entry:.2f}")

        # Bollinger Bands
        if len(closes) >= 40:
            upper_band, lower_band = [], []
            for i in range(len(closes)):
                if i < 40:
                    upper_band.append(closes[i])
                    lower_band.append(closes[i])
                else:
                    w_slice = closes[i-40:i]
                    mean = sum(w_slice) / 40
                    var = sum((p - mean)**2 for p in w_slice) / 40
                    std = math.sqrt(var)
                    upper_band.append(mean + 2*std)
                    lower_band.append(mean - 2*std)
            
            band_path = QPainterPath()
            band_path.moveTo(xs(0), ys(upper_band[0]))
            for i in range(1, len(closes)): band_path.lineTo(xs(i), ys(upper_band[i]))
            for i in range(len(closes)-1, -1, -1): band_path.lineTo(xs(i), ys(lower_band[i]))
            band_path.closeSubpath()
            painter.setBrush(QBrush(QColor(0, 212, 255, 15)))
            painter.setPen(Qt.NoPen)
            painter.drawPath(band_path)

        max_vol = max(c['v'] for c in vis_candles)
        vol_y0 = h - 10
        vol_h = 40
        vol_top = vol_y0 - vol_h
        
        rsi_h = 40
        rsi_bot = vol_top - 10
        rsi_top = rsi_bot - rsi_h
        
        for i, c in enumerate(vis_candles):
            x = xs(i)
            is_green = c['c'] >= c['o']
            color = COLORS['green'] if is_green else COLORS['red']
            
            painter.setPen(QPen(color, 1.5))
            painter.drawLine(QPointF(x, ys(c['h'])), QPointF(x, ys(c['l'])))
            
            body_h = abs(ys(c['o']) - ys(c['c']))
            body_y = min(ys(c['o']), ys(c['c']))
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(x - 3, body_y, 6, max(1, body_h)))
            
            vh = (c['v'] / max_vol) * vol_h
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 60)))
            painter.drawRect(QRectF(x - 3, vol_y0 - vh, 6, vh))

        # RSI Panel
        painter.setPen(QPen(COLORS['grid'], 1, Qt.SolidLine))
        painter.drawLine(QPointF(m_left, rsi_top), QPointF(m_left + cw, rsi_top))
        painter.drawLine(QPointF(m_left, rsi_bot), QPointF(m_left + cw, rsi_bot))
        painter.setPen(QPen(COLORS['text_dimmer']))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(QRectF(4, rsi_top - 5, m_left - 8, 14), Qt.AlignRight, "RSI")
        y_70 = rsi_top + (1 - 70/100) * rsi_h
        y_30 = rsi_top + (1 - 30/100) * rsi_h
        painter.setPen(QPen(COLORS['red'], 1, Qt.DotLine))
        painter.drawLine(QPointF(m_left, y_70), QPointF(m_left + cw, y_70))
        painter.setPen(QPen(COLORS['green'], 1, DotLine if hasattr(Qt, 'DotLine') else Qt.DashLine))
        painter.drawLine(QPointF(m_left, y_30), QPointF(m_left + cw, y_30))
        
        _, rsi_hist = calculate_rsi(self.sim.history, 14)
        if rsi_hist:
            vis_rsi_len = min(len(vis_candles), len(rsi_hist))
            vis_rsi = rsi_hist[-vis_rsi_len:]
            offset = len(vis_candles) - len(vis_rsi)
            if len(vis_rsi) > 1:
                path = QPainterPath()
                for i, r in enumerate(vis_rsi):
                    vis_idx = i + offset
                    x = m_left + (vis_idx / max(1, len(vis_candles) - 1)) * cw
                    y = rsi_top + (1 - r/100) * rsi_h
                    if vis_idx == offset: path.moveTo(x, y)
                    else: path.lineTo(x, y)
                painter.setPen(QPen(COLORS['purple'], 1.5))
                painter.drawPath(path)

        # Trade markers
        start_idx = max(0, len(self.sim.history) - self.window_size)
        for trader in self.traders:
            alpha = 255 if self.selected_ai is None or self.selected_ai == trader.name else 40
            for trade in trader.trades:
                if trade.tick < start_idx or trade.tick >= len(self.sim.history): continue
                idx = trade.tick - start_idx
                if idx < 0 or idx >= len(vis_candles): continue
                tx = xs(idx)
                ty = ys(trade.price)
                if trade.action in ('BUY', 'COVER'):
                    ty_off = ty + 13
                    tri = QPolygonF([QPointF(tx, ty_off - 4), QPointF(tx - 4, ty_off + 3), QPointF(tx + 4, ty_off + 3)])
                else:
                    ty_off = ty - 13
                    tri = QPolygonF([QPointF(tx, ty_off + 4), QPointF(tx - 4, ty_off - 3), QPointF(tx + 4, ty_off - 3)])
                c = trader.color
                glow = QRadialGradient(tx, ty_off, 8)
                glow.setColorAt(0, QColor(c.red(), c.green(), c.blue(), 110 if alpha == 255 else 20))
                glow.setColorAt(1, QColor(c.red(), c.green(), c.blue(), 0))
                painter.setBrush(QBrush(glow))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(tx, ty_off), 8, 8)
                painter.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), alpha)))
                painter.setPen(QPen(QColor(255, 255, 255, 90 if alpha == 255 else 20), 0.6))
                painter.drawPolygon(tri)

        # Hover Crosshair & Tooltip
        if self.hover_x > m_left and self.hover_x < m_left + cw and self.hover_y > m_top and self.hover_y < vol_y0:
            idx = int((self.hover_x - m_left) / cw * (len(vis_candles) - 1))
            idx = max(0, min(len(vis_candles) - 1, idx))
            c = vis_candles[idx]
            
            painter.setPen(QPen(QColor(255, 255, 255, 60), 1, Qt.DashLine))
            painter.drawLine(QPointF(xs(idx), m_top), QPointF(xs(idx), vol_y0))
            if self.hover_y < rsi_top:
                painter.drawLine(QPointF(m_left, self.hover_y), QPointF(m_left + cw, self.hover_y))
            
            tip_w, tip_h = 140, 95
            tip_x = xs(idx) + 15
            if tip_x + tip_w > w - 10: tip_x = xs(idx) - tip_w - 15
            tip_y = max(m_top, min(rsi_bot - tip_h - 5, self.hover_y - tip_h - 10))
            
            painter.setBrush(QBrush(QColor(13, 19, 32, 230)))
            painter.setPen(QPen(COLORS['border'], 1))
            painter.drawRoundedRect(QRectF(tip_x, tip_y, tip_w, tip_h), 4, 4)
            painter.setPen(QPen(COLORS['text']))
            painter.setFont(QFont("Consolas", 9, QFont.Bold))
            painter.drawText(QRectF(tip_x + 8, tip_y + 5, tip_w - 16, 16), Qt.AlignLeft, f"Tick {start_idx + idx}")
            o, h_, l, cl, v = c['o'], c['h'], c['l'], c['c'], c['v']
            painter.setFont(QFont("Consolas", 9))
            painter.setPen(QPen(COLORS['text_dim']))
            painter.drawText(QRectF(tip_x + 8, tip_y + 22, 30, 14), Qt.AlignLeft, "O:")
            painter.drawText(QRectF(tip_x + 8, tip_y + 38, 30, 14), Qt.AlignLeft, "H:")
            painter.drawText(QRectF(tip_x + 8, tip_y + 54, 30, 14), Qt.AlignLeft, "L:")
            painter.drawText(QRectF(tip_x + 8, tip_y + 70, 30, 14), Qt.AlignLeft, "V:")
            painter.setPen(QPen(COLORS['green'] if cl >= o else COLORS['red']))
            painter.drawText(QRectF(tip_x + 35, tip_y + 22, 60, 14), Qt.AlignLeft, f"${o:.2f}")
            painter.drawText(QRectF(tip_x + 35, tip_y + 38, 60, 14), Qt.AlignLeft, f"${h_:.2f}")
            painter.drawText(QRectF(tip_x + 35, tip_y + 54, 60, 14), Qt.AlignLeft, f"${l:.2f}")
            painter.setPen(QPen(COLORS['text']))
            painter.drawText(QRectF(tip_x + 35, tip_y + 70, 60, 14), Qt.AlignLeft, f"{v:,}")
            painter.setPen(QPen(COLORS['text_dim']))
            painter.drawText(QRectF(tip_x + 85, tip_y + 22, 50, 14), Qt.AlignLeft, f"C:")
            painter.setPen(QPen(COLORS['green'] if cl >= o else COLORS['red']))
            painter.drawText(QRectF(tip_x + 100, tip_y + 22, 40, 14), Qt.AlignLeft, f"${cl:.2f}")

        cur_x, cur_y = xs(len(vis_candles) - 1), ys(vis_candles[-1]['c'])
        painter.setPen(QPen(QColor(0, 212, 255, 90), 1, Qt.DashLine))
        painter.drawLine(QPointF(m_left, cur_y), QPointF(m_left + cw, cur_y))
        glow = QRadialGradient(cur_x, cur_y, 18)
        glow.setColorAt(0, QColor(0, 212, 255, 200))
        glow.setColorAt(1, QColor(0, 212, 255, 0))
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cur_x, cur_y), 18, 18)
        painter.setBrush(QBrush(COLORS['accent']))
        painter.drawEllipse(QPointF(cur_x, cur_y), 4.5, 4.5)

        tag = QRectF(m_left + cw + 4, cur_y - 11, m_right - 6, 22)
        painter.setBrush(QBrush(QColor(0, 212, 255, 30)))
        painter.setPen(QPen(COLORS['accent'], 1))
        painter.drawRoundedRect(tag, 4, 4)
        painter.setPen(QPen(COLORS['accent']))
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        painter.drawText(tag, Qt.AlignCenter, f"${vis_candles[-1]['c']:.2f}")

        if self.sim.news_timer > 0:
            alpha = int(255 * min(1.0, self.sim.news_timer))
            news_rect = QRectF(w//2 - 150, 10, 300, 24)
            painter.setBrush(QBrush(QColor(255, 71, 87, alpha)))
            painter.setPen(QPen(QColor(255, 255, 255, alpha), 1))
            painter.drawRoundedRect(news_rect, 4, 4)
            painter.setPen(QPen(QColor(255, 255, 255, alpha)))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(news_rect, Qt.AlignCenter, f"Breaking: {self.sim.news_text}")

        for p in self.particles:
            a = int(255 * max(0, p['life']))
            c = p['color']
            painter.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), a)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(p['x'], p['y']), p['size'], p['size'])


# ── Portfolio chart ────────────────────────────────────────────────────
class PortfolioChartWidget(QWidget):
    def __init__(self, sim: StockSimulator, traders: List[AITrader]):
        super().__init__()
        self.sim = sim
        self.traders = traders
        self.title = "AI PORTFOLIO  ·  ABSOLUTE EQUITY ($)"
        self.window_size = 180
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.selected_ai: Optional[str] = None

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor("#131a2a"))
        grad.setColorAt(1, QColor("#070b16"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(COLORS['border'], 1))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)

        painter.setPen(QPen(COLORS['text_dim']))
        f = QFont("Arial", 9, QFont.Bold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        painter.setFont(f)
        painter.drawText(QRectF(14, 8, 300, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        m_left, m_right, m_top, m_bot = 70, 14, 38, 22
        cw, ch = w - m_left - m_right, h - m_top - m_bot
        if cw < 20 or ch < 20: return

        max_len = max((len(t.history) for t in self.traders), default=0)
        if max_len < 2:
            painter.setPen(QPen(COLORS['text_dim']))
            painter.drawText(rect, Qt.AlignCenter, "Waiting for trade data…")
            return

        vlen = min(self.window_size, max_len)
        all_vals = []
        for t in self.traders: all_vals.extend(t.history[-vlen:])
        if not all_vals: return
        
        max_abs = max(abs(v) for v in all_vals)
        y_max = max_abs * 1.15 if max_abs > 0 else 10000
        y_min = 0
        y_rng = y_max - y_min

        def xs(i): return m_left + (i / max(1, vlen - 1)) * cw
        def ys(v): return m_top + (1 - (v - y_min) / y_rng) * ch

        zero_y = ys(self.traders[0].starting_cash)
        painter.setPen(QPen(QColor(139, 149, 168, 110), 1, Qt.DashLine))
        painter.drawLine(QPointF(m_left, zero_y), QPointF(m_left + cw, zero_y))
        painter.setPen(QPen(COLORS['text_dimmer']))
        painter.setFont(QFont("Consolas", 9))
        painter.drawText(QRectF(4, zero_y - 9, m_left - 8, 18), Qt.AlignRight | Qt.AlignVCenter, f"${self.traders[0].starting_cash/1000:.0f}k")

        for i in range(5):
            y = m_top + (i / 4) * ch
            painter.setPen(QPen(COLORS['grid'], 1, Qt.DotLine))
            painter.drawLine(QPointF(m_left, y), QPointF(m_left + cw, y))
            val = y_max - (i / 4) * y_rng
            painter.setPen(QPen(COLORS['text_dimmer']))
            painter.setFont(QFont("Consolas", 9))
            painter.drawText(QRectF(4, y - 9, m_left - 8, 18), Qt.AlignRight | Qt.AlignVCenter, f"${val/1000:.1f}k")

        for trader in self.traders:
            hist = trader.history[-vlen:]
            if len(hist) < 2: continue
            path = QPainterPath()
            path.moveTo(xs(0), ys(hist[0]))
            for i in range(1, len(hist)): path.lineTo(xs(i), ys(hist[i]))
            c = trader.color
            alpha = 255 if self.selected_ai is None or self.selected_ai == trader.name else 40
            
            painter.setPen(QPen(QColor(c.red(), c.green(), c.blue(), alpha // 4), 4))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(c.red(), c.green(), c.blue(), alpha), 1.8))
            painter.drawPath(path)
            cx, cy = xs(len(hist) - 1), ys(hist[-1])
            painter.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), alpha)))
            painter.setPen(QPen(QColor(255, 255, 255, alpha // 2), 0.8))
            painter.drawEllipse(QPointF(cx, cy), 3.5, 3.5)


# ── Sentiment Gauge ───────────────────────────────────────────────────
class SentimentGauge(QWidget):
    def __init__(self, sim: StockSimulator, traders: List[AITrader]):
        super().__init__()
        self.sim = sim
        self.traders = traders
        self.setMinimumHeight(100)
        self.setMaximumHeight(120)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor("#131a2a"))
        grad.setColorAt(1, QColor("#070b16"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(COLORS['border'], 1))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)

        painter.setPen(QPen(COLORS['text_dim']))
        f = QFont("Arial", 9, QFont.Bold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        painter.setFont(f)
        painter.drawText(QRectF(14, 8, 200, 20), Qt.AlignLeft | Qt.AlignVCenter, "AI MARKET SENTIMENT")

        net_shares = sum(t.shares for t in self.traders)
        max_pos = sum(t.starting_cash / self.sim.price for t in self.traders)
        ratio = max(-1.0, min(1.0, net_shares / max_pos)) if max_pos > 0 else 0

        cx, cy = w / 2, h - 15
        radius = min(w / 2 - 30, h - 35)
        
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(COLORS['red'], 8, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(QRectF(cx - radius, cy - radius, radius * 2, radius * 2), 0 * 16, 90 * 16)
        painter.setPen(QPen(COLORS['green'], 8, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(QRectF(cx - radius, cy - radius, radius * 2, radius * 2), 90 * 16, 90 * 16)

        angle = 90 + (ratio * 90)
        rad = math.radians(angle)
        nx = cx + radius * math.cos(rad)
        ny = cy - radius * math.sin(rad)
        
        painter.setPen(QPen(COLORS['text'], 2))
        painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))
        painter.setBrush(QBrush(COLORS['text']))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cx, cy), 4, 4)

        painter.setPen(QPen(COLORS['green']))
        painter.setFont(QFont("Consolas", 9, QFont.Bold))
        painter.drawText(QRectF(cx + radius - 30, cy - 10, 60, 20), Qt.AlignCenter, "LONG")
        painter.setPen(QPen(COLORS['red']))
        painter.drawText(QRectF(cx - radius - 30, cy - 10, 60, 20), Qt.AlignCenter, "SHORT")


# ── Order Book Widget ─────────────────────────────────────────────────
class OrderBookWidget(QWidget):
    def __init__(self, sim: StockSimulator):
        super().__init__()
        self.sim = sim
        self.title = "ORDER BOOK DEPTH"
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor("#131a2a"))
        grad.setColorAt(1, QColor("#070b16"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(COLORS['border'], 1))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)

        painter.setPen(QPen(COLORS['text_dim']))
        f = QFont("Arial", 9, QFont.Bold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        painter.setFont(f)
        painter.drawText(QRectF(14, 8, 200, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        if not self.sim.order_book['bids']: return

        m_left, m_right, m_top, m_bot = 60, 14, 38, 12
        cw, ch = w - m_left - m_right, h - m_top - m_bot
        if cw < 20 or ch < 20: return

        max_vol = max(max(b['volume'] for b in self.sim.order_book['bids']), 
                      max(a['volume'] for a in self.sim.order_book['asks']))
        
        row_h = ch / 20
        y = m_top
        
        # Asks (Top half, reversed)
        for ask in reversed(self.sim.order_book['asks']):
            vol_w = (ask['volume'] / max_vol) * (cw / 2)
            painter.setBrush(QBrush(QColor(255, 71, 87, 60)))
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(m_left + cw/2 - vol_w, y, vol_w, row_h - 1))
            
            painter.setPen(QPen(COLORS['red']))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(QRectF(m_left, y, cw/2 - 5, row_h), Qt.AlignRight | Qt.AlignVCenter, f"{ask['price']:.2f}")
            painter.setPen(QPen(COLORS['text_dim']))
            painter.drawText(QRectF(m_left + cw/2 + 5, y, cw/2 - 5, row_h), Qt.AlignLeft | Qt.AlignVCenter, f"{ask['volume']:,}")
            y += row_h

        # Spread
        spread = self.sim.order_book['asks'][0]['price'] - self.sim.order_book['bids'][0]['price']
        spread_rect = QRectF(m_left, y, cw, row_h)
        painter.setBrush(QBrush(QColor(0, 212, 255, 20)))
        painter.setPen(QPen(COLORS['accent'], 1))
        painter.drawRect(spread_rect)
        painter.setPen(QPen(COLORS['accent']))
        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.drawText(spread_rect, Qt.AlignCenter, f"SPREAD: ${spread:.3f}")
        y += row_h

        # Bids (Bottom half)
        for bid in self.sim.order_book['bids']:
            vol_w = (bid['volume'] / max_vol) * (cw / 2)
            painter.setBrush(QBrush(QColor(0, 255, 136, 60)))
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(m_left + cw/2 - vol_w, y, vol_w, row_h - 1))
            
            painter.setPen(QPen(COLORS['green']))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(QRectF(m_left, y, cw/2 - 5, row_h), Qt.AlignRight | Qt.AlignVCenter, f"{bid['price']:.2f}")
            painter.setPen(QPen(COLORS['text_dim']))
            painter.drawText(QRectF(m_left + cw/2 + 5, y, cw/2 - 5, row_h), Qt.AlignLeft | Qt.AlignVCenter, f"{bid['volume']:,}")
            y += row_h


# ── Leaderboard ────────────────────────────────────────────────────────
class LeaderboardWidget(QWidget):
    def __init__(self, sim: StockSimulator, traders: List[AITrader]):
        super().__init__()
        self.sim = sim
        self.traders = traders
        self.title = "LEADERBOARD & AI STATE"
        self.setMinimumWidth(340)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.selected_ai: Optional[str] = None
        self.row_rects: List[Tuple[str, QRectF]] = []
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        x, y = event.position().x(), event.position().y()
        for name, rect in self.row_rects:
            if rect.contains(x, y):
                self.selected_ai = None if self.selected_ai == name else name
                self.parent().parent().parent().selected_ai_changed(self.selected_ai)
                self.update()
                break

    def paintEvent(self, _):
        self.row_rects.clear()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor("#131a2a"))
        grad.setColorAt(1, QColor("#070b16"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(COLORS['border'], 1))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)

        painter.setPen(QPen(COLORS['text_dim']))
        f = QFont("Arial", 10, QFont.Bold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.8)
        painter.setFont(f)
        painter.drawText(QRectF(14, 8, 300, 22), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        price = self.sim.price
        ranked = sorted(self.traders, key=lambda t: -t.pct_at(price))

        avail = h - 50
        row_h = max(60, min(90, avail // len(ranked)))
        y_start = 38
        
        for i, trader in enumerate(ranked):
            y = y_start + i * row_h
            if y + row_h > h - 6: break
            row_rect = QRectF(8, y, w - 16, row_h - 6)
            self.row_rects.append((trader.name, row_rect))

            is_selected = self.selected_ai == trader.name
            base_color = QColor(18, 26, 43) if i > 0 else QColor(255, 217, 61, 18)
            if is_selected:
                base_color = QColor(trader.color.red(), trader.color.green(), trader.color.blue(), 40)
            
            painter.setBrush(QBrush(base_color))
            border_col = QColor(255, 217, 61, 130) if i == 0 else COLORS['border']
            if is_selected: border_col = trader.color
            painter.setPen(QPen(border_col, 1 if not is_selected else 2))
            painter.drawRoundedRect(row_rect, 6, 6)

            if trader.pulse > 0:
                a = int(70 * trader.pulse)
                painter.setBrush(QBrush(QColor(trader.color.red(), trader.color.green(), trader.color.blue(), a)))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(row_rect, 6, 6)

            painter.setPen(QPen(COLORS['yellow'] if i == 0 else COLORS['text_dimmer']))
            painter.setFont(QFont("Consolas", 12, QFont.Bold))
            painter.drawText(QRectF(row_rect.left() + 8, row_rect.top() + 2, 30, 24), Qt.AlignLeft | Qt.AlignVCenter, f"#{i+1}")

            if i == 0:
                painter.setPen(QPen(COLORS['yellow']))
                painter.setFont(QFont("Arial", 11, QFont.Bold))
                painter.drawText(QRectF(row_rect.left() + 32, row_rect.top() + 2, 30, 24), Qt.AlignLeft | Qt.AlignVCenter, "👑")

            cx, cy = row_rect.left() + 46, row_rect.top() + 16
            if i == 0:
                glow = QRadialGradient(cx, cy, 14)
                glow.setColorAt(0, QColor(255, 217, 61, 200))
                glow.setColorAt(1, QColor(255, 217, 61, 0))
                painter.setBrush(QBrush(glow))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(cx, cy), 14, 14)
            painter.setBrush(QBrush(trader.color))
            painter.drawEllipse(QPointF(cx, cy), 5, 5)

            painter.setPen(QPen(COLORS['text']))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(QRectF(row_rect.left() + 60, row_rect.top() + 2, 120, 18), Qt.AlignLeft | Qt.AlignVCenter, trader.name)

            pct = trader.pct_at(price)
            ch_color = COLORS['green'] if pct >= 0 else COLORS['red']
            painter.setPen(QPen(ch_color))
            painter.setFont(QFont("Consolas", 11, QFont.Bold))
            painter.drawText(QRectF(row_rect.right() - 100, row_rect.top() + 2, 92, 22), Qt.AlignRight | Qt.AlignVCenter, f"{pct:+6.2f}%")

            painter.setPen(QPen(trader.color))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(QRectF(row_rect.left() + 60, row_rect.top() + 20, 180, 12), Qt.AlignLeft | Qt.AlignVCenter, f"› {trader.state_text}")

            val = trader.value_at(price)
            wr = (trader.wins / (trader.wins + trader.losses) * 100) if (trader.wins + trader.losses) > 0 else 0
            
            # Sharpe Ratio
            sharpe = 0.0
            if len(trader.history) > 10:
                returns = [(trader.history[i] - trader.history[i-1]) / trader.history[i-1] for i in range(1, len(trader.history))]
                mean_r = sum(returns) / len(returns)
                std_r = (sum((r - mean_r)**2 for r in returns) / len(returns))**0.5
                sharpe = (mean_r / std_r) * (252**0.5) if std_r > 0 else 0.0
                
            pf = trader.gross_profit / trader.gross_loss if trader.gross_loss > 0 else 9.9
            stats_txt = f"Val ${val:,.0f} | Sharpe {sharpe:.2f} | PF {pf:.1f}"
            painter.setPen(QPen(COLORS['text_dimmer']))
            painter.setFont(QFont("Consolas", 7))
            painter.drawText(QRectF(row_rect.left() + 60, row_rect.top() + 36, 250, 11), Qt.AlignLeft | Qt.AlignVCenter, stats_txt)

            # Exposure Bar
            max_exp = trader.starting_cash * 2.0
            cur_exp = trader.shares * price
            exp_ratio = max(-1.0, min(1.0, cur_exp / max_exp))

            bar_x = row_rect.left() + 60
            bar_y = row_rect.top() + 48
            bar_w = 120
            bar_h = 5

            painter.setBrush(QBrush(QColor(31, 41, 66, 150)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2, 2)

            center_x = bar_x + bar_w / 2
            painter.setPen(QPen(COLORS['text_dimmer'], 1))
            painter.drawLine(QPointF(center_x, bar_y - 1), QPointF(center_x, bar_y + bar_h + 1))

            fill_w = abs(exp_ratio * (bar_w / 2))
            if exp_ratio > 0:
                painter.setBrush(QBrush(COLORS['green']))
                painter.drawRect(QRectF(center_x, bar_y, fill_w, bar_h))
            else:
                painter.setBrush(QBrush(COLORS['red']))
                painter.drawRect(QRectF(center_x - fill_w, bar_y, fill_w, bar_h))

            # Rank History Sparkline
            if len(trader.rank_history) > 2:
                spark_x = bar_x + bar_w + 10
                spark_y = bar_y - 2
                spark_w = 40
                spark_h = 9
                path = QPainterPath()
                max_rank = len(self.traders)
                for idx, rank in enumerate(trader.rank_history):
                    px = spark_x + (idx / (len(trader.rank_history)-1)) * spark_w
                    py = spark_y + (rank / max_rank) * spark_h
                    if idx == 0: path.moveTo(px, py)
                    else: path.lineTo(px, py)
                painter.setPen(QPen(trader.color, 1.2))
                painter.drawPath(path)

            act_col = COLORS['green'] if "BUY" in trader.last_action or "COVER" in trader.last_action else COLORS['red'] if "SELL" in trader.last_action or "SHORT" in trader.last_action else COLORS['text_dimmer']
            act_bg = QColor(act_col.red(), act_col.green(), act_col.blue(), 30)
            act_rect = QRectF(row_rect.right() - 100, row_rect.top() + 28, 92, 16)
            painter.setBrush(QBrush(act_bg))
            painter.setPen(QPen(act_col, 1))
            painter.drawRoundedRect(act_rect, 3, 3)
            painter.setPen(QPen(act_col))
            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.drawText(act_rect, Qt.AlignCenter, trader.last_action)


# ── Trade feed ─────────────────────────────────────────────────────────
class TradeFeedWidget(QWidget):
    def __init__(self, traders: List[AITrader]):
        super().__init__()
        self.traders = traders
        self.recent: deque = deque(maxlen=14)
        self.title = "EXECUTED TRADES"
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    def add_trade(self, idx: int, action: str, qty: int, price: float, tick: int):
        self.recent.appendleft({'idx': idx, 'action': action, 'qty': qty, 'price': price, 'tick': tick})

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor("#131a2a"))
        grad.setColorAt(1, QColor("#070b16"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(COLORS['border'], 1))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)

        painter.setPen(QPen(COLORS['text_dim']))
        f = QFont("Arial", 10, QFont.Bold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.8)
        painter.setFont(f)
        painter.drawText(QRectF(14, 8, 200, 22), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        y_hdr = 36
        painter.setPen(QPen(COLORS['text_dimmer']))
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.drawText(QRectF(30, y_hdr, 110, 16), Qt.AlignLeft, "TRADER")
        painter.drawText(QRectF(140, y_hdr, 60, 16), Qt.AlignLeft, "ACTION")
        painter.drawText(QRectF(210, y_hdr, 80, 16), Qt.AlignLeft, "QTY")
        painter.drawText(QRectF(280, y_hdr, 80, 16), Qt.AlignLeft, "PRICE")

        y = y_hdr + 18
        row_h = 20
        for t in self.recent:
            if y + row_h > h - 8: break
            trader = self.traders[t['idx']]
            painter.setBrush(QBrush(trader.color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(20, y + 10), 3.5, 3.5)
            painter.setPen(QPen(COLORS['text']))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(QRectF(30, y, 110, row_h), Qt.AlignLeft | Qt.AlignVCenter, trader.name)

            act_col = COLORS['green'] if t['action'] in ('BUY', 'COVER') else COLORS['red']
            act_bg = QColor(act_col.red(), act_col.green(), act_col.blue(), 30)
            act_rect = QRectF(140, y + 2, 50, 16)
            painter.setBrush(QBrush(act_bg))
            painter.setPen(QPen(act_col, 1))
            painter.drawRoundedRect(act_rect, 3, 3)
            painter.setPen(QPen(act_col))
            painter.setFont(QFont("Consolas", 9, QFont.Bold))
            painter.drawText(act_rect, Qt.AlignCenter, t['action'])

            painter.setPen(QPen(COLORS['text_dim']))
            painter.setFont(QFont("Consolas", 10))
            painter.drawText(QRectF(210, y, 80, row_h), Qt.AlignLeft | Qt.AlignVCenter, f"{t['qty']:,}")
            painter.setPen(QPen(COLORS['text']))
            painter.drawText(QRectF(280, y, 80, row_h), Qt.AlignLeft | Qt.AlignVCenter, f"${t['price']:.2f}")
            y += row_h


# ── Stats bar ─────────────────────────────────────────────────────────
class StatsBar(QWidget):
    def __init__(self, sim: StockSimulator, traders: List[AITrader]):
        super().__init__()
        self.sim = sim
        self.traders = traders
        self.setMinimumHeight(54)
        self.setMaximumHeight(64)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor("#0d1320"))
        grad.setColorAt(1, QColor("#050810"))
        painter.fillRect(rect, grad)
        painter.setPen(QPen(COLORS['border'], 1))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)

        price = self.sim.price
        
        regime = "RANGING"
        regime_col = COLORS['yellow']
        if len(self.sim.history) >= 50:
            ma_now = sum(self.sim.history[-10:]) / 10
            ma_old = sum(self.sim.history[-50:-40]) / 10
            slope = (ma_now - ma_old) / ma_old * 100
            if slope > 0.2:
                regime = "BULLISH"
                regime_col = COLORS['green']
            elif slope < -0.2:
                regime = "BEARISH"
                regime_col = COLORS['red']

        total_vol = sum(c['v'] for c in self.sim.candles[-100:]) if len(self.sim.candles) > 0 else 0

        stats = [
            ("TICK",       f"{self.sim.ticks:,}",       COLORS['text']),
            ("PRICE",      f"${price:.2f}",             COLORS['accent']),
            ("REGIME",     regime,                      regime_col),
            ("VOL (100)",  f"{total_vol/1_000_000:.1f}M", COLORS['text']),
            ("TRADES",     f"{sum(len(t.trades) for t in self.traders):,}", COLORS['text']),
        ]
        if self.traders:
            pcts = [t.pct_at(price) for t in self.traders]
            stats.append(("LEADER",  f"{max(pcts):+.2f}%", COLORS['green']))
            stats.append(("LAGGARD", f"{min(pcts):+.2f}%", COLORS['red']))

        col_w = w / len(stats)
        for i, (label, value, val_col) in enumerate(stats):
            x = i * col_w
            painter.setPen(QPen(COLORS['text_dimmer']))
            f = QFont("Arial", 8, QFont.Bold)
            f.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)
            painter.setFont(f)
            painter.drawText(QRectF(x + 14, 9, col_w - 14, 14), Qt.AlignLeft, label)
            painter.setPen(QPen(val_col))
            painter.setFont(QFont("Consolas", 13, QFont.Bold))
            painter.drawText(QRectF(x + 14, 26, col_w - 14, 22), Qt.AlignLeft, value)
            if i > 0:
                painter.setPen(QPen(QColor(37, 48, 74, 100)))
                painter.drawLine(QPointF(x, 12), QPointF(x, h - 12))


# ── Main window ───────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trading Arena V6 - Order Book & Analytics")
        self.setMinimumSize(1320, 860)
        self.setStyleSheet(f"QMainWindow {{ background: {COLORS['bg'].name()}; }}")

        self.sim = StockSimulator(100.0)
        self.traders = [
            AITrader("MomentumBot", AI_PALETTE[0], strat_momentum),
            AITrader("MeanRevBot",  AI_PALETTE[1], strat_mean_rev),
            AITrader("RSIHunter",   AI_PALETTE[2], strat_rsi),
            AITrader("ScalperBot",  AI_PALETTE[3], strat_scalper),
            AITrader("TrendRider",  AI_PALETTE[4], strat_trend),
            AITrader("Contrarian",  AI_PALETTE[5], strat_contrarian),
            AITrader("BreakoutBot", AI_PALETTE[6], strat_breakout),
            AITrader("MACDBot",     AI_PALETTE[7], strat_macd),
        ]
        self.running = False
        self.speed_mult = 1.0

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._anim_update)
        self.anim_timer.start(16)
        self.elapsed = QElapsedTimer()
        self.elapsed.start()
        self.last_time = 0

        self._start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_header())

        self.stats_bar = StatsBar(self.sim, self.traders)
        layout.addWidget(self.stats_bar)

        main = QHBoxLayout()
        main.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(8)
        self.price_chart = PriceChartWidget(self.sim, self.traders)
        self.portfolio_chart = PortfolioChartWidget(self.sim, self.traders)
        left.addWidget(self.price_chart, 3)
        left.addWidget(self.portfolio_chart, 2)
        main.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(8)
        self.sentiment_gauge = SentimentGauge(self.sim, self.traders)
        self.leaderboard = LeaderboardWidget(self.sim, self.traders)
        self.order_book = OrderBookWidget(self.sim)
        self.trade_feed = TradeFeedWidget(self.traders)
        
        right.addWidget(self.sentiment_gauge, 1)
        right.addWidget(self.leaderboard, 4)
        right.addWidget(self.order_book, 3)
        right.addWidget(self.trade_feed, 2)
        main.addLayout(right, 1)

        layout.addLayout(main, 1)

    def selected_ai_changed(self, name: Optional[str]):
        self.price_chart.selected_ai = name
        self.portfolio_chart.selected_ai = name
        self.price_chart.update()
        self.portfolio_chart.update()

    def _build_header(self):
        header = QWidget()
        header.setFixedHeight(56)
        h = QHBoxLayout(header)
        h.setContentsMargins(16, 8, 16, 8)
        h.setSpacing(10)

        title = QLabel("◆  AI TRADING ARENA")
        title.setStyleSheet(f"color: {COLORS['text'].name()}; font-size: 16pt; font-weight: bold; letter-spacing: 2px;")
        h.addWidget(title)

        sub = QLabel("· V6: Depth & Interactive Tracking")
        sub.setStyleSheet(f"color: {COLORS['text_dim'].name()}; font-size: 10pt;")
        h.addWidget(sub)

        h.addStretch()

        self.speed_label = QLabel("1x")
        self.speed_label.setStyleSheet(f"color: {COLORS['accent'].name()}; font-size: 11pt; font-weight: bold; min-width: 32px;")
        self.speed_label.setAlignment(Qt.AlignCenter)
        h.addWidget(self.speed_label)

        spd_down = QPushButton("−")
        spd_down.setFixedSize(28, 28)
        spd_down.setStyleSheet(self._btn_style())
        spd_down.clicked.connect(lambda: self._adjust_speed(-1))
        h.addWidget(spd_down)

        spd_up = QPushButton("+")
        spd_up.setFixedSize(28, 28)
        spd_up.setStyleSheet(self._btn_style())
        spd_up.clicked.connect(lambda: self._adjust_speed(1))
        h.addWidget(spd_up)

        fomo_btn = QPushButton("🚀 FOMO PUMP")
        fomo_btn.setStyleSheet("""
            QPushButton {
                background-color: #00ff88; color: #080c18;
                border: none; border-radius: 6px;
                font-size: 10pt; font-weight: bold; letter-spacing: 1px;
                padding: 0 14px;
            }
            QPushButton:hover { background-color: #33ffaa; }
            QPushButton:pressed { background-color: #00cc66; }
        """)
        fomo_btn.setFixedHeight(32)
        fomo_btn.setMinimumWidth(140)
        fomo_btn.clicked.connect(lambda: self.sim.inject_shock(0.1))
        h.addWidget(fomo_btn)

        swan_btn = QPushButton("🔻 BLACK SWAN")
        swan_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4757; color: #ffffff;
                border: none; border-radius: 6px;
                font-size: 10pt; font-weight: bold; letter-spacing: 1px;
                padding: 0 14px;
            }
            QPushButton:hover { background-color: #ff6b7a; }
            QPushButton:pressed { background-color: #cc2c3a; }
        """)
        swan_btn.setFixedHeight(32)
        swan_btn.setMinimumWidth(140)
        swan_btn.clicked.connect(lambda: self.sim.inject_shock(-0.1))
        h.addWidget(swan_btn)

        self.play_btn = QPushButton("⏸  PAUSE")
        self.play_btn.setStyleSheet(self._btn_style(primary=True))
        self.play_btn.setFixedHeight(32)
        self.play_btn.setMinimumWidth(120)
        self.play_btn.clicked.connect(self._toggle_run)
        h.addWidget(self.play_btn)

        rst = QPushButton("↻  RESET")
        rst.setStyleSheet(self._btn_style())
        rst.setFixedHeight(32)
        rst.setMinimumWidth(100)
        rst.clicked.connect(self._reset)
        h.addWidget(rst)

        return header

    def _btn_style(self, primary=False):
        if primary:
            return """
                QPushButton {
                    background-color: #00d4ff; color: #080c18;
                    border: none; border-radius: 6px;
                    font-size: 10pt; font-weight: bold; letter-spacing: 1px;
                    padding: 0 14px;
                }
                QPushButton:hover { background-color: #33ddff; }
                QPushButton:pressed { background-color: #0099cc; }
            """
        return """
            QPushButton {
                background-color: #1a2234; color: #e4e9f2;
                border: 1px solid #25304a; border-radius: 6px;
                font-size: 11pt; font-weight: bold; padding: 0 8px;
            }
            QPushButton:hover { background-color: #25304a; border-color: #00d4ff; color: #00d4ff; }
            QPushButton:pressed { background-color: #161e30; }
        """

    def _adjust_speed(self, delta):
        speeds = [0.5, 1, 2, 4, 8]
        idx = speeds.index(self.speed_mult) if self.speed_mult in speeds else 1
        idx = max(0, min(len(speeds) - 1, idx + delta))
        self.speed_mult = speeds[idx]
        self.speed_label.setText(f"{self.speed_mult:g}x")
        self._update_timer_interval()

    def _update_timer_interval(self):
        if self.running:
            base = 90
            self.timer.start(max(15, int(base / self.speed_mult)))

    def _start(self):
        self.running = True
        self.play_btn.setText("⏸  PAUSE")
        self._update_timer_interval()

    def _toggle_run(self):
        if self.running:
            self.running = False
            self.timer.stop()
            self.play_btn.setText("▶  RESUME")
        else:
            self._start()

    def _reset(self):
        self.sim.reset()
        for t in self.traders: t.reset()
        self.trade_feed.recent.clear()
        self.price_chart.particles.clear()
        self.leaderboard.selected_ai = None
        self.selected_ai_changed(None)
        self._start()

    def _tick(self):
        price = self.sim.next()
        prices = self.sim.history
        vol = self.sim.candles[-1]['v']

        for i, trader in enumerate(self.traders):
            decision, state_str = trader.strategy(trader, prices, None, self.sim.ticks)
            trader.state_text = state_str
            
            if decision is None:
                trader.last_action = "HOLD"
                continue
                
            action, qty = decision
            
            risk_mult = max(0.15, 1.0 - (trader.max_drawdown / 50.0))
            qty = int(qty * risk_mult)
            if qty == 0: continue

            max_exposure = trader.starting_cash * 2.0
            current_exposure = abs(trader.shares * price)
            if current_exposure + qty * price > max_exposure:
                qty = int((max_exposure - current_exposure) / price)
                if qty <= 0: continue

            prev_shares = trader.shares
            
            slippage_pct = min(0.01, (qty * price) / max(10000.0, vol * 100))
            exec_price = price * (1 + slippage_pct) if action == 'BUY' else price * (1 - slippage_pct)
            
            if action == 'BUY':
                cost = qty * exec_price
                if cost <= trader.cash or prev_shares < 0:
                    if prev_shares < 0:
                        cover_qty = min(qty, -prev_shares)
                        trader.cash -= cover_qty * exec_price
                        trader.shares += cover_qty
                        
                        profit = (trader.avg_entry - exec_price) * cover_qty
                        if profit > 0: trader.gross_profit += profit
                        else: trader.gross_loss += abs(profit)
                        if exec_price < trader.avg_entry: trader.wins += 1
                        else: trader.losses += 1
                        
                        # Record Closed Trade
                        trader.closed_trades.append(ClosedTrade(
                            trader.entry_tick, trader.avg_entry, self.sim.ticks, exec_price, cover_qty, profit
                        ))
                        if len(trader.closed_trades) > 50: trader.closed_trades.pop(0)
                        
                        act_label = "COVER"
                        if trader.shares == 0: 
                            trader.avg_entry = 0.0
                            trader.entry_tick = -1
                    else:
                        trader.cash -= cost
                        trader.shares += qty
                        if prev_shares == 0: trader.entry_tick = self.sim.ticks
                        trader.avg_entry = (trader.avg_entry * prev_shares + exec_price * qty) / trader.shares
                        act_label = "BUY"
                        
                    trader.trades.append(Trade(i, act_label, qty, exec_price, self.sim.ticks))
                    trader.last_action = f"{act_label}  +{qty}"
                    trader.pulse = 1.0
                    self.trade_feed.add_trade(i, act_label, qty, exec_price, self.sim.ticks)
                    self._spawn_trade_particles(exec_price, trader.color)
                    
            elif action == 'SELL':
                if prev_shares > 0:
                    sell_qty = min(qty, prev_shares)
                    trader.cash += sell_qty * exec_price
                    trader.shares -= sell_qty
                    
                    profit = (exec_price - trader.avg_entry) * sell_qty
                    if profit > 0: trader.gross_profit += profit
                    else: trader.gross_loss += abs(profit)
                    if exec_price > trader.avg_entry: trader.wins += 1
                    else: trader.losses += 1
                    
                    # Record Closed Trade
                    trader.closed_trades.append(ClosedTrade(
                        trader.entry_tick, trader.avg_entry, self.sim.ticks, exec_price, sell_qty, profit
                    ))
                    if len(trader.closed_trades) > 50: trader.closed_trades.pop(0)
                    
                    act_label = "SELL"
                    if trader.shares == 0: 
                        trader.avg_entry = 0.0
                        trader.entry_tick = -1
                else:
                    trader.cash += qty * exec_price
                    trader.shares -= qty
                    if prev_shares == 0: 
                        trader.avg_entry = exec_price
                        trader.entry_tick = self.sim.ticks
                    else:
                        trader.avg_entry = (trader.avg_entry * abs(prev_shares) + exec_price * qty) / abs(trader.shares)
                    act_label = "SHORT"
                    
                trader.trades.append(Trade(i, act_label, qty, exec_price, self.sim.ticks))
                trader.last_action = f"{act_label}  -{qty}"
                trader.pulse = 1.0
                self.trade_feed.add_trade(i, act_label, qty, exec_price, self.sim.ticks)
                self._spawn_trade_particles(exec_price, trader.color)

        # Rank History
        ranked = sorted(self.traders, key=lambda t: -t.value_at(price))
        for rank, t in enumerate(ranked):
            t.rank_history.append(rank + 1)
            if len(t.rank_history) > 50: t.rank_history.pop(0)

        for trader in self.traders:
            cur_val = trader.value_at(price)
            if cur_val > trader.peak_value:
                trader.peak_value = cur_val
            dd = (trader.peak_value - cur_val) / trader.peak_value * 100.0 if trader.peak_value > 0 else 0.0
            if dd > trader.max_drawdown:
                trader.max_drawdown = dd
                
            trader.history.append(cur_val)
            if len(trader.history) > 2000:
                trader.history = trader.history[-2000:]
            if len(trader.trades) > 250:
                trader.trades = trader.trades[-250:]

    def _spawn_trade_particles(self, price: float, color: QColor):
        chart = self.price_chart
        w, h = chart.width(), chart.height()
        m_left, m_right, m_top, m_bot = 64, 100, 38, 100
        cw, ch = w - m_left - m_right, h - m_top - m_bot
        if cw < 10 or ch < 10: return
        vis_candles = self.sim.candles[-chart.window_size:]
        if len(vis_candles) < 2: return
        p_min = min(c['l'] for c in vis_candles) * 0.995
        p_max = max(c['h'] for c in vis_candles) * 1.005
        p_rng = max(0.01, p_max - p_min)
        idx = len(vis_candles) - 1
        sx = m_left + (idx / max(1, len(vis_candles) - 1)) * cw
        sy = m_top + (1 - (price - p_min) / p_rng) * ch
        chart.add_trade_effect(sx, sy, color)

    def _anim_update(self):
        now = self.elapsed.elapsed()
        dt = (now - self.last_time) / 1000.0
        self.last_time = now
        dt = min(dt, 0.05)

        for t in self.traders:
            t.pulse = max(0.0, t.pulse - dt * 2.0)
            
        if self.sim.news_timer > 0:
            self.sim.news_timer -= dt

        self.price_chart.update_anim(dt)

        self.price_chart.update()
        self.portfolio_chart.update()
        self.leaderboard.update()
        self.trade_feed.update()
        self.stats_bar.update()
        self.sentiment_gauge.update()
        self.order_book.update()

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Space: self._toggle_run()
        elif k == Qt.Key_R: self._reset()
        elif k == Qt.Key_Escape: self.close()
        elif k in (Qt.Key_Plus, Qt.Key_Equal): self._adjust_speed(1)
        elif k == Qt.Key_Minus: self._adjust_speed(-1)
        elif k == Qt.Key_B: self.sim.inject_shock(-0.1)
        elif k == Qt.Key_P: self.sim.inject_shock(0.1)
        else: super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self.anim_timer.stop()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(pal.Window, COLORS['bg'])
    pal.setColor(pal.WindowText, COLORS['text'])
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()