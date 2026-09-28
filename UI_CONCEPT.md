# BTQuant UI Architecture Concept

## Design Philosophy: "Simplicity Through Clarity"

BTQuant is a high-frequency trading system handling 4,381 trades/sec with 5 parallel C++ detectors analyzing market manipulation. The UI must hide this complexity while providing precise control.

---

## Visual Language: Precision Dark + Fintech Purple

### Hybrid Color System
- **Base**: Linear's near-black canvas (`#08090a`) for reduced eye strain during long sessions
- **Accents**: Kraken purple (`#7132f5`) for financial actions, signals, and primary CTAs
- **Success**: Linear green (`#27a644`) for profitable trades, active status
- **Warning**: Amber (`#ffa500`) for risk alerts, drawdown warnings

### Typography
- **Inter Variable** with `cv01, ss03` features (signature 510 weight for UI text)
- **JetBrains Mono** for P&L numbers, trade IDs, timestamps (technical clarity)
- **Negative letter-spacing** on headlines (-1.056px at 32px) for compressed, authoritative feel

---

## Core UI Architecture

### 1. Command Palette Navigation (Primary)
```
Cmd+K or / opens unified search
→ "Run SMA_Cross on BTC/USDT (24h)" → instant execution
→ "Show agency strategies" → filtered view
→ "Pause detector: stop_hunt" → immediate toggle
```

### 2. Three-Layer Mental Model (Simplified)
Instead of exposing C++/Python/AI layers, present:

```
┌─────────────────────────────────────────┐
│  [DATA]     [STRATEGY]     [EXECUTION] │
│  Real-time   Signal Gen      Trade Result │
└─────────────────────────────────────────┘
```

**Layer Abstraction:**
- DATA: "Feed Health" panel - not "HotSpine SHM metrics"
- STRATEGY: "Signal Matrix" - not "Backtrader + Evolution Engine"
- EXECUTION: "Trade Flow" - not "CCXTBroker + Telegram alerts"

### 3. Information Architecture

#### A. Main Dashboard (Primary View)
```
┌─[Header: Balance, Equity, daily P&L (↑ $1,247)]────────────────────┐
│                                                                    │
│  [Signal Matrix]                                                   │
│  ┌────┬──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │Sym │ Position │ Signal   │ Confidence│ Detector │ Action   │   │
│  │BTC │ 12.4 ETH │ LONG     │ 87%      │ ✓        │ EXECUTE  │   │
│  │ETH │ -3.2 BTC │ SHORT    │ 73%      │ ✓        │ EXECUTE  │   │
│  │SOL │ 0        │ WAIT     │ --       │ ✗        │ ---      │   │
│  └────┴──────────┴──────────┴──────────┴──────────┴──────────┘   │
│                                                                    │
│  [Live Trade Feed]                                                  │
│  • BTC/USDT long filled @ $97,433 (0.02s latency)               │
│  • ETH/USDT short executed @ $2,654                               │
│  • SOL/USDT signal rejected (confidence < 60%)                     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

#### B. Deep Dive Panels (Progressive Disclosure)

**1. Signal Matrix (Strategy Layer)**
- Columns: Symbol, Strategy, Timeframe, Signal, Confidence, Sharpe, Max DD, Actions
- Each row: Expandable to show strategy parameters, indicator values
- Color: Purple accent on active signals, green on executed

**2. Feed Health (Data Layer)**
- Real-time latency (µs), throughput, connection status
- Detector status (stop_hunt, liquidity_imbalance, whale_frontrun, spread_arbitrage, spoofing)
- Shared memory buffer status (green/yellow/red)

**3. Performance (Execution Layer)**
- Equity curve, drawdown, win rate, avg holding time
- Strategy-by-strategy breakdown
- Export to QuantStats button

### 4. Interaction Patterns

#### A. Microsecond Feedback Loop
- All actions: <100ms perceived response
- Skeleton screens while loading strategy configs
- Optimistic UI: Show trade as "PENDING" immediately, confirm when broker responds

#### B. Risk Control Overlay
```
When max_drawdown_limit hits 15%:
┌─────────────────────────────────────────┐
│ ⚠️  MAX DRAWDOWN LIMIT (15%) REACHED    │
│ Current DD: 15.3% across 3 strategies   │
│ [Pause All] [Reduce Position] [Analyze]  │
└─────────────────────────────────────────┘
```

#### C. Error Recovery
- Failed strategy: "Retry" button, not technical logs
- Connection lost: "Reconnecting..." with retry counter
- Graceful degradation: Core trading continues if Telegram alert fails

---

## Component Library

### Cards (Elevated Surfaces)
```css
/* Level 2 surface - strategy cards */
background: rgba(255,255,255,0.02);
border: 1px solid rgba(255,255,255,0.08);
border-radius: 8px;
```

### Buttons
```css
/* Primary - Execute/Deploy */
background: #7132f5;  /* Kraken Purple */
color: #ffffff;
padding: 8px 16px;
border-radius: 6px;

/* Ghost - Secondary actions */
background: rgba(255,255,255,0.02);
color: #e2e4e7;
border: 1px solid rgb(36, 40, 44);
border-radius: 6px;

/* Subtle - Internal controls */
background: rgba(255,255,255,0.04);
padding: 0 6px;
border-radius: 6px;
```

### Status Indicators
- **Active**: `#27a644` (green dot) - strategy running
- **Pending**: `#7170ff` (violet pulse) - signal waiting execution  
- **Paused**: `#62666d` (gray) - manual override
- **Error**: `#ff4444` (red) - failed state

### Data Tables (Trading Feeds)
- Fixed header, scrollable body
- Right-aligned numbers, monospace font
- Zebra striping with subtle `#23252a` dividers
- Hover: slight opacity increase (`rgba(255,255,255,0.04)`)

---

## Real-Time WebSocket Integration

### Event Flow
```
1. CCAPI → HotSpine SHM (/dev/shm/BTQ...)
2. HotSpine Reader lib → MCP → WebSocket
3. WebSocket → React state -> UI update (<100ms)
4. User action → WebSocket -> Python broker
5. Broker ACK -> UI confirmation
```

### Update Strategies
- **Immediate**: P&L, positions, latency (every tick)
- **Debounced**: Chart updates (100ms intervals)
- **Batched**: Detector alerts (group by symbol)

---

## Accessibility Considerations

- Tab order: Header → Signal Matrix → Controls → Live Feed
- ARIA labels: "BTC-USDT signal: LONG, confidence 87%, execute trade"
- High contrast: `#f7f8f8` on `#08090a` (ratio > 15:1)
- Keyboard shortcuts: 
  - `Space` - pause/resume selected strategy
  - `E` - execute signal
  - `R` - refresh feed
  - `/` - open command palette

---

## Responsive Design

### Breakpoints
- **Mobile** (<640px): Signal list, P&L summary, essential controls only
- **Tablet** (640-1024px): Matrix + one detail panel
- **Desktop** (>1024px): Full three-panel view

### Priority Content
1. Position/P&L (always visible)
2. Signal Matrix (condenses on mobile)
3. Live Feed (full on desktop, collapsed on mobile)

---

## Technical Implementation Stack

### Frontend
- React + TypeScript (strong typing for financial data)
- WebSocket for real-time updates
- TailwindCSS with custom theme (Linear dark + Kraken purple)
- Recharts for equity curves (lightweight)

### Backend Bridge
- FastAPI WebSocket endpoint (`/ws`)
- Connects to HotSpine reader lib
- Proxies to CCXTBroker when needed
- Serves QuantStats reports

### Deployment
- Static build served by FastAPI
- Health endpoint: `/health` returns `{status: "ok", latency_ms: 23}`
- Metrics endpoint: `/metrics` returns Prometheus format

---

## Validation Checklist

- [ ] All interactive elements have visible focus states
- [ ] Color contrast meets WCAG AA (4.5:1 minimum)
- [ ] Keyboard navigation complete (tab, space, enter)
- [ ] Screen reader announces: "Signal: BTC-USDT LONG, confidence 87 percentage"
- [ ] Loading states show immediately (skeleton screens)
- [ ] Error states provide actionable recovery
- [ ] Mobile view shows P&L + signal summary
- [ ] 100ms target met for all user actions
- [ ] Offline queue syncs trades when connection restored
- [ ] Undo available for manual trades (30s window)

---

## Mock Data Structure

```typescript
interface Signal {
  symbol: string;           // "BTC/USDT"
  strategy: string;         // "SMA_Cross_Simple"
  signal: "LONG" | "SHORT" | "WAIT";
  confidence: number;       // 0-100
  entry_price: number;      // predicted entry
  sharpe: number;           // strategy sharpe ratio
  max_dd: number;           // max drawdown %
  detector_status: "ok" | "warning" | "error";
  position: number;       // current position size
  latency_us: number;       // microseconds
}
```

---

*This concept prioritizes simplicity while respecting BTQuant's full complexity. The real-time nature (4,381 trades/sec) demands instant feedback; the hybrid C++/Python/AI stack demands abstraction. The result: a precision interface that feels calm despite the computational storm underneath.*