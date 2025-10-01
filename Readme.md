# Trade Simulator - Real-Time Market Impact Analysis System

## Overview

This is a high-performance trade simulator that connects to live cryptocurrency exchange WebSocket feeds to estimate transaction costs and market impact in real-time. The system processes L2 orderbook data streams from OKX exchange and calculates multiple trading metrics including slippage, fees, market impact, and maker/taker proportions using machine learning models and quantitative finance algorithms.

The application is built with Streamlit for the UI, processes real-time WebSocket data, and employs statistical models (linear regression for slippage, logistic regression for maker/taker classification) alongside the Almgren-Chriss model for market impact estimation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology**: Streamlit web framework

**Design Pattern**: Single-page application with two-panel layout

- Left panel displays input parameters (exchange, symbol, quantity, fee tier, volatility)
- Right panel shows calculated outputs (slippage, fees, market impact, net cost, latency metrics)

**State Management**: Streamlit's session_state is used to persist:

- WebSocket connection status and client instance
- Orderbook data snapshots
- Historical performance metrics (processing times, UI update times, end-to-end latency)
- Calculator instances and recorded data

**Rationale**: Streamlit was chosen for rapid UI development with minimal frontend code. The session_state approach maintains consistency across Streamlit's rerun model while avoiding global variables.

### Backend Architecture

**WebSocket Client** (`websocket_client.py`)

- Async/await pattern using `websockets` library
- Automatic reconnection with exponential backoff (max 5 retries)
- Ping/pong keepalive (20s interval, 10s timeout)
- Callback-based message handling for decoupling data ingestion from processing

**Trading Calculator** (`calculations.py`)

- Centralized calculation engine that orchestrates multiple models
- Feature extraction pipeline converts orderbook snapshots to ML-ready feature vectors
- Collects training data for continuous model improvement
- Separates concerns: feature engineering, model invocation, metrics calculation

**Model Layer** (`models.py`)

- Three specialized models:
  - `SlippageModel`: Linear regression with StandardScaler preprocessing
  - `MakerTakerModel`: Logistic regression for order type classification
  - `AlmgrenChrissModel`: Quantitative finance model for market impact
- Each model encapsulates training, prediction, and state management
- Graceful degradation: models return baseline estimates when untrained

**Fee Calculator** (`fee_calculator.py`)

- Rule-based fee system supporting multiple exchanges (OKX, Binance, Coinbase, Kraken)
- Tiered fee structure (VIP levels with different maker/taker rates)
- Pure function design for deterministic fee calculation

**Threading Model**

- Main thread: Streamlit UI and synchronous operations
- WebSocket thread: Async event loop for real-time data streaming
- Thread-safe communication via session_state updates
- Proper cleanup on disconnect to prevent resource leaks

**Rationale**: Async WebSocket handling prevents blocking while Streamlit's synchronous model remains simple. The threading separation isolates network I/O from UI rendering for better performance.

### Data Processing Pipeline

**Flow**:

1. WebSocket receives orderbook JSON
2. Message callback updates session_state.orderbook_data
3. Calculator extracts features (spread, depth, imbalance metrics)
4. Models predict slippage, maker/taker probability, market impact
5. Fee calculator applies exchange-specific rules
6. Aggregate metrics displayed in UI

**Performance Optimization**:

- Limited orderbook depth analysis (top 10 levels) for consistent processing time
- Deque-based circular buffers (maxlen=100) for memory-efficient history tracking
- Non-blocking CPU sampling in profiler to avoid measurement overhead
- Snapshot throttling (1 Hz) to reduce profiling overhead

### Data Storage

**In-Memory Storage**:

- No persistent database; all data held in session_state
- Orderbook history stored as list of dictionaries
- Model training data accumulated in NumPy arrays

**Export Capabilities** (`export_manager.py`):

- **Simulation Results Export**: CSV/JSON formats with complete metrics (slippage, fees, market impact, maker/taker ratios, latency)
- **Performance Reports Export**: CSV/JSON formats with statistical analysis (mean, P50, P95, P99) of latency metrics, optional memory/thread profiling
- **Historical Backtest Export**: CSV format for replay session results
- **Security Features**: Filename sanitization (path.basename + extension enforcement) to prevent path traversal attacks
- **Data Alignment**: Export fields match actual calculated metrics with derived percentages computed on export

**Rationale**: In-memory storage suits the real-time nature and ephemeral session model. Export provides persistence only when explicitly requested, with separate formats for different analysis workflows.

### Machine Learning Architecture

**Training Strategy**:

- Online learning approach: accumulate data during live streaming
- Models train when sufficient samples collected (minimum 10 for slippage)
- Feature vectors include: quantity, spread, depth imbalance, volatility, price levels

**Model Selection**:

- Linear regression for slippage: assumes linear relationship between trade size and price impact
- Logistic regression for maker/taker: binary classification based on orderbook state
- Almgren-Chriss: closed-form solution (no training required), industry-standard for temporary market impact

**Alternatives Considered**:

- Deep learning models: rejected due to data requirements and latency constraints
- Non-parametric models (KNN, decision trees): rejected for interpretability and speed

### Performance Monitoring

**System Profiler** (`profiling.py`):

- Memory tracking: RSS, VMS, available memory
- Thread monitoring: active count, names, daemon status
- CPU usage: non-blocking sampling via psutil
- Snapshot history with circular buffer

**Latency Tracking**:

- Processing time: data parsing to metric calculation
- UI update time: rendering overhead
- End-to-end time: complete simulation loop
- All stored as lists in session_state for trend analysis

**Rationale**: Comprehensive profiling ensures the system meets the requirement to process data faster than stream reception rate.

### Historical Replay System

**Replay Engine** (`replay.py`):

- Records timestamped orderbook snapshots during live streaming
- Playback at configurable speeds (1x, 2x, etc.)
- Separate thread for non-blocking replay
- JSON serialization for portability

**Use Cases**:

- Backtesting strategy modifications
- Debugging edge cases
- Performance regression testing

**Rationale**: Decouples data collection from analysis, enabling offline development and testing without live connections.

## External Dependencies

### WebSocket Data Source

**OKX Exchange WebSocket**

- Endpoint: `wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP`
- Protocol: L2 orderbook snapshots with timestamp, exchange, symbol, asks/bids arrays
- Format: JSON with price-quantity tuples `[["price", "qty"], ...]`
- Update frequency: Real-time (tick-by-tick)
- Note: Requires VPN access for OKX connectivity

### Python Libraries

**Core Framework**:

- `streamlit`: Web UI framework for rapid prototyping
- `websockets`: Async WebSocket client library

**Data Processing**:

- `pandas`: Data manipulation and CSV export
- `numpy`: Numerical computing and array operations

**Machine Learning**:

- `scikit-learn`: Linear regression, logistic regression, StandardScaler

**System Monitoring**:

- `psutil`: Cross-platform process and system monitoring

**Standard Library**:

- `asyncio`: Async/await runtime
- `threading`: Thread management for WebSocket isolation
- `json`: Data serialization

### Exchange Fee Structures

**Supported Exchanges**:

- OKX: 6 VIP tiers (0.08% to 0.02% maker, 0.10% to 0.05% taker)
- Binance: 4 tiers (0.10% to 0.06% maker)
- Coinbase: 2 tiers (0.40% to 0.25% maker)
- Kraken: 3 tiers (0.16% to 0.12% maker)

**Integration**: Hardcoded fee schedules in `fee_calculator.py` based on exchange documentation

### Quantitative Finance Models

**Almgren-Chriss Model**:

- Reference: LinkedIn article on optimal portfolio execution
- Parameters: volatility, trade quantity, risk aversion, time horizon
- Output: Estimated permanent and temporary market impact
- Implementation: Closed-form mathematical formula (no external API)
