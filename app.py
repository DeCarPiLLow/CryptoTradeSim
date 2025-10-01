import streamlit as st
import asyncio
import threading
import time
from datetime import datetime
import pandas as pd
import numpy as np
from websocket_client import OrderbookWebSocketClient
from calculations import TradingCalculator
from fee_calculator import FeeCalculator
from profiling import SystemProfiler
from replay import HistoricalReplayEngine
from export_manager import ExportManager
import json

# Page configuration
st.set_page_config(
    page_title="Trade Simulator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'orderbook_data' not in st.session_state:
    st.session_state.orderbook_data = None
if 'ws_client' not in st.session_state:
    st.session_state.ws_client = None
if 'ws_thread' not in st.session_state:
    st.session_state.ws_thread = None
if 'is_connected' not in st.session_state:
    st.session_state.is_connected = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'orderbook_history' not in st.session_state:
    st.session_state.orderbook_history = []
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = []
if 'ui_update_times' not in st.session_state:
    st.session_state.ui_update_times = []
if 'end_to_end_times' not in st.session_state:
    st.session_state.end_to_end_times = []
if 'last_start_time' not in st.session_state:
    st.session_state.last_start_time = None
if 'ui_render_start' not in st.session_state:
    st.session_state.ui_render_start = None

# Initialize calculator
if 'calculator' not in st.session_state:
    st.session_state.calculator = TradingCalculator()

# Initialize fee calculator
if 'fee_calculator' not in st.session_state:
    st.session_state.fee_calculator = FeeCalculator()

# Initialize system profiler
if 'profiler' not in st.session_state:
    st.session_state.profiler = SystemProfiler()

# Initialize replay engine
if 'replay_engine' not in st.session_state:
    st.session_state.replay_engine = HistoricalReplayEngine()

def get_websocket_url(exchange: str, symbol: str) -> str:
    """Generate WebSocket URL based on exchange and symbol"""
    base_url = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook"
    return f"{base_url}/{exchange.lower()}/{symbol}"

def start_websocket(exchange: str, symbol: str):
    """Start WebSocket connection in a separate thread"""
    if st.session_state.ws_client is None or not st.session_state.is_connected:
        url = get_websocket_url(exchange, symbol)
        st.session_state.ws_client = OrderbookWebSocketClient(
            url=url,
            on_message_callback=on_orderbook_update
        )
        
        def run_websocket():
            try:
                asyncio.run(st.session_state.ws_client.connect())
            except Exception as e:
                st.session_state.is_connected = False
                st.error(f"WebSocket error: {str(e)}")
        
        st.session_state.ws_thread = threading.Thread(target=run_websocket, daemon=True)
        st.session_state.ws_thread.start()
        st.session_state.is_connected = True

def on_orderbook_update(data):
    """Callback function when orderbook data is received"""
    start_time = time.perf_counter()
    
    st.session_state.orderbook_data = data
    
    # Record data if recording is enabled
    if st.session_state.replay_engine.is_recording:
        st.session_state.replay_engine.record_orderbook(data)
    
    # Store orderbook history for regression models (keep last 1000)
    st.session_state.orderbook_history.append(data)
    if len(st.session_state.orderbook_history) > 1000:
        st.session_state.orderbook_history.pop(0)
    
    # Track end-to-end time (interval between message arrivals)
    if st.session_state.last_start_time is not None:
        end_to_end = (start_time - st.session_state.last_start_time) * 1000  # ms
        st.session_state.end_to_end_times.append(end_to_end)
        if len(st.session_state.end_to_end_times) > 100:
            st.session_state.end_to_end_times.pop(0)
    st.session_state.last_start_time = start_time
    
    # Calculate metrics if we have input parameters
    if 'quantity_usd' in st.session_state and st.session_state.quantity_usd:
        calculate_metrics()
    
    # Track processing time
    processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
    st.session_state.processing_times.append(processing_time)
    if len(st.session_state.processing_times) > 100:
        st.session_state.processing_times.pop(0)

def calculate_metrics():
    """Calculate all trading metrics"""
    if st.session_state.orderbook_data is None:
        return
    
    orderbook = st.session_state.orderbook_data
    quantity_usd = st.session_state.get('quantity_usd', 100)
    order_type = st.session_state.get('order_type', 'market')
    volatility = st.session_state.get('volatility', 0.02)
    fee_tier = st.session_state.get('fee_tier', 'VIP 0')
    
    # Parse orderbook
    asks = [(float(price), float(qty)) for price, qty in orderbook.get('asks', [])]
    bids = [(float(price), float(qty)) for price, qty in orderbook.get('bids', [])]
    
    if not asks or not bids:
        return
    
    mid_price = (asks[0][0] + bids[0][0]) / 2
    quantity_crypto = quantity_usd / mid_price
    
    # Calculate slippage using regression model
    slippage = st.session_state.calculator.calculate_slippage(
        orderbook_history=st.session_state.orderbook_history,
        quantity=quantity_crypto,
        side='buy'
    )
    
    # Calculate fees using the selected exchange
    selected_exchange = st.session_state.get('selected_exchange', 'OKX')
    fees = st.session_state.fee_calculator.calculate_fees(
        exchange=selected_exchange,
        quantity_usd=quantity_usd,
        fee_tier=fee_tier,
        order_type='market'
    )
    
    # Calculate market impact using Almgren-Chriss model
    market_impact = st.session_state.calculator.calculate_market_impact_almgren_chriss(
        quantity=quantity_crypto,
        volatility=volatility,
        mid_price=mid_price,
        orderbook_depth=len(bids)
    )
    
    # Calculate maker/taker proportion
    maker_taker_ratio = st.session_state.calculator.predict_maker_taker_proportion(
        orderbook_history=st.session_state.orderbook_history,
        quantity=quantity_crypto
    )
    
    # Calculate net cost
    net_cost = slippage + fees + market_impact
    
    # Calculate internal latency
    avg_latency = np.mean(st.session_state.processing_times) if st.session_state.processing_times else 0
    
    # Store metrics
    st.session_state.metrics = {
        'slippage': slippage,
        'fees': fees,
        'market_impact': market_impact,
        'net_cost': net_cost,
        'maker_ratio': maker_taker_ratio,
        'taker_ratio': 1 - maker_taker_ratio,
        'internal_latency': avg_latency,
        'mid_price': mid_price,
        'quantity_crypto': quantity_crypto
    }

# Track UI render start time
ui_render_start = time.perf_counter()

# Main UI
st.title("🔄 Real-Time Trade Simulator")

# Create two columns for the layout
col1, col2 = st.columns([1, 1])

# Left Panel - Input Parameters
with col1:
    st.header("📊 Input Parameters")
    
    # Exchange selection
    exchange = st.selectbox(
        "Exchange",
        ["OKX", "Binance", "Coinbase", "Kraken"],
        index=0,
        help="Select cryptocurrency exchange"
    )
    
    # Symbol mapping for each exchange
    symbol_options = {
        "OKX": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "BNB-USDT-SWAP"],
        "Binance": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
        "Coinbase": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "Kraken": ["XBTUSD", "ETHUSD", "SOLUSD"]
    }
    
    # Spot Asset
    spot_asset = st.selectbox(
        "Spot Asset",
        symbol_options.get(exchange, ["BTC-USDT-SWAP"]),
        index=0,
        help="Select cryptocurrency trading pair"
    )
    
    # Store selected exchange and symbol
    st.session_state.selected_exchange = exchange
    st.session_state.selected_symbol = spot_asset
    
    # Order Type
    order_type = st.selectbox(
        "Order Type",
        ["market"],
        index=0
    )
    st.session_state.order_type = order_type
    
    # Quantity
    quantity_usd = st.number_input(
        "Quantity (USD equivalent)",
        min_value=1.0,
        value=100.0,
        step=10.0,
        help="Enter the quantity in USD equivalent"
    )
    st.session_state.quantity_usd = quantity_usd
    
    # Volatility
    volatility = st.number_input(
        "Volatility (market parameter)",
        min_value=0.001,
        max_value=1.0,
        value=0.02,
        step=0.001,
        format="%.3f",
        help="Market volatility parameter for Almgren-Chriss model"
    )
    st.session_state.volatility = volatility
    
    # Fee Tier - dynamic based on exchange
    available_tiers = st.session_state.fee_calculator.get_fee_tiers_for_exchange(exchange)
    fee_tier = st.selectbox(
        "Fee Tier",
        available_tiers,
        index=0,
        help=f"Select your {exchange} fee tier"
    )
    st.session_state.fee_tier = fee_tier
    
    st.divider()
    
    # WebSocket connection control
    if not st.session_state.is_connected:
        if st.button("🔌 Connect to WebSocket", type="primary", use_container_width=True):
            start_websocket(exchange, spot_asset)
            st.rerun()
    else:
        current_pair = f"{st.session_state.get('selected_exchange', 'OKX')} - {st.session_state.get('selected_symbol', 'BTC-USDT-SWAP')}"
        st.success(f"✅ Connected to {current_pair}")
        if st.button("🔌 Disconnect", type="secondary", use_container_width=True):
            if st.session_state.ws_client:
                st.session_state.ws_client.close()
            st.session_state.is_connected = False
            st.session_state.ws_client = None
            st.rerun()

# Right Panel - Output Parameters
with col2:
    st.header("📈 Output Metrics")
    
    # Show metrics when connected OR replaying
    if (st.session_state.is_connected or st.session_state.replay_engine.is_replaying) and st.session_state.orderbook_data:
        # Display timestamp
        timestamp = st.session_state.orderbook_data.get('timestamp', 'N/A')
        st.caption(f"Last Update: {timestamp}")
        
        if st.session_state.metrics:
            metrics = st.session_state.metrics
            
            # Display metrics in a grid
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric(
                    "Expected Slippage",
                    f"${metrics['slippage']:.4f}",
                    help="Calculated using linear regression on orderbook depth"
                )
                
                st.metric(
                    "Expected Fees",
                    f"${metrics['fees']:.4f}",
                    help="Rule-based fee calculation based on exchange fee tiers"
                )
                
                st.metric(
                    "Expected Market Impact",
                    f"${metrics['market_impact']:.4f}",
                    help="Almgren-Chriss model estimation"
                )
                
                st.metric(
                    "Net Cost",
                    f"${metrics['net_cost']:.4f}",
                    help="Slippage + Fees + Market Impact"
                )
            
            with metric_col2:
                st.metric(
                    "Maker Proportion",
                    f"{metrics['maker_ratio']:.2%}",
                    help="Predicted using logistic regression"
                )
                
                st.metric(
                    "Taker Proportion",
                    f"{metrics['taker_ratio']:.2%}",
                    help="Predicted using logistic regression"
                )
                
                st.metric(
                    "Internal Latency",
                    f"{metrics['internal_latency']:.3f} ms",
                    help="Processing time per tick (average of last 100)"
                )
                
                st.metric(
                    "Mid Price",
                    f"${metrics['mid_price']:.2f}",
                    help="Current mid price from orderbook"
                )
            
            st.divider()
            
            # Additional information
            with st.expander("📊 Orderbook Summary"):
                orderbook = st.session_state.orderbook_data
                st.write(f"**Exchange:** {orderbook.get('exchange', 'N/A')}")
                st.write(f"**Symbol:** {orderbook.get('symbol', 'N/A')}")
                st.write(f"**Quantity (Crypto):** {metrics['quantity_crypto']:.8f}")
                st.write(f"**Ask Levels:** {len(orderbook.get('asks', []))}")
                st.write(f"**Bid Levels:** {len(orderbook.get('bids', []))}")
                
                # Display top 5 asks and bids
                st.write("**Top 5 Asks:**")
                asks_df = pd.DataFrame(orderbook.get('asks', [])[:5], columns=['Price', 'Quantity'])
                st.dataframe(asks_df, use_container_width=True, hide_index=True)
                
                st.write("**Top 5 Bids:**")
                bids_df = pd.DataFrame(orderbook.get('bids', [])[:5], columns=['Price', 'Quantity'])
                st.dataframe(bids_df, use_container_width=True, hide_index=True)
        else:
            st.info("Waiting for metrics calculation...")
    else:
        st.info("👈 Connect to WebSocket to see real-time metrics")

# Performance Benchmarking Dashboard
st.divider()
st.header("⚡ Performance Benchmarking Dashboard")

if st.session_state.is_connected and st.session_state.processing_times:
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.subheader("Data Processing Latency")
        avg_processing = np.mean(st.session_state.processing_times)
        max_processing = np.max(st.session_state.processing_times)
        min_processing = np.min(st.session_state.processing_times)
        
        st.metric("Average", f"{avg_processing:.3f} ms")
        st.metric("Max", f"{max_processing:.3f} ms")
        st.metric("Min", f"{min_processing:.3f} ms")
        
        # Show histogram
        if len(st.session_state.processing_times) > 10:
            st.line_chart(st.session_state.processing_times[-50:])
    
    with perf_col2:
        st.subheader("UI Update Latency")
        if st.session_state.ui_update_times:
            avg_ui = np.mean(st.session_state.ui_update_times)
            max_ui = np.max(st.session_state.ui_update_times)
            min_ui = np.min(st.session_state.ui_update_times)
            
            st.metric("Average", f"{avg_ui:.3f} ms")
            st.metric("Max", f"{max_ui:.3f} ms")
            st.metric("Min", f"{min_ui:.3f} ms")
            
            if len(st.session_state.ui_update_times) > 10:
                st.line_chart(st.session_state.ui_update_times[-50:])
        else:
            st.info("Collecting data...")
    
    with perf_col3:
        st.subheader("End-to-End Loop Latency")
        if st.session_state.end_to_end_times:
            avg_e2e = np.mean(st.session_state.end_to_end_times)
            max_e2e = np.max(st.session_state.end_to_end_times)
            min_e2e = np.min(st.session_state.end_to_end_times)
            
            st.metric("Average", f"{avg_e2e:.3f} ms")
            st.metric("Max", f"{max_e2e:.3f} ms")
            st.metric("Min", f"{min_e2e:.3f} ms")
            
            if len(st.session_state.end_to_end_times) > 10:
                st.line_chart(st.session_state.end_to_end_times[-50:])
        else:
            st.info("Collecting data...")
    
    # Second row for throughput & stats
    st.divider()
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        # Calculate throughput (messages per second)
        if st.session_state.end_to_end_times:
            avg_interval = np.mean(st.session_state.end_to_end_times) / 1000  # Convert to seconds
            throughput = 1 / avg_interval if avg_interval > 0 else 0
            st.metric("Throughput", f"{throughput:.2f} msg/s")
    
    with stat_col2:
        st.metric("Total Messages", len(st.session_state.orderbook_history))
        
    with stat_col3:
        # Performance status
        if avg_processing < 10:
            st.success("✅ Excellent Performance")
        elif avg_processing < 50:
            st.info("ℹ️ Good Performance")
        else:
            st.warning("⚠️ Performance Degradation")
    
    # Detailed performance analysis
    with st.expander("📊 Detailed Performance Analysis"):
        st.write("### Latency Statistics (milliseconds)")
        
        perf_df = pd.DataFrame({
            'Metric': ['Data Processing', 'UI Update', 'End-to-End Loop'],
            'Average (ms)': [
                np.mean(st.session_state.processing_times),
                np.mean(st.session_state.ui_update_times) if st.session_state.ui_update_times else 0,
                np.mean(st.session_state.end_to_end_times) if st.session_state.end_to_end_times else 0
            ],
            'Std Dev (ms)': [
                np.std(st.session_state.processing_times),
                np.std(st.session_state.ui_update_times) if st.session_state.ui_update_times else 0,
                np.std(st.session_state.end_to_end_times) if st.session_state.end_to_end_times else 0
            ],
            'P95 (ms)': [
                np.percentile(st.session_state.processing_times, 95) if st.session_state.processing_times else 0,
                np.percentile(st.session_state.ui_update_times, 95) if st.session_state.ui_update_times else 0,
                np.percentile(st.session_state.end_to_end_times, 95) if st.session_state.end_to_end_times else 0
            ],
            'P99 (ms)': [
                np.percentile(st.session_state.processing_times, 99) if st.session_state.processing_times else 0,
                np.percentile(st.session_state.ui_update_times, 99) if st.session_state.ui_update_times else 0,
                np.percentile(st.session_state.end_to_end_times, 99) if st.session_state.end_to_end_times else 0
            ]
        })
        
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        st.write("### Performance Notes")
        st.write("""
        - **Data Processing Latency**: Time to process incoming orderbook data and calculate metrics
        - **UI Update Latency**: Time to render the complete UI interface
        - **End-to-End Loop Latency**: Time interval between consecutive orderbook message arrivals
        - **Target**: < 10ms processing time for high-frequency trading applications
        - **Current Status**: Processing faster than stream reception (real-time capable)
        """)
else:
    st.info("Connect to WebSocket to see performance metrics")

# Memory Profiling & Thread Management Analysis
st.divider()
st.header("🔬 Memory Profiling & Thread Management")

# Record system snapshot
snapshot = st.session_state.profiler.record_snapshot()

mem_col1, mem_col2, mem_col3 = st.columns(3)

with mem_col1:
    st.subheader("Memory Usage")
    memory = snapshot['memory']
    st.metric("RSS (Resident)", f"{memory['rss_mb']:.2f} MB")
    st.metric("VMS (Virtual)", f"{memory['vms_mb']:.2f} MB")
    st.metric("Memory %", f"{memory['percent']:.1f}%")
    st.metric("Available", f"{memory['available_mb']:.2f} MB")
    
    # Memory trend chart
    if len(st.session_state.profiler.memory_history) > 10:
        # Convert deque to list for slicing
        mem_history_list = list(st.session_state.profiler.memory_history)
        mem_data = [m['rss_mb'] for m in mem_history_list[-50:]]
        st.line_chart(mem_data)

with mem_col2:
    st.subheader("Thread Management")
    threads = snapshot['threads']
    st.metric("Active Threads", threads['active_threads'])
    st.metric("Daemon Threads", threads['daemon_threads'])
    st.metric("Main Thread", "✅ Alive" if threads['main_thread_alive'] else "❌ Dead")
    
    with st.expander("Thread Details"):
        for i, name in enumerate(threads['thread_names'], 1):
            st.write(f"{i}. {name}")

with mem_col3:
    st.subheader("CPU & System")
    st.metric("CPU Usage", f"{snapshot['cpu']:.1f}%")
    
    # System recommendations
    recommendations = st.session_state.profiler.get_optimization_recommendations()
    st.write("**Optimization Status:**")
    for rec in recommendations:
        st.write(rec)

# Optimization Techniques
with st.expander("💡 Advanced Optimization Techniques"):
    opt_tab1, opt_tab2, opt_tab3 = st.tabs(["Memory Management", "Network Optimization", "Model Optimization"])
    
    with opt_tab1:
        st.write("### Memory Management Strategies")
        st.write("""
        **Current Implementation:**
        - Orderbook history: Circular buffer (max 1000 entries)
        - Performance metrics: Rolling window (max 100 samples)
        - Session state management: Streamlit's built-in caching
        
        **Optimization Techniques:**
        - Using numpy arrays for numerical data (already implemented)
        - Limiting buffer sizes to prevent memory bloat
        - Clearing old data with .pop(0) instead of recreating lists
        - Memory-efficient data structures (deque could be faster for FIFO)
        """)
        
        if st.session_state.profiler.memory_history:
            mem_stats = pd.DataFrame({
                'Time': range(len(st.session_state.profiler.memory_history)),
                'RSS (MB)': [m['rss_mb'] for m in st.session_state.profiler.memory_history],
                'VMS (MB)': [m['vms_mb'] for m in st.session_state.profiler.memory_history]
            })
            st.write("**Memory Usage History:**")
            st.dataframe(mem_stats.tail(10), use_container_width=True, hide_index=True)
    
    with opt_tab2:
        st.write("### Network Communication Optimization")
        tips = st.session_state.profiler.get_network_optimization_tips()
        for tip in tips:
            st.write(f"• {tip}")
        
        st.write("\n**Current Implementation:**")
        st.write("- WebSocket connection with ping/pong keepalive")
        st.write("- Exponential backoff for reconnections")
        st.write("- Async message handling in separate thread")
        st.write("- JSON parsing with error handling")
    
    with opt_tab3:
        st.write("### Regression Model Efficiency")
        tips = st.session_state.profiler.get_model_optimization_tips()
        for tip in tips:
            st.write(f"• {tip}")
        
        st.write("\n**Current Implementation:**")
        st.write("- Linear Regression for slippage prediction")
        st.write("- Logistic Regression for maker/taker classification")
        st.write("- Feature extraction from orderbook data")
        st.write("- StandardScaler for feature normalization")
        st.write("- Lazy training (only when sufficient data available)")

# Historical Data Replay Mode
st.divider()
st.header("📹 Historical Data Replay & Backtesting")

replay_col1, replay_col2 = st.columns(2)

with replay_col1:
    st.subheader("Recording Controls")
    
    recording_stats = st.session_state.replay_engine.get_recording_stats()
    
    if st.session_state.replay_engine.is_recording:
        st.warning("🔴 Recording in progress...")
        st.metric("Snapshots Recorded", recording_stats['total_snapshots'])
        st.metric("Duration", f"{recording_stats['duration_seconds']:.1f}s")
        
        if st.button("⏹️ Stop Recording", use_container_width=True):
            st.session_state.replay_engine.stop_recording()
            st.rerun()
    else:
        st.info("⚪ Not recording")
        if recording_stats['total_snapshots'] > 0:
            st.metric("Last Recording", f"{recording_stats['total_snapshots']} snapshots")
            st.metric("Duration", f"{recording_stats['duration_seconds']:.1f}s")
        
        # Only allow recording if not replaying
        if st.session_state.is_connected and not st.session_state.replay_engine.is_replaying:
            if st.button("⏺️ Start Recording", type="primary", use_container_width=True):
                st.session_state.replay_engine.start_recording()
                st.rerun()
        elif st.session_state.replay_engine.is_replaying:
            st.info("Stop replay before recording")
    
    # Save/Load recording
    st.divider()
    
    if recording_stats['total_snapshots'] > 0:
        save_filename = st.text_input("Filename", "recording.json")
        if st.button("💾 Save Recording", use_container_width=True):
            if st.session_state.replay_engine.save_to_file(save_filename):
                st.success(f"Saved to {save_filename}")
            else:
                st.error("Failed to save")

with replay_col2:
    st.subheader("Replay Controls")
    
    load_filename = st.text_input("Load Filename", "recording.json")
    
    if st.button("📁 Load Recording", use_container_width=True):
        if st.session_state.replay_engine.load_from_file(load_filename):
            st.success(f"Loaded {st.session_state.replay_engine.get_recording_stats()['total_snapshots']} snapshots")
        else:
            st.error("Failed to load file")
    
    replay_speed = st.selectbox(
        "Replay Speed",
        [0.5, 1.0, 2.0, 5.0, 10.0],
        index=1,
        help="Speed multiplier for replay (1.0 = real-time)"
    )
    
    if st.session_state.replay_engine.is_replaying:
        st.success("▶️ Replay in progress...")
        if st.button("⏹️ Stop Replay", use_container_width=True):
            st.session_state.replay_engine.stop_replay()
            st.rerun()
    else:
        if recording_stats['total_snapshots'] > 0:
            # Only allow replay if not recording
            if st.session_state.replay_engine.is_recording:
                st.info("Stop recording before replaying")
            else:
                if st.button("▶️ Start Replay", type="primary", use_container_width=True):
                    # Disconnect live feed if connected
                    if st.session_state.is_connected:
                        if st.session_state.ws_client:
                            st.session_state.ws_client.close()
                        st.session_state.is_connected = False
                        st.session_state.ws_client = None
                    
                    # Start replay
                    st.session_state.replay_engine.start_replay(
                        callback=on_orderbook_update,
                        speed=replay_speed
                    )
                    st.rerun()
        else:
            st.info("Load a recording to replay")

# Backtesting info
with st.expander("ℹ️ About Backtesting Mode"):
    st.write("""
    ### How to Use Historical Replay:
    
    1. **Record Data**: Connect to live WebSocket and click "Start Recording" to capture orderbook snapshots
    2. **Save Recording**: Save your recorded data to a JSON file for later use
    3. **Load & Replay**: Load a saved recording and replay it at various speeds
    4. **Analyze**: All metrics (slippage, fees, market impact) are calculated during replay as if it were live
    
    ### Use Cases:
    - **Strategy Backtesting**: Test trading strategies against historical data
    - **Performance Analysis**: Analyze market behavior during specific periods
    - **Model Training**: Use historical data to train regression models
    - **Comparison**: Compare live vs historical performance
    
    ### Notes:
    - Replay disconnects live WebSocket feed automatically
    - Replay speed affects timing but not calculations
    - All performance metrics are captured during replay
    """)

# Export Results Section
st.divider()
st.header("💾 Export Results")

export_col1, export_col2 = st.columns(2)

with export_col1:
    st.subheader("Simulation Results")
    
    # Check if we have data to export
    has_metrics = st.session_state.metrics and st.session_state.orderbook_data
    
    if has_metrics:
        st.write("Export current simulation metrics and orderbook data")
        
        export_format = st.radio("Format", ["CSV", "JSON"], horizontal=True, key="sim_format")
        
        sim_filename = st.text_input("Filename", 
            f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
            key="sim_filename")
        
        if st.button("📥 Export Simulation Results", use_container_width=True):
            try:
                # Calculate spread metrics
                best_bid = float(st.session_state.orderbook_data.get('bids', [[0]])[0][0])
                best_ask = float(st.session_state.orderbook_data.get('asks', [[0]])[0][0])
                spread_usd = best_ask - best_bid
                mid_price = st.session_state.metrics.get('mid_price', (best_ask + best_bid) / 2)
                spread_pct = (spread_usd / mid_price * 100) if mid_price > 0 else 0
                
                orderbook_summary = {
                    'mid_price': mid_price,
                    'spread_usd': spread_usd,
                    'spread_pct': spread_pct,
                    'best_bid': best_bid,
                    'best_ask': best_ask
                }
                
                if export_format == "CSV":
                    filename = ExportManager.export_simulation_results_csv(
                        st.session_state.metrics,
                        orderbook_summary,
                        st.session_state.selected_exchange,
                        st.session_state.selected_symbol,
                        sim_filename
                    )
                else:  # JSON
                    # Prepare input parameters (only include existing keys)
                    input_params = {
                        'quantity_usd': st.session_state.get('quantity_usd', 0),
                        'order_type': st.session_state.get('order_type', 'market'),
                        'volatility': st.session_state.get('volatility', 0.02),
                        'fee_tier': st.session_state.get('fee_tier', 'VIP 0')
                    }
                    
                    filename = ExportManager.export_simulation_results_json(
                        st.session_state.metrics,
                        orderbook_summary,
                        st.session_state.selected_exchange,
                        st.session_state.selected_symbol,
                        input_params,
                        sim_filename
                    )
                
                st.success(f"✅ Exported to {filename}")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    else:
        st.info("Connect to WebSocket and calculate metrics to enable export")

with export_col2:
    st.subheader("Performance Report")
    
    # Check if we have performance data
    has_perf_data = (len(st.session_state.processing_times) > 0 or 
                     len(st.session_state.ui_update_times) > 0 or 
                     len(st.session_state.end_to_end_times) > 0)
    
    if has_perf_data:
        st.write("Export performance metrics and system analysis")
        
        perf_format = st.radio("Format", ["CSV", "JSON"], horizontal=True, key="perf_format")
        
        perf_filename = st.text_input("Filename", 
            f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{perf_format.lower()}",
            key="perf_filename")
        
        if st.button("📥 Export Performance Report", use_container_width=True):
            try:
                if perf_format == "CSV":
                    filename = ExportManager.export_performance_report_csv(
                        st.session_state.processing_times,
                        st.session_state.ui_update_times,
                        st.session_state.end_to_end_times,
                        perf_filename
                    )
                else:  # JSON
                    # Get memory and thread info if available
                    memory_info = None
                    thread_info = None
                    
                    if st.session_state.profiler.memory_history:
                        memory_info = {
                            'current': st.session_state.profiler.memory_history[-1],
                            'history_samples': len(st.session_state.profiler.memory_history)
                        }
                    
                    if st.session_state.profiler.thread_history:
                        thread_info = {
                            'current': st.session_state.profiler.thread_history[-1],
                            'history_samples': len(st.session_state.profiler.thread_history)
                        }
                    
                    filename = ExportManager.export_performance_report_json(
                        st.session_state.processing_times,
                        st.session_state.ui_update_times,
                        st.session_state.end_to_end_times,
                        memory_info,
                        thread_info,
                        perf_filename
                    )
                
                st.success(f"✅ Exported to {filename}")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    else:
        st.info("Run the simulator to collect performance data")

# Export info
with st.expander("ℹ️ About Export Formats"):
    st.write("""
    ### Simulation Results:
    
    **CSV Format:**
    - Single row with all calculated metrics
    - Ideal for spreadsheet analysis
    - Contains: timestamp, exchange, symbol, prices, slippage, fees, market impact
    
    **JSON Format:**
    - Complete simulation state with metadata
    - Includes input parameters and orderbook summary
    - Structured format for programmatic analysis
    
    ### Performance Reports:
    
    **CSV Format:**
    - Statistical summary table
    - Metrics: mean, P50, P95, P99, min, max
    - Covers: data processing, UI update, end-to-end latency
    
    **JSON Format:**
    - Comprehensive performance data
    - Includes memory and thread profiling
    - Complete latency statistics with metadata
    - Suitable for automated analysis pipelines
    """)

# Track UI render time
ui_render_time = (time.perf_counter() - ui_render_start) * 1000
st.session_state.ui_update_times.append(ui_render_time)
if len(st.session_state.ui_update_times) > 100:
    st.session_state.ui_update_times.pop(0)

# Auto-refresh for real-time updates (live or replay)
if st.session_state.is_connected or st.session_state.replay_engine.is_replaying:
    time.sleep(0.1)  # Small delay to prevent too frequent updates
    st.rerun()

# Add UI render tracking here, at the very bottom
if st.session_state.ui_render_start is not None:
    ui_render_end = time.perf_counter()
    ui_latency = (ui_render_end - st.session_state.ui_render_start) * 1000
    st.session_state.ui_update_times.append(ui_latency)
    if len(st.session_state.ui_update_times) > 100:
        st.session_state.ui_update_times.pop(0)
    st.session_state.ui_render_start = None