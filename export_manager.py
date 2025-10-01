import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any

class ExportManager:
    """Manages export of simulation results and performance reports"""
    
    @staticmethod
    def _sanitize_filename(filename: str, default_ext: str) -> str:
        """Sanitize filename and ensure it has the correct extension"""
        # Remove any path separators to prevent path traversal
        filename = os.path.basename(filename)
        
        # Ensure correct extension
        if not filename.endswith(f'.{default_ext}'):
            # Remove any existing extension and add the correct one
            filename = os.path.splitext(filename)[0] + f'.{default_ext}'
        
        return filename
    
    @staticmethod
    def export_simulation_results_csv(metrics: Dict[str, Any], orderbook_summary: Dict[str, Any], 
                                       exchange: str, symbol: str, filename: str = None) -> str:
        """Export current simulation results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.csv"
        else:
            filename = ExportManager._sanitize_filename(filename, 'csv')
        
        # Calculate derived metrics
        mid_price = metrics.get('mid_price', 0)
        slippage_pct = (metrics.get('slippage', 0) / mid_price * 100) if mid_price > 0 else 0
        fee_pct = (metrics.get('fees', 0) / mid_price * 100) if mid_price > 0 else 0
        impact_pct = (metrics.get('market_impact', 0) / mid_price * 100) if mid_price > 0 else 0
        
        # Prepare data for export
        data = {
            'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'Exchange': [exchange],
            'Symbol': [symbol],
            'Mid_Price_USD': [mid_price],
            'Quantity_Crypto': [metrics.get('quantity_crypto', 0)],
            'Best_Bid': [orderbook_summary.get('best_bid', 0)],
            'Best_Ask': [orderbook_summary.get('best_ask', 0)],
            'Spread_USD': [orderbook_summary.get('spread_usd', 0)],
            'Spread_Percent': [orderbook_summary.get('spread_pct', 0)],
            'Slippage_USD': [metrics.get('slippage', 0)],
            'Slippage_Percent': [slippage_pct],
            'Exchange_Fee_USD': [metrics.get('fees', 0)],
            'Fee_Percent': [fee_pct],
            'Market_Impact_USD': [metrics.get('market_impact', 0)],
            'Market_Impact_Percent': [impact_pct],
            'Maker_Ratio': [metrics.get('maker_ratio', 0)],
            'Taker_Ratio': [metrics.get('taker_ratio', 0)],
            'Total_Cost_USD': [metrics.get('net_cost', 0)],
            'Internal_Latency_MS': [metrics.get('internal_latency', 0)]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return filename
    
    @staticmethod
    def export_simulation_results_json(metrics: Dict[str, Any], orderbook_summary: Dict[str, Any], 
                                        exchange: str, symbol: str, input_params: Dict[str, Any],
                                        filename: str = None) -> str:
        """Export current simulation results to JSON with full details"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.json"
        else:
            filename = ExportManager._sanitize_filename(filename, 'json')
        
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'exchange': exchange,
                'symbol': symbol
            },
            'input_parameters': input_params,
            'orderbook_summary': orderbook_summary,
            'calculated_metrics': metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename
    
    @staticmethod
    def export_performance_report_csv(processing_times: List[float], ui_update_times: List[float], 
                                       end_to_end_times: List[float], filename: str = None) -> str:
        """Export performance metrics to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.csv"
        else:
            filename = ExportManager._sanitize_filename(filename, 'csv')
        
        # Calculate statistics
        def calc_stats(data):
            if not data:
                return {'mean': 0, 'p50': 0, 'p95': 0, 'p99': 0, 'min': 0, 'max': 0}
            return {
                'mean': sum(data) / len(data),
                'p50': sorted(data)[int(len(data) * 0.5)] if data else 0,
                'p95': sorted(data)[int(len(data) * 0.95)] if data else 0,
                'p99': sorted(data)[int(len(data) * 0.99)] if data else 0,
                'min': min(data),
                'max': max(data)
            }
        
        proc_stats = calc_stats(processing_times)
        ui_stats = calc_stats(ui_update_times)
        e2e_stats = calc_stats(end_to_end_times)
        
        data = {
            'Metric': ['Data Processing (ms)', 'UI Update (ms)', 'End-to-End Loop (ms)'],
            'Mean': [proc_stats['mean'], ui_stats['mean'], e2e_stats['mean']],
            'P50': [proc_stats['p50'], ui_stats['p50'], e2e_stats['p50']],
            'P95': [proc_stats['p95'], ui_stats['p95'], e2e_stats['p95']],
            'P99': [proc_stats['p99'], ui_stats['p99'], e2e_stats['p99']],
            'Min': [proc_stats['min'], ui_stats['min'], e2e_stats['min']],
            'Max': [proc_stats['max'], ui_stats['max'], e2e_stats['max']],
            'Samples': [len(processing_times), len(ui_update_times), len(end_to_end_times)]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return filename
    
    @staticmethod
    def export_performance_report_json(processing_times: List[float], ui_update_times: List[float], 
                                        end_to_end_times: List[float], memory_info: Dict[str, Any] = None,
                                        thread_info: Dict[str, Any] = None, filename: str = None) -> str:
        """Export comprehensive performance report to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        else:
            filename = ExportManager._sanitize_filename(filename, 'json')
        
        # Calculate statistics
        def calc_stats(data):
            if not data:
                return {'mean': 0, 'p50': 0, 'p95': 0, 'p99': 0, 'min': 0, 'max': 0}
            return {
                'mean': sum(data) / len(data),
                'p50': sorted(data)[int(len(data) * 0.5)] if data else 0,
                'p95': sorted(data)[int(len(data) * 0.95)] if data else 0,
                'p99': sorted(data)[int(len(data) * 0.99)] if data else 0,
                'min': min(data),
                'max': max(data)
            }
        
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'report_type': 'performance_analysis'
            },
            'latency_metrics': {
                'data_processing_ms': calc_stats(processing_times),
                'ui_update_ms': calc_stats(ui_update_times),
                'end_to_end_loop_ms': calc_stats(end_to_end_times)
            },
            'sample_counts': {
                'processing_samples': len(processing_times),
                'ui_samples': len(ui_update_times),
                'e2e_samples': len(end_to_end_times)
            }
        }
        
        if memory_info:
            export_data['memory_profile'] = memory_info
        
        if thread_info:
            export_data['thread_info'] = thread_info
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename
    
    @staticmethod
    def export_historical_results_csv(recorded_data: List[Dict], metrics_history: List[Dict], 
                                       filename: str = None) -> str:
        """Export historical replay/backtesting results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.csv"
        else:
            filename = ExportManager._sanitize_filename(filename, 'csv')
        
        if not metrics_history:
            return None
        
        # Convert metrics history to DataFrame
        df = pd.DataFrame(metrics_history)
        df.to_csv(filename, index=False)
        return filename
