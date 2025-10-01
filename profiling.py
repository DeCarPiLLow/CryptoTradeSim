import psutil
import threading
import time
from typing import Dict, List
import os
from collections import deque

class SystemProfiler:
    """System profiler for memory and thread management analysis"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.memory_history = deque(maxlen=100)
        self.thread_history = deque(maxlen=100)
        self.last_snapshot_time = 0
        self.last_cpu = 0
        self.snapshot_interval = 1.0  # Take snapshot every 1 second
        # Prime CPU sampling for non-blocking calls
        self.process.cpu_percent(interval=None)
        
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics"""
        mem_info = self.process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size in MB
            'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def get_thread_info(self) -> Dict:
        """Get current thread information"""
        threads = threading.enumerate()
        
        return {
            'active_threads': len(threads),
            'thread_names': [t.name for t in threads],
            'daemon_threads': sum(1 for t in threads if t.daemon),
            'main_thread_alive': threading.main_thread().is_alive()
        }
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage (non-blocking)"""
        return self.process.cpu_percent(interval=None)
    
    def record_snapshot(self, force=False):
        """
        Record a snapshot of system metrics (throttled to 1 Hz)
        
        Args:
            force: Force snapshot even if within throttle interval
        """
        current_time = time.time()
        
        # Throttle snapshots to prevent performance impact
        if not force and (current_time - self.last_snapshot_time) < self.snapshot_interval:
            # Return last snapshot if available (fully cached)
            if self.memory_history and self.thread_history:
                return {
                    'timestamp': self.last_snapshot_time,
                    'memory': self.memory_history[-1],
                    'threads': self.thread_history[-1],
                    'cpu': self.last_cpu
                }
            # First call, proceed
        
        memory = self.get_memory_usage()
        threads = self.get_thread_info()
        cpu = self.get_cpu_usage()
        
        snapshot = {
            'timestamp': current_time,
            'memory': memory,
            'threads': threads,
            'cpu': cpu
        }
        
        self.memory_history.append(memory)
        self.thread_history.append(threads)
        self.last_snapshot_time = current_time
        self.last_cpu = cpu
        
        return snapshot
    
    # def get_optimization_recommendations(self) -> List[str]:
    #     """Generate optimization recommendations based on current metrics"""
    #     recommendations = []
        
    #     if not self.memory_history:
    #         return recommendations
        
    #     # Analyze memory usage
    #     current_mem = self.memory_history[-1]
    #     if current_mem['percent'] > 50:
    #         recommendations.append("⚠️ High memory usage detected. Consider reducing orderbook history buffer size.")
        
    #     if len(self.memory_history) > 10:
    #         mem_trend = [m['rss_mb'] for m in self.memory_history[-10:]]
    #         if mem_trend[-1] > mem_trend[0] * 1.2:
    #             recommendations.append("📈 Memory usage growing. Check for memory leaks in data structures.")
        
    #     # Analyze thread usage
    #     if self.thread_history:
    #         current_threads = self.thread_history[-1]
    #         if current_threads['active_threads'] > 5:
    #             recommendations.append("🧵 Multiple threads detected. Ensure proper thread synchronization.")
            
    #         if current_threads['daemon_threads'] > 2:
    #             recommendations.append("⚡ Multiple daemon threads running. Monitor for zombie threads.")
        
    #     if not recommendations:
    #         recommendations.append("✅ System resources are optimally utilized.")
        
    #     return recommendations

    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on current metrics"""
        recommendations = []
        
        if not self.memory_history:
            return recommendations
        
        # Analyze memory usage
        current_mem = self.memory_history[-1]
        if current_mem['percent'] > 50:
            recommendations.append(
                "⚠️ High memory usage detected. Consider reducing orderbook history buffer size."
            )
        
        # Convert deque → list before slicing
        if len(self.memory_history) > 10:
            mem_trend = [m['rss_mb'] for m in list(self.memory_history)[-10:]]
            if mem_trend[-1] > mem_trend[0] * 1.2:
                recommendations.append(
                    "📈 Memory usage growing. Check for memory leaks in data structures."
                )
        
        # Analyze thread usage
        if self.thread_history:
            current_threads = self.thread_history[-1]
            if current_threads['active_threads'] > 5:
                recommendations.append(
                    "🧵 Multiple threads detected. Ensure proper thread synchronization."
                )
            
            if current_threads['daemon_threads'] > 2:
                recommendations.append(
                    "⚡ Multiple daemon threads running. Monitor for zombie threads."
                )
        
        if not recommendations:
            recommendations.append("✅ System resources are optimally utilized.")
        
        return recommendations
    
    def get_network_optimization_tips(self) -> List[str]:
        """Network and data structure optimization tips"""
        return [
            "Use binary message formats (e.g., MessagePack) instead of JSON for faster parsing",
            "Implement connection pooling for WebSocket reconnections",
            "Use numpy arrays instead of Python lists for numerical data",
            "Enable compression on WebSocket messages if supported",
            "Batch orderbook updates if processing multiple symbols"
        ]
    
    def get_model_optimization_tips(self) -> List[str]:
        """Machine learning model optimization tips"""
        return [
            "Use incremental/online learning for real-time model updates",
            "Cache model predictions for repeated inputs",
            "Use lower precision (float32) instead of float64 for faster computation",
            "Implement feature caching to avoid redundant calculations",
            "Consider using LightGBM or XGBoost for faster tree-based models"
        ]
