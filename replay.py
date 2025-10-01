import json
import time
from typing import List, Dict, Callable
from datetime import datetime
import threading

class HistoricalReplayEngine:
    """Engine for replaying historical orderbook data"""
    
    def __init__(self):
        self.recorded_data: List[Dict] = []
        self.is_recording = False
        self.is_replaying = False
        self.replay_thread = None
        self.replay_speed = 1.0  # 1.0 = real-time, 2.0 = 2x speed
        
    def start_recording(self):
        """Start recording orderbook data"""
        self.is_recording = True
        self.recorded_data = []
        
    def stop_recording(self):
        """Stop recording orderbook data"""
        self.is_recording = False
        
    def record_orderbook(self, orderbook_data: Dict):
        """Record a single orderbook snapshot"""
        if self.is_recording:
            snapshot = {
                'timestamp': time.time(),
                'data': orderbook_data
            }
            self.recorded_data.append(snapshot)
    
    def save_to_file(self, filename: str):
        """Save recorded data to JSON file"""
        if not self.recorded_data:
            return False
        
        data_to_save = {
            'recording_start': datetime.fromtimestamp(self.recorded_data[0]['timestamp']).isoformat(),
            'recording_end': datetime.fromtimestamp(self.recorded_data[-1]['timestamp']).isoformat(),
            'total_snapshots': len(self.recorded_data),
            'snapshots': self.recorded_data
        }
        
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        return True
    
    def load_from_file(self, filename: str) -> bool:
        """Load recorded data from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.recorded_data = data.get('snapshots', [])
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def start_replay(self, callback: Callable, speed: float = 1.0):
        """
        Start replaying recorded data
        
        Args:
            callback: Function to call with each orderbook snapshot
            speed: Replay speed multiplier (1.0 = real-time)
        """
        if not self.recorded_data:
            return False
        
        if self.is_replaying:
            return False
        
        self.replay_speed = speed
        self.is_replaying = True
        
        def replay_worker():
            # Work on a copy to avoid mutation during iteration
            data_copy = list(self.recorded_data)
            
            for i, snapshot in enumerate(data_copy):
                if not self.is_replaying:
                    break
                
                # Calculate delay based on original timing
                if i > 0:
                    time_diff = snapshot['timestamp'] - data_copy[i-1]['timestamp']
                    delay = time_diff / self.replay_speed
                    time.sleep(max(0, delay))
                
                # Call the callback with the orderbook data
                callback(snapshot['data'])
            
            self.is_replaying = False
        
        self.replay_thread = threading.Thread(target=replay_worker, daemon=True)
        self.replay_thread.start()
        
        return True
    
    def stop_replay(self):
        """Stop the replay"""
        self.is_replaying = False
        if self.replay_thread:
            self.replay_thread.join(timeout=1)
    
    def get_recording_stats(self) -> Dict:
        """Get statistics about the current recording"""
        if not self.recorded_data:
            return {
                'total_snapshots': 0,
                'duration_seconds': 0,
                'start_time': None,
                'end_time': None
            }
        
        start_ts = self.recorded_data[0]['timestamp']
        end_ts = self.recorded_data[-1]['timestamp']
        
        return {
            'total_snapshots': len(self.recorded_data),
            'duration_seconds': end_ts - start_ts,
            'start_time': datetime.fromtimestamp(start_ts).isoformat(),
            'end_time': datetime.fromtimestamp(end_ts).isoformat()
        }
