import numpy as np
from typing import List, Dict, Tuple
from models import SlippageModel, MakerTakerModel, AlmgrenChrissModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingCalculator:
    """Main calculator for all trading metrics"""
    
    def __init__(self):
        self.slippage_model = SlippageModel()
        self.maker_taker_model = MakerTakerModel()
        self.almgren_chriss = AlmgrenChrissModel()
        
        # Storage for training data
        self.slippage_features = []
        self.slippage_targets = []
        self.maker_taker_features = []
        self.maker_taker_targets = []
    
    def extract_orderbook_features(
        self, 
        orderbook: Dict,
        quantity: float,
        side: str = 'buy'
    ) -> np.ndarray:
        """
        Extract features from orderbook for ML models
        
        Args:
            orderbook: Orderbook data
            quantity: Trade quantity
            side: 'buy' or 'sell'
            
        Returns:
            Feature array
        """
        asks = orderbook.get('asks', [])
        bids = orderbook.get('bids', [])
        
        if not asks or not bids:
            return np.array([quantity, 0, 0, 0, 0])
        
        # Parse prices and quantities
        ask_prices = [float(a[0]) for a in asks[:10]]
        ask_qtys = [float(a[1]) for a in asks[:10]]
        bid_prices = [float(b[0]) for b in bids[:10]]
        bid_qtys = [float(b[1]) for b in bids[:10]]
        
        # Calculate features
        mid_price = (ask_prices[0] + bid_prices[0]) / 2
        spread = ask_prices[0] - bid_prices[0]
        spread_bps = (spread / mid_price) * 10000
        
        # Order imbalance
        total_ask_qty = sum(ask_qtys)
        total_bid_qty = sum(bid_qtys)
        imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty) if (total_bid_qty + total_ask_qty) > 0 else 0
        
        # Depth
        depth = len(asks) + len(bids)
        
        features = np.array([
            quantity,
            spread_bps,
            imbalance,
            depth,
            mid_price
        ])
        
        return features
    
    def calculate_actual_slippage(
        self,
        orderbook: Dict,
        quantity: float,
        side: str = 'buy'
    ) -> float:
        """
        Calculate actual slippage by simulating order execution
        
        Args:
            orderbook: Current orderbook
            quantity: Trade quantity
            side: 'buy' or 'sell'
            
        Returns:
            Slippage in USD
        """
        levels = orderbook.get('asks' if side == 'buy' else 'bids', [])
        
        if not levels:
            return 0
        
        # Parse levels
        parsed_levels = [(float(price), float(qty)) for price, qty in levels]
        
        # Calculate mid price
        asks = orderbook.get('asks', [])
        bids = orderbook.get('bids', [])
        if asks and bids:
            mid_price = (float(asks[0][0]) + float(bids[0][0])) / 2
        else:
            mid_price = parsed_levels[0][0]
        
        # Simulate execution
        remaining_qty = quantity
        total_cost = 0
        
        for price, available_qty in parsed_levels:
            if remaining_qty <= 0:
                break
            
            executed_qty = min(remaining_qty, available_qty)
            total_cost += executed_qty * price
            remaining_qty -= executed_qty
        
        if quantity > 0:
            avg_price = total_cost / quantity
            slippage = abs(avg_price - mid_price) * quantity
        else:
            slippage = 0
        
        return slippage
    
    def calculate_slippage(
        self,
        orderbook_history: List[Dict],
        quantity: float,
        side: str = 'buy'
    ) -> float:
        """
        Calculate expected slippage using regression model
        
        Args:
            orderbook_history: Historical orderbook data
            quantity: Trade quantity
            side: 'buy' or 'sell'
            
        Returns:
            Expected slippage in USD
        """
        if not orderbook_history:
            # Fallback: simple estimate
            return quantity * 0.001
        
        current_orderbook = orderbook_history[-1]
        
        # Extract features for current orderbook
        features = self.extract_orderbook_features(current_orderbook, quantity, side)
        
        # Train model if we have enough historical data
        if len(orderbook_history) >= 20 and len(self.slippage_features) < 100:
            # Generate training data from history
            for ob in orderbook_history[-100:]:
                feat = self.extract_orderbook_features(ob, quantity, side)
                actual_slip = self.calculate_actual_slippage(ob, quantity, side)
                
                self.slippage_features.append(feat)
                self.slippage_targets.append(actual_slip)
            
            # Train model
            if len(self.slippage_features) >= 20:
                X = np.array(self.slippage_features)
                y = np.array(self.slippage_targets)
                self.slippage_model.train(X, y)
        
        # Predict slippage
        return self.slippage_model.predict(features)
    
    def calculate_market_impact_almgren_chriss(
        self,
        quantity: float,
        volatility: float,
        mid_price: float,
        orderbook_depth: int
    ) -> float:
        """
        Calculate market impact using Almgren-Chriss model
        
        Args:
            quantity: Trade quantity (in crypto)
            volatility: Market volatility
            mid_price: Current mid price
            orderbook_depth: Depth of orderbook
            
        Returns:
            Market impact cost in USD
        """
        return self.almgren_chriss.calculate_total_impact(
            quantity=quantity,
            volatility=volatility,
            mid_price=mid_price,
            orderbook_depth=orderbook_depth
        )
    
    def predict_maker_taker_proportion(
        self,
        orderbook_history: List[Dict],
        quantity: float
    ) -> float:
        """
        Predict maker proportion using logistic regression
        
        Args:
            orderbook_history: Historical orderbook data
            quantity: Trade quantity
            
        Returns:
            Probability of being a maker (0-1)
        """
        if not orderbook_history:
            return 0.1  # Default: mostly taker for market orders
        
        current_orderbook = orderbook_history[-1]
        features = self.extract_orderbook_features(current_orderbook, quantity)
        
        # Train model if we have enough data
        if len(orderbook_history) >= 20 and len(self.maker_taker_features) < 100:
            for ob in orderbook_history[-100:]:
                feat = self.extract_orderbook_features(ob, quantity)
                
                # Simulate: larger spreads and smaller quantities more likely to be maker
                spread = 0 if not ob.get('asks') or not ob.get('bids') else (
                    float(ob['asks'][0][0]) - float(ob['bids'][0][0])
                )
                is_maker = 1 if spread > 0.01 and quantity < 10 else 0
                
                self.maker_taker_features.append(feat)
                self.maker_taker_targets.append(is_maker)
            
            # Train model
            if len(self.maker_taker_features) >= 20:
                X = np.array(self.maker_taker_features)
                y = np.array(self.maker_taker_targets)
                self.maker_taker_model.train(X, y)
        
        # Predict maker probability
        return self.maker_taker_model.predict_proba(features)
