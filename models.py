import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlippageModel:
    """Linear regression model for slippage estimation"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the slippage model
        
        Args:
            X: Feature matrix (orderbook features)
            y: Target variable (actual slippage values)
        """
        if len(X) < 10:  # Need minimum samples
            return
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            logger.info("Slippage model trained successfully")
        except Exception as e:
            logger.error(f"Error training slippage model: {e}")
    
    def predict(self, X: np.ndarray) -> float:
        """
        Predict slippage
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted slippage value
        """
        if not self.is_trained:
            # Return a simple estimate if not trained
            return X[0] * 0.001  # 0.1% of quantity as baseline
        
        try:
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            return max(0, self.model.predict(X_scaled)[0])
        except Exception as e:
            logger.error(f"Error predicting slippage: {e}")
            return X[0] * 0.001

class MakerTakerModel:
    """Logistic regression model for maker/taker proportion prediction"""
    
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the maker/taker model
        
        Args:
            X: Feature matrix (orderbook features)
            y: Target variable (0 for taker, 1 for maker)
        """
        if len(X) < 10:
            return
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            logger.info("Maker/Taker model trained successfully")
        except Exception as e:
            logger.error(f"Error training maker/taker model: {e}")
    
    def predict_proba(self, X: np.ndarray) -> float:
        """
        Predict probability of being a maker
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability of being a maker (0-1)
        """
        if not self.is_trained:
            # Return default estimate based on market order assumption
            # Market orders are typically takers, so return low maker probability
            return 0.1
        
        try:
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            # Return probability of class 1 (maker)
            return self.model.predict_proba(X_scaled)[0][1]
        except Exception as e:
            logger.error(f"Error predicting maker/taker: {e}")
            return 0.1

class AlmgrenChrissModel:
    """
    Almgren-Chriss model for optimal execution and market impact estimation
    
    Reference: https://www.linkedin.com/pulse/understanding-almgren-chriss-model-optimal-portfolio-execution-pal-pmeqc/
    """
    
    @staticmethod
    def calculate_temporary_impact(
        quantity: float,
        volatility: float,
        gamma: float = 0.1
    ) -> float:
        """
        Calculate temporary market impact
        
        Args:
            quantity: Trade quantity
            volatility: Market volatility
            gamma: Temporary impact coefficient
            
        Returns:
            Temporary market impact cost
        """
        return gamma * volatility * quantity
    
    @staticmethod
    def calculate_permanent_impact(
        quantity: float,
        mid_price: float,
        eta: float = 0.01
    ) -> float:
        """
        Calculate permanent market impact
        
        Args:
            quantity: Trade quantity
            mid_price: Current mid price
            eta: Permanent impact coefficient
            
        Returns:
            Permanent market impact cost
        """
        return eta * mid_price * (quantity ** 2)
    
    @staticmethod
    def calculate_total_impact(
        quantity: float,
        volatility: float,
        mid_price: float,
        orderbook_depth: int,
        gamma: float = 0.1,
        eta: float = 0.01
    ) -> float:
        """
        Calculate total market impact using Almgren-Chriss model
        
        Args:
            quantity: Trade quantity
            volatility: Market volatility
            mid_price: Current mid price
            orderbook_depth: Number of price levels in orderbook
            gamma: Temporary impact coefficient
            eta: Permanent impact coefficient
            
        Returns:
            Total market impact cost in USD
        """
        # Adjust coefficients based on orderbook depth
        # Deeper orderbook = lower impact
        depth_factor = 1 / np.log(max(orderbook_depth, 2))
        adjusted_gamma = gamma * depth_factor
        adjusted_eta = eta * depth_factor
        
        # Calculate components
        temporary = AlmgrenChrissModel.calculate_temporary_impact(
            quantity, volatility, adjusted_gamma
        )
        permanent = AlmgrenChrissModel.calculate_permanent_impact(
            quantity, mid_price, adjusted_eta
        )
        
        # Total impact in USD
        total_impact = (temporary + permanent) * mid_price
        
        return total_impact
