from typing import Dict

class FeeCalculator:
    """Calculator for exchange trading fees"""
    
    # OKX fee tiers (maker/taker fees in percentage)
    OKX_FEE_TIERS = {
        'VIP 0': {'maker': 0.0008, 'taker': 0.0010},
        'VIP 1': {'maker': 0.0006, 'taker': 0.0009},
        'VIP 2': {'maker': 0.0005, 'taker': 0.0008},
        'VIP 3': {'maker': 0.0004, 'taker': 0.0007},
        'VIP 4': {'maker': 0.0003, 'taker': 0.0006},
        'VIP 5': {'maker': 0.0002, 'taker': 0.0005},
    }
    
    # Binance fee tiers
    BINANCE_FEE_TIERS = {
        'Regular': {'maker': 0.0010, 'taker': 0.0010},
        'VIP 1': {'maker': 0.0009, 'taker': 0.0010},
        'VIP 2': {'maker': 0.0008, 'taker': 0.0009},
        'VIP 3': {'maker': 0.0006, 'taker': 0.0008},
    }
    
    # Coinbase fee tiers
    COINBASE_FEE_TIERS = {
        'Taker': {'maker': 0.0040, 'taker': 0.0060},
        'Advanced': {'maker': 0.0025, 'taker': 0.0040},
    }
    
    # Kraken fee tiers
    KRAKEN_FEE_TIERS = {
        'Starter': {'maker': 0.0016, 'taker': 0.0026},
        'Intermediate': {'maker': 0.0014, 'taker': 0.0024},
        'Pro': {'maker': 0.0012, 'taker': 0.0022},
    }
    
    @staticmethod
    def get_fee_tiers_for_exchange(exchange: str) -> list:
        """Get available fee tiers for a given exchange"""
        tier_map = {
            'OKX': list(FeeCalculator.OKX_FEE_TIERS.keys()),
            'Binance': list(FeeCalculator.BINANCE_FEE_TIERS.keys()),
            'Coinbase': list(FeeCalculator.COINBASE_FEE_TIERS.keys()),
            'Kraken': list(FeeCalculator.KRAKEN_FEE_TIERS.keys()),
        }
        return tier_map.get(exchange, ['Standard'])
    
    def calculate_fees(
        self,
        exchange: str,
        quantity_usd: float,
        fee_tier: str = 'VIP 0',
        order_type: str = 'market',
        maker_ratio: float = 0.0
    ) -> float:
        """
        Calculate trading fees based on exchange and fee tier
        
        Args:
            exchange: Exchange name (e.g., 'OKX')
            quantity_usd: Trade quantity in USD
            fee_tier: Fee tier level
            order_type: 'market' or 'limit'
            maker_ratio: Proportion of order that will be maker (0-1)
            
        Returns:
            Total fee in USD
        """
        tier_maps = {
            'OKX': self.OKX_FEE_TIERS,
            'Binance': self.BINANCE_FEE_TIERS,
            'Coinbase': self.COINBASE_FEE_TIERS,
            'Kraken': self.KRAKEN_FEE_TIERS,
        }
        
        tiers = tier_maps.get(exchange, {})
        default_tier = list(tiers.keys())[0] if tiers else 'Standard'
        fee_structure = tiers.get(fee_tier, tiers.get(default_tier, {'maker': 0.001, 'taker': 0.001}))
        
        # Market orders are always takers
        if order_type == 'market':
            taker_fee = quantity_usd * fee_structure['taker']
            return taker_fee
        
        # For limit orders, use maker/taker ratio
        maker_fee = quantity_usd * maker_ratio * fee_structure['maker']
        taker_fee = quantity_usd * (1 - maker_ratio) * fee_structure['taker']
        
        return maker_fee + taker_fee
    
    def get_fee_structure(self, exchange: str, fee_tier: str) -> Dict[str, float]:
        """
        Get fee structure for a given exchange and tier
        
        Args:
            exchange: Exchange name
            fee_tier: Fee tier level
            
        Returns:
            Dictionary with maker and taker fees
        """
        tier_maps = {
            'OKX': self.OKX_FEE_TIERS,
            'Binance': self.BINANCE_FEE_TIERS,
            'Coinbase': self.COINBASE_FEE_TIERS,
            'Kraken': self.KRAKEN_FEE_TIERS,
        }
        
        tiers = tier_maps.get(exchange, {})
        default_tier = list(tiers.keys())[0] if tiers else 'Standard'
        return tiers.get(fee_tier, tiers.get(default_tier, {'maker': 0.001, 'taker': 0.001}))
