"""
Behavioral rules for fraud detection.
"""

def velocity_rule(tx_count: int, threshold: int = 5) -> bool:
    """
    Velocity rule: Flag as high risk if transaction count exceeds threshold.
    
    Args:
        tx_count: Number of transactions in a time window
        threshold: Maximum allowed transactions before flagging (default: 5)
    
    Returns:
        True if velocity is suspicious, False otherwise
    """
    return tx_count > threshold


def drift_rule(current_amount: float, avg_amount: float, tolerance: float = 2.0) -> bool:
    """
    Drift rule: Flag as high risk if transaction amount deviates significantly from user's average.
    
    Args:
        current_amount: Current transaction amount
        avg_amount: User's average transaction amount
        tolerance: Multiplier for acceptable deviation (default: 2.0x average)
    
    Returns:
        True if amount deviates too much from average, False otherwise
    """
    if avg_amount == 0:
        return False
    
    return current_amount > (avg_amount * tolerance)
