"""Stock price analyzer."""

def calculate_returns(prices):
    """Calculate daily returns from price list."""
    returns = []
    for i in range(1, len(prices)):
        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(daily_return)
    return returns

def calculate_volatility(returns):
    """Calculate volatility (standard deviation of returns)."""
    # BUG: This doesn't handle empty lists!
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return variance ** 0.5

def main():
    # Sample AAPL prices
    prices = [150.0, 152.5, 151.0, 155.0, 153.5]
    returns = calculate_returns(prices)
    vol = calculate_volatility(returns)
    print(f"Returns: {returns}")
    print(f"Volatility: {vol:.4f}")

if __name__ == "__main__":
    main()
