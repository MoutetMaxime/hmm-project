import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# steps
N = 100
types = {
    "Done(buy)": 1,
    "Done(sell)": 2,
    "Traded Away (buy)": 3,
    "Traded Away (sell)": 4,
    # "D2D Trade": 5,
}
sigma = 1e-4
sigma_bid_ask = 0.83
mu = -9.79
mu2 = -9.87
mus_bid_ask = [mu, mu2]

sigmas_epsilons = [5e-2*0.79e-4*2, 5e-2*0.73e-4*2]

def generate_correlated_random_walks(initial_prices, volatilities, correlation, steps):
    """Generate two correlated random walk price series
    
    Args:
        initial_prices: List of initial prices for both bonds
        volatilities: List of volatilities for both bonds
        correlation: Correlation coefficient between the two bonds
        steps: Number of time steps
    """
    # Create correlation matrix
    corr_matrix = np.array([[1, correlation], 
                           [correlation, 1]])
    
    # Generate correlated normal random variables
    mean = [0, 0]
    random_walks = np.random.multivariate_normal(mean, corr_matrix, steps)
    
    # Initialize price series
    prices_1 = [initial_prices[0]]
    prices_2 = [initial_prices[1]]
    
    # Generate correlated price walks
    for i in range(steps):
        prices_1.append(prices_1[-1] * (1 + volatilities[0] * random_walks[i,0]))
        prices_2.append(prices_2[-1] * (1 + volatilities[1] * random_walks[i,1]))
        
    return prices_1, prices_2


initial_prices = [0.72, 0.74]  # Initial prices for both bonds
volatilities = [0.5e-4, 0.62e-4]  # Different volatilities for each bond
correlation = 0.843  # Correlation coefficient between bonds

price_walk_1, price_walk_2 = generate_correlated_random_walks(
    initial_prices, volatilities, correlation, N
)


def generate_buy_sell_prices(price_walk, bid_ask_spreads, sigma=sigma):
    buy_sell_prices = pd.DataFrame(price_walk)
    buy_sell_prices["buy_price"] = buy_sell_prices[0] + bid_ask_spreads[0]
    buy_sell_prices["sell_price"] = buy_sell_prices[0] - bid_ask_spreads[0]
    # Adding gaussian noise to the buy and sell prices
    buy_sell_prices["buy_price"] = buy_sell_prices["buy_price"] + np.abs(
        np.random.normal(0, sigma, len(buy_sell_prices))
    )
    buy_sell_prices["sell_price"] = buy_sell_prices["sell_price"] - np.abs(
        np.random.normal(0, sigma, len(buy_sell_prices))
    )
    buy_sell_prices.columns = ["price", "buy_price", "sell_price"]
    buy_sell_prices["time"] = range(len(buy_sell_prices))
    return buy_sell_prices


def generate_bid_ask_prices(steps, initial_bid_ask_spread=0, sigma=sigma_bid_ask, mu=mu):
    spreads = [initial_bid_ask_spread]
    for i in range(1, steps):
        spread = np.exp(np.random.normal(mu, sigma))
        spreads.append(spread)

    return pd.DataFrame(spreads)


def generate_trades(buy_sell_prices_1, buy_sell_prices_2, types, prob_trade=0.6):
    """Generate trades for two bonds ensuring they don't trade at the same time"""
    trades = []
    for i in range(len(buy_sell_prices_1)):
        is_traded = np.random.choice([True, False], p=[prob_trade, 1 - prob_trade])
        if is_traded:
            # Randomly choose which bond trades
            bond_index = np.random.choice([1, 2])
            trade_type = np.random.choice(list(types.values()))
            
            # Get price based on which bond was selected
            buy_sell_prices = buy_sell_prices_1 if bond_index == 1 else buy_sell_prices_2
            price = (
                buy_sell_prices.loc[i, "buy_price"]
                if trade_type == 1 or trade_type == 3
                else buy_sell_prices.loc[i, "sell_price"]
            )
            trades.append((i, trade_type, price, bond_index))
        else:
            trades.append((i, 0, 0, 0))  # No trade

    return pd.DataFrame(trades, columns=["time", "type", "price", "bond_index"])


# Generate data for both bonds
bid_ask_spreads_1 = generate_bid_ask_prices(N, mu=mus_bid_ask[0])
bid_ask_spreads_2 = generate_bid_ask_prices(N, mu=mus_bid_ask[1])

# Add noise to the spreads (properly handling DataFrame addition)
bid_ask_spreads_1[0] = bid_ask_spreads_1[0] + np.random.normal(0, sigmas_epsilons[0], N)
bid_ask_spreads_2[0] = bid_ask_spreads_2[0] + np.random.normal(0, sigmas_epsilons[1], N)

buy_sell_prices_1 = generate_buy_sell_prices(price_walk_1, bid_ask_spreads_1)
buy_sell_prices_2 = generate_buy_sell_prices(price_walk_2, bid_ask_spreads_2)

trades = generate_trades(buy_sell_prices_1, buy_sell_prices_2, types)

# Remove zero trades
trades = trades[trades["type"] != 0]

# Save combined trades to CSV
trades.to_csv("trades.csv", index=False)

# Separate trades for plotting
trades_1 = trades[trades["bond_index"] == 1]
trades_2 = trades[trades["bond_index"] == 2]

# Create subplots for both bonds
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot Bond 1
for trades, buy_sell_prices, ax, title in [(trades_1, buy_sell_prices_1, ax1, "Bond 1"),
                                         (trades_2, buy_sell_prices_2, ax2, "Bond 2")]:
    # Plot done trades
    done_trades_buy = trades[trades["type"] == 1]
    done_trades_sell = trades[trades["type"] == 2]
    ax.scatter(done_trades_buy["time"], done_trades_buy["price"], 
              color="red", marker="o", label="Done Trades - Buy")
    ax.scatter(done_trades_sell["time"], done_trades_sell["price"], 
              color="green", marker="o", label="Done Trades - Sell")

    # Plot traded away trades
    traded_away_buy = trades[trades["type"] == 3]
    traded_away_sell = trades[trades["type"] == 4]
    ax.scatter(traded_away_buy["time"], traded_away_buy["price"], 
              color="red", marker="+", label="Traded Away - Buy")
    ax.scatter(traded_away_sell["time"], traded_away_sell["price"], 
              color="green", marker="+", label="Traded Away - Sell")

    # Plot price line
    ax.plot(buy_sell_prices["time"], buy_sell_prices["price"], label="Price")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig("correlated_trades.png")

# Print correlation between the two price series
correlation = np.corrcoef(buy_sell_prices_1["price"], buy_sell_prices_2["price"])[0,1]
print(f"Realized correlation between bonds: {correlation:.3f}")
print("\nBond 1 prices:")
print(buy_sell_prices_1.head())
print("\nBond 2 prices:")
print(buy_sell_prices_2.head())
