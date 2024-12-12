import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_random_walk_price(initial_price, volatility, steps):
    prices = [initial_price]
    for i in range(steps):
        prices.append(prices[-1] * (1 + np.random.normal(0, volatility)))
    return prices


types = {
    "Done(buy)": 1,
    "Done(sell)": 2,
    "Traded Away (buy)": 3,
    "Traded Away (sell)": 4,
    # "D2D Trade": 5,
}

N = 100
price_walk = generate_random_walk_price(50, 0.05, N)


def generate_buy_sell_prices(price_walk, bid_ask_spreads):
    buy_sell_prices = pd.DataFrame(price_walk)
    buy_sell_prices["buy_price"] = buy_sell_prices[0] + bid_ask_spreads[0]
    buy_sell_prices["sell_price"] = buy_sell_prices[0] - bid_ask_spreads[0]
    # Adding gaussian noise to the buy and sell prices
    buy_sell_prices["buy_price"] = buy_sell_prices["buy_price"] + np.abs(
        np.random.normal(0, 1, len(buy_sell_prices))
    )
    buy_sell_prices["sell_price"] = buy_sell_prices["sell_price"] - np.abs(
        np.random.normal(0, 1, len(buy_sell_prices))
    )
    buy_sell_prices.columns = ["price", "buy_price", "sell_price"]
    buy_sell_prices["time"] = range(len(buy_sell_prices))
    return buy_sell_prices


def generate_bid_ask_prices(steps, initial_bid_ask_spread=0, A=1, V=1, Phi=1):
    spreads = [initial_bid_ask_spread]
    for i in range(1, steps):
        spread = -A * spreads[-1] + V * np.random.normal(0, 1)
        spreads.append(Phi * np.exp(spread))

    return pd.DataFrame(spreads)


def generate_trades(buy_sell_prices, types, prob_trade=0.2):
    trades = []
    for i in range(len(buy_sell_prices)):
        is_traded = np.random.choice([True, False], p=[prob_trade, 1 - prob_trade])
        if is_traded:
            trade_type = np.random.choice(list(types.values()))
            price = (
                buy_sell_prices.loc[i, "buy_price"]
                if trade_type == 1 or trade_type == 3
                else buy_sell_prices.loc[i, "sell_price"]
            )
            trades.append((i, trade_type, price))
        else:
            trades.append((i, 0, 0))

    return pd.DataFrame(trades, columns=["time", "type", "price"])


bid_ask_spreads = generate_bid_ask_prices(N)

buy_sell_prices = generate_buy_sell_prices(price_walk, bid_ask_spreads)
trades = generate_trades(buy_sell_prices, types)
# Removing trades that are 0
trades = trades[trades["type"] != 0]
trades.to_csv("trades.csv", index=False)
# Plot done trades (type 1 and 2) as points
done_trades_buy = trades[trades["type"] == 1]
done_trades_sell = trades[trades["type"] == 2]
plt.scatter(
    done_trades_buy["time"],
    done_trades_buy["price"],
    color="red",
    marker="o",
    label="Done Trades - Buy",
)
plt.scatter(
    done_trades_sell["time"],
    done_trades_sell["price"],
    color="green",
    marker="o",
    label="Done Trades - Sell",
)

# Plot traded away trades (type 3 and 4) as plus markers
traded_away_buy = trades[trades["type"] == 3]
traded_away_sell = trades[trades["type"] == 4]
plt.scatter(
    traded_away_buy["time"],
    traded_away_buy["price"],
    color="red",
    marker="+",
    label="Traded Away - Buy",
)
plt.scatter(
    traded_away_sell["time"],
    traded_away_sell["price"],
    color="green",
    marker="+",
    label="Traded Away - Sell",
)

plt.plot(buy_sell_prices["time"], buy_sell_prices["price"], label="Price")

plt.legend()
print(buy_sell_prices.head())
plt.savefig("trades.png")
