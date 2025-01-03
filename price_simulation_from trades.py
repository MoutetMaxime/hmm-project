import pandas as pd
import numpy as np
from scipy.stats import norm

path = "trades.csv"

trades = pd.read_csv(path)

price_0 = trades.loc[0, "price"]

K = 10
A = 1
V = 1
Phi = 1


def sample_K_y0(K, center):
    return np.random.normal(center, 1, size=K)


def sample_K_x0(K, center=0):
    return np.random.normal(center, 1, size=K)


def gamma(tau):
    return (1 / 2) * (1 - np.exp(-2 * A * tau)) * V**2


def compute_weights(phi_hat, trades, time_index, y_hats, sigma_i=1):
    """
    Compute weights based on trade type and direction

    Args:
        phi_hat: Current phi estimate
        trades: DataFrame containing trade information
        time_index: Current time index
        y_hats: Current y estimates
        sigma_i: Bond-specific volatility parameter (default=1)
    """
    trade = trades.loc[time_index]
    prev_trade = trades.loc[time_index - 1]
    tau_diff = trade["time"] - prev_trade["time"]

    # Common denominator term for all cases
    denominator = np.sqrt(sigma_i**2 * tau_diff + sigma_i**2)

    trade_type = trade["type"]

    if trade_type == 1:  # Done(buy) - D2C buy
        weights = norm.cdf((y_hats + phi_hat - y_hats) / denominator)
    elif trade_type == 2:  # Done(sell) - D2C sell
        weights = norm.cdf((y_hats - phi_hat - y_hats) / denominator)
    elif trade_type == 3:  # Traded Away (buy)
        weights = norm.cdf((-trade["price"] - phi_hat - y_hats) / denominator)
    elif trade_type == 4:  # Traded Away (sell)
        weights = norm.cdf((-trade["price"] + phi_hat - y_hats) / denominator)

    else:
        raise ValueError(f"Unknown trade type: {trade_type}")

    # Normalize weights
    return weights / np.sum(weights)


x_estimates = [sample_K_x0(K)]
y_estimates = [sample_K_y0(K, price_0)]
trajectory = []
for time_index in range(1, len(trades)):
    print("Time index: ", time_index)
    print(np.shape(x_estimates))
    # ------------ Step 1: Drawing half bid-ask spreads ------------ #
    weights = np.zeros(K)
    x_hat = np.random.normal(
        np.exp(
            -A
            * (trades.loc[time_index, "time"] - trades.loc[time_index - 1, "time"])
            * x_estimates[-1]
        ),
        gamma(trades.loc[time_index, "time"] - trades.loc[time_index - 1, "time"]),
    )
    phi_hat = Phi * np.exp(x_hat)
    print(np.shape(x_hat))
    # ------------ Step 2: Computing weights ------------ #
    weights = compute_weights(phi_hat, trades, time_index, y_estimates[-1])

    # ------------ Step 3: Resampling ------------ #
    print(np.shape(weights))
    sampling = np.random.choice(K, size=K, replace=True, p=weights)

    x_estimates.append(x_hat[sampling])
    y_prev = y_estimates[-1][sampling]  # Gemini
    phi_resampled = phi_hat[sampling]

    y_estimates.append(y_prev + x_hat[sampling])

    new_trajectory = x_estimates[-1][sampling] + x_hat[sampling]
    trajectory.append(new_trajectory)
    print("New trajectory shape: ", np.shape(new_trajectory))
    phi_estimates = Phi * np.exp(x_hat[sampling])
