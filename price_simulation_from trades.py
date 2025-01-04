import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm

path = "trades.csv"

trades = pd.read_csv(path)

price_0 = trades.loc[0, "price"]

K = 10
A = 1
V = 1
Phi = 1


def sample_K_y0(K, loc=0):
    return np.random.normal(loc, 1, size=K)


def sample_K_x0(K, loc=0, scale=1):
    return np.random.normal(loc, scale, size=K)


def gamma(tau):
    return (1 / 2) * (1 - np.exp(-2 * A * tau)) * V**2

def draw_y_tilde(trades, y_prev, time_index, psi_hat, sigma_i=1., sigma_e=1.):
    """
    Draw y_tilde from the conditional distribution given trades.

    Args:
        trades: DataFrame containing trade information
    """
    trade = trades.loc[time_index]
    prev_trade = trades.loc[time_index - 1]
    tau_diff = trade["time"] - prev_trade["time"]

    trade_type = trades.loc[time_index, "type"]
    trade_price = trades.loc[time_index, "price"]

    scale = np.sqrt(sigma_i**2 * tau_diff + sigma_e**2)

    if trade_type == 1:
        y_tilde = trade_price - psi_hat
    elif trade_type == 2:
        y_tilde = trade_price + psi_hat
    elif trade_type == 3:
        # Right sided truncated normal distribution
        a = (trade_price + psi_hat - y_prev) / scale  # Normalize the lower bound
        y_tilde = truncnorm.rvs(
            a=a, b=np.inf, loc=y_prev, scale=scale, size=K
        )
    elif trade_type == 4:
        # Left-sided truncated normal distribution
        b = (trade_price - psi_hat - y_prev) / scale
        y_tilde = truncnorm.rvs(
            a=-np.inf, b=b, loc=y_prev, scale=scale, size=K
        )
    elif trade_type == 5:
        # Two-sided truncated normal distribution
        # alpha = 0.01 heres
        a = (trade_price - 0.01 * trade_price - (y_prev - psi_hat)) / sigma_e
        b = (trade_price + 0.01 * trade_price - (y_prev - psi_hat)) / sigma_e
        y_tilde = truncnorm.rvs(
            a=a, b=b, loc=y_prev - psi_hat, scale=sigma_e, size=K
        )
    else:
        raise ValueError(f"Unknown trade type: {trade_type}")
    return y_tilde


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
    trade_price = trade["price"]
    trade_type = trade["type"]

    # Common denominator term for all cases
    denominator = np.sqrt(sigma_i**2 * tau_diff + sigma_i**2)


    if trade_type == 1:  # Done(buy) - D2C buy
        weights = norm.pdf((trade_price + phi_hat - y_hats) / denominator)
    elif trade_type == 2:  # Done(sell) - D2C sell
        weights = norm.pdf((trade_price - phi_hat - y_hats) / denominator)
    elif trade_type == 3:  # Traded Away (buy)
        weights = norm.cdf((-trade_price - phi_hat - y_hats) / denominator)
    elif trade_type == 4:  # Traded Away (sell)
        weights = norm.cdf((-trade_price + phi_hat - y_hats) / denominator)

    else:
        raise ValueError(f"Unknown trade type: {trade_type}")

    # Normalize weights
    return weights / np.sum(weights)



x_estimates = np.array([sample_K_x0(K)]) # (m, K) = (1, K)
y_estimates = np.array([sample_K_y0(K, price_0)]) # (m, K) = (1, K)
for time_index in range(1, len(trades)):
    tau_diff = trades.loc[time_index, "time"] - trades.loc[time_index - 1, "time"]

    # ------------ Step 1: Drawing half bid-ask spreads ------------ #
    x_hat = np.array([np.random.normal(
        np.exp(
            -A
            * tau_diff
            * x_estimates[-1]
        ),
        gamma(tau_diff),
    )])
    phi_hat = Phi * np.exp(x_hat)

    # ------------ Step 2: Computing weights ------------ #
    # Some adjustments may be needed in the computation of weights@
    weights = compute_weights(phi_hat, trades, time_index, y_estimates[-1])

    # # ------------ Step 3: Resampling ------------ #
    sampling = np.random.choice(K, size=K, replace=True, p=weights[0])
    x_resampled = np.expand_dims(x_estimates[-1, sampling], axis=0) # expand_dims to keep the shape (m, K)
    phi_resampled = phi_hat[:, sampling]

    # Add new time step to the new x trajectory
    x_estimates = np.concatenate([x_resampled, x_hat[:, sampling]], axis=0) # (m+1, K)

    y_estimates = y_estimates[:, sampling] # (m, K), still need the m+1-th time step

    # ------------ Step 4: Drawing y_tilde ------------ #
    # This function still doesn't work because of the hyperparameters which lead to overflow
    y_tilde = draw_y_tilde(trades, y_estimates[-1], time_index, phi_resampled)

    # ------------ Step 5: Drawing the price y_i ------------ #
    # From this, I couldn't test the function because of the hyperparameters
    sigma_i = 1 # need to be adjusted
    sigma_e = 1 # need to be adjusted
    loc = (sigma_i ** 2 * tau_diff * y_tilde + sigma_e ** 2 * y_estimates[-1]) / (sigma_i ** 2 * tau_diff + sigma_e ** 2)
    scale = (sigma_i ** 2 * sigma_e ** 2) / (sigma_i ** 2 * tau_diff + sigma_e ** 2)
    next_y = np.array([np.random.normal(
        loc=loc,
        scale=scale,
    )])
    y_estimates = np.concatenate([y_estimates, next_y], axis=0)

    # ------------ Step 6: Drawing the price y_i' ------------ #
    # This step is used when we have several bonds (d > 1)
    # In our case, d=1, so we don't need to consider this step for now
