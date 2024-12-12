import pandas as pd
import numpy as np
from scipy.stats import norm, truncnorm
import matplotlib.pyplot as plt

path = "trades.csv"

trades = pd.read_csv(path)

price_0 = trades.loc[0, "price"]

K = 10000  # Increased number of particles for better approximation
A = 1  # Ornstein-Uhlenbeck parameter
V = 1  # Ornstein-Uhlenbeck parameter
Phi = 1  # Spread scaling parameter
sigma_e = 0.05 * price_0  # 5% of initial price
sigma_i = 1


def sample_K_y0(K, center):
    return np.random.normal(center, 1, size=K)


def sample_K_x0(K, center=0):
    return np.random.normal(center, 1, size=K)


def gamma(tau):
    return (1 / 2) * (1 - np.exp(-2 * A * tau)) * V**2


def compute_weights(phi_hat, trades, time_index, y_hats, sigma_i, sigma_e):
    trade = trades.loc[time_index]
    prev_trade = trades.loc[time_index - 1 if time_index > 0 else 0]
    tau_diff = trade["time"] - prev_trade["time"]
    denominator = np.sqrt(sigma_i**2 * tau_diff + sigma_e**2)

    trade_type = trade["type"]
    trade_price = trade["price"]

    # Ensure denominator is not too close to zero
    denominator = np.maximum(denominator, 1e-10)

    # Calculate standardized values
    z = (trade_price - y_hats) / denominator
    z_plus_phi = z + phi_hat / denominator
    z_minus_phi = z - phi_hat / denominator

    # Initialize log weights
    log_weights = np.zeros_like(y_hats, dtype=np.float64)

    try:
        if trade_type == 1:  # Done(buy) - D2C buy
            log_weights = (
                -0.5 * np.clip(z_minus_phi**2, -700, 700)
                - np.log(denominator)
                - 0.5 * np.log(2 * np.pi)
            )

        elif trade_type == 2:  # Done(sell) - D2C sell
            log_weights = (
                -0.5 * np.clip(z_plus_phi**2, -700, 700)
                - np.log(denominator)
                - 0.5 * np.log(2 * np.pi)
            )

        elif trade_type == 3:  # Traded Away (buy)
            mask = z_minus_phi > 6
            log_weights[mask] = -np.clip(z_minus_phi[mask] ** 2 / 2, -700, 700)
            log_weights[~mask] = np.log(
                np.maximum(1 - norm.cdf(z_minus_phi[~mask]), 1e-300)
            )

        elif trade_type == 4:  # Traded Away (sell)
            mask = z_plus_phi < -6
            log_weights[mask] = -np.clip(z_plus_phi[mask] ** 2 / 2, -700, 700)
            log_weights[~mask] = np.log(np.maximum(norm.cdf(z_plus_phi[~mask]), 1e-300))

        elif trade_type == 5:  # D2D
            epsilon = 0.01
            upper_z = z + epsilon * trade_price / denominator - phi_hat / denominator
            lower_z = z - epsilon * trade_price / denominator - phi_hat / denominator

            # Handle extreme values with more stable computation
            cdf_diff = norm.cdf(upper_z) - norm.cdf(lower_z)
            cdf_diff = np.maximum(cdf_diff, 1e-300)  # Ensure positive values
            log_weights = np.log(cdf_diff)

        else:
            raise ValueError(f"Unknown trade type: {trade_type}")

        # Replace any infinite values with very negative/positive numbers
        log_weights = np.nan_to_num(log_weights, nan=-1e10, posinf=700, neginf=-700)

        # Subtract maximum log weight for numerical stability
        max_log_weight = np.max(log_weights)
        log_weights -= max_log_weight

        # Convert back from log space with careful clipping
        weights = np.exp(np.clip(log_weights, -700, 700))

        # Handle the case where all weights are zero
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)

        # Normalize
        weights = weights / np.sum(weights)

        return weights

    except Exception as e:
        print(f"Error in compute_weights: {e}")
        print(f"trade_type: {trade_type}")
        print(f"z range: [{np.min(z)}, {np.max(z)}]")
        print(f"phi_hat range: [{np.min(phi_hat)}, {np.max(phi_hat)}]")
        # Return uniform weights as fallback
        return np.ones_like(y_hats) / len(y_hats)


x_estimates = [sample_K_x0(K)]
y_estimates = [sample_K_y0(K, price_0)]


for time_index in range(1, len(trades)):
    tau_diff = trades.loc[time_index, "time"] - trades.loc[time_index - 1, "time"]

    # ------------ Step 1: Drawing half bid-ask spreads ------------ #
    x_hat = np.random.normal(
        np.exp(-A * tau_diff) * x_estimates[-1], np.sqrt(gamma(tau_diff))
    )
    phi_hat = Phi * np.exp(x_hat)

    # ------------ Step 2: Computing weights ------------ #
    weights = compute_weights(
        phi_hat, trades, time_index, y_estimates[-1], sigma_i=1, sigma_e=sigma_e
    )  # Assume sigma_i=1 for simplicity

    # ------------ Step 3: Resampling ------------ #
    resampled_indices = np.random.choice(K, size=K, replace=True, p=weights)
    x_estimates.append(x_hat[resampled_indices])
    y_prev = y_estimates[-1][resampled_indices]
    phi_resampled = phi_hat[resampled_indices]

    # ------------ Step 4: Drawing Y_hat ------------ #
    trade = trades.loc[time_index]
    trade_type = trade["type"]
    trade_price = trade["price"]

    prev_trade = trades.loc[time_index - 1 if time_index > 0 else 0]
    tau_diff = trade["time"] - prev_trade["time"]

    y_hat = np.zeros(K)

    if trade_type == 1:
        y_hat = trade_price + phi_resampled
    if trade_type == 2:
        y_hat = trade_price - phi_resampled
    if trade_type == 3:

        a = trade_price + phi_resampled
        y_hat = truncnorm.rvs(
            a=a, b=np.inf, loc=y_prev, scale=sigma_i**2 * tau_diff + sigma_e**2, size=K
        )

    if trade_type == 4:

        b = trade_price - phi_resampled
        y_hat = truncnorm.rvs(
            a=-np.inf, b=b, loc=y_prev, scale=sigma_i**2 * tau_diff + sigma_e**2, size=K
        )

    if trade_type == 5:
        a = (trade_price - 0.01 * trade_price - y_prev + phi_resampled) / sigma_e
        b = (trade_price + 0.01 * trade_price - y_prev + phi_resampled) / sigma_e
        y_hat = truncnorm.rvs(
            a=a, b=b, loc=y_prev - phi_resampled, scale=sigma_e, size=K
        )

    # ------------ Step 5: Drawing y ------------ #

    # Calculate the variance terms
    var_term1 = sigma_i**2 * tau_diff
    var_term2 = sigma_e**2

    # Calculate mean and variance for the normal distribution
    mean = (var_term1 * y_hat + var_term2 * y_prev) / (var_term1 + var_term2)
    variance = (var_term1 * var_term2) / (var_term1 + var_term2)

    # Draw y values from the normal distribution
    y = np.random.normal(mean, np.sqrt(variance), size=K)

    y_estimates.append(y)


# Convert to numpy arrays for easier manipulation
x_estimates = np.array(x_estimates)
y_estimates = np.array(y_estimates)

# Example: get the estimated mid-price at the last time step
estimated_mid_price = np.median(y_estimates[-1])
print(f"Estimated mid-price at last time step: {estimated_mid_price}")


# Visualization
plt.figure(figsize=(10, 6))
time = trades["time"]

# Plot estimated mid-YtB (median of particles)
estimated_mid_ytb = np.median(y_estimates, axis=1)
plt.plot(time, estimated_mid_ytb, label="Estimated Mid-YtB", color="blue")


# Plot 95% confidence interval
lower_bound = np.percentile(y_estimates, 2.5, axis=1)
upper_bound = np.percentile(y_estimates, 97.5, axis=1)
plt.fill_between(
    time,
    lower_bound,
    upper_bound,
    color="blue",
    alpha=0.2,
    label="95% Confidence Interval",
)


# Plot observed trade prices, colored by trade type
trade_types = trades["type"].unique()
colors = [
    "green",
    "red",
    "orange",
    "purple",
    "black",
]  # Assign colors to each trade type
for trade_type, color in zip(trade_types, colors):
    trade_times = trades[trades["type"] == trade_type]["time"]
    trade_prices = trades[trades["type"] == trade_type]["price"]
    marker = "+" if trade_type < 5 else "*"
    plt.scatter(
        trade_times,
        trade_prices,
        label=f"Trade Type {trade_type}",
        color=color,
        marker=marker,
    )


plt.xlabel("Time")
plt.ylabel("YtB")
plt.title("Particle Filter Estimation of Mid-YtB")
plt.legend()
plt.grid(True)
plt.savefig("estimated_mid_ytb.png")

# Additional plots to visualize bid and ask prices along with the estimated mid prices and confidence interval.


plt.figure(figsize=(10, 6))


# Plot estimated spread
estimated_spread = np.median(Phi * np.exp(x_estimates), axis=1)


plt.plot(time, estimated_mid_ytb, label="Estimated Mid", color="blue")

plt.plot(time, estimated_mid_ytb + estimated_spread, label="Estimated Ask", color="red")

plt.plot(
    time, estimated_mid_ytb - estimated_spread, label="Estimated Bid", color="green"
)

plt.fill_between(
    time,
    lower_bound,
    upper_bound,
    color="blue",
    alpha=0.2,
    label="95% Confidence Interval",
)

for trade_type, color in zip(trade_types, colors):
    trade_times = trades[trades["type"] == trade_type]["time"]
    trade_prices = trades[trades["type"] == trade_type]["price"]
    marker = "+" if trade_type < 5 else "*"
    plt.scatter(
        trade_times,
        trade_prices,
        label=f"Trade Type {trade_type}",
        color=color,
        marker=marker,
    )


plt.xlabel("Time")
plt.ylabel("YtB")
plt.title("Estimated prices and spread of YtB")
plt.legend()
plt.grid(True)
plt.savefig("estimated_prices_and_spread.png")
