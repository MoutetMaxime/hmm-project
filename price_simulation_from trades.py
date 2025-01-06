import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm

path = "trades.csv"

trades = pd.read_csv(path)

price_0 = trades.loc[0, "YtB"]

Phi = 1
K = 5  # Number of particles
d = 2  # Number of bonds
A = 1000 * np.eye(d)
V = np.eye(d)
sigmas_i = np.array([0.5 * 1e-4, 0.62 * 1e-4])
sigmas_eps = np.array([5e-2 * 0.79e-4 * 2, 5e-2 * 0.73e-4 * 2])
cov_matrix = np.array(
    [
        [sigmas_i[0] ** 2, sigmas_i[0] * sigmas_i[1] * 0.843],
        [sigmas_i[0] * sigmas_i[1] * 0.843, sigmas_i[1] ** 2],
    ]
)  # Covariance matrix for the bonds


def sample_K_y0(d, loc=0.7, scale=0.6 * 1e-4, K=1000):
    return np.random.normal(loc, scale, size=(d, K))


def sample_K_x0(d, loc=-9, scale=0.83, K=1000):
    return np.random.normal(loc, scale, size=(d, K))


def gamma(tau, A, V):
    return (1 / np.sum(A)) * (1 - np.exp(-2 * A * tau)) * (V @ V.T)


def draw_y_tilde(
    ytb,
    y_prev,
    psi_hat,
    trade_type,
    bond_index,
    tau_diff,
    cov_matrix,
    sigmas_eps,
    K=1000,
    alpha_i=0.5,
):
    """
    Draw y_tilde from the conditional distribution given trades.

    Args:
        trades: DataFrame containing trade information
    """
    scale = np.sqrt(
        cov_matrix[bond_index, bond_index] * tau_diff + sigmas_eps[bond_index] ** 2
    )

    if trade_type == 1:
        y_tilde = ytb - psi_hat[bond_index]
    elif trade_type == 2:
        y_tilde = ytb + psi_hat[bond_index]
    elif trade_type == 3:
        # Right sided truncated normal distribution
        a = (
            ytb + psi_hat[bond_index] - y_prev[bond_index]
        ) / scale  # Normalize the lower bound
        y_tilde = truncnorm.rvs(
            a=a, b=np.inf, loc=y_prev[bond_index], scale=scale, size=K
        )
    elif trade_type == 4:
        # Left-sided truncated normal distribution
        b = (ytb - psi_hat[bond_index] - y_prev[bond_index]) / scale
        y_tilde = truncnorm.rvs(
            a=-np.inf, b=b, loc=y_prev[bond_index], scale=scale, size=K
        )
    elif trade_type == 5:
        # Two-sided truncated normal distribution
        a = (
            ytb
            - alpha_i * psi_hat[bond_index]
            - (y_prev[bond_index] - psi_hat[bond_index])
        ) / sigmas_eps[bond_index]
        b = (
            ytb
            + alpha_i * psi_hat[bond_index]
            - (y_prev[bond_index] - psi_hat[bond_index])
        ) / sigmas_eps[bond_index]
        y_tilde = truncnorm.rvs(
            a=a,
            b=b,
            loc=y_prev[bond_index] - psi_hat[bond_index],
            scale=sigmas_eps[bond_index],
            size=K,
        )
    else:
        raise ValueError(f"Unknown trade type: {trade_type}")
    return y_tilde


import numpy as np
from scipy.stats import norm


def compute_weights(
    phi_hat,
    ytb,
    tau_diff,
    trade_type,
    bond_index,
    y_hats,
    cov_matrix,
    sigmas_eps,
    alpha_i=0.5,
):
    """
    Compute weights based on trade type and direction

    Args:
        phi_hat: Current phi estimate
        trades: DataFrame containing trade information
        time_index: Current time index
        y_hats: Current y estimates
        sigma_i: Bond-specific volatility parameter (default=1)
    """
    # Common denominator term for all cases
    denominator = np.sqrt(
        cov_matrix[bond_index, bond_index] * tau_diff + sigmas_eps[bond_index] ** 2
    )

    if trade_type == 1:  # Done(buy) - D2C buy
        weights = norm.pdf(
            (ytb + phi_hat[bond_index] - y_hats[bond_index]) / denominator
        )
    elif trade_type == 2:  # Done(sell) - D2C sell
        weights = norm.pdf(
            (ytb - phi_hat[bond_index] - y_hats[bond_index]) / denominator
        )
    elif trade_type == 3:  # Traded Away (buy)
        weights = norm.cdf(
            (-ytb - phi_hat[bond_index] - y_hats[bond_index]) / denominator
        )
    elif trade_type == 4:  # Traded Away (sell)
        weights = norm.cdf(
            (-ytb + phi_hat[bond_index] - y_hats[bond_index]) / denominator
        )
    elif trade_type == 5:  # Traded Away (two-sided)
        weights = norm.cdf(
            (-ytb + alpha_i * phi_hat[bond_index] - y_hats[bond_index]) / denominator
        ) - norm.cdf(
            (-ytb - alpha_i * phi_hat[bond_index] - y_hats[bond_index]) / denominator
        )

    else:
        raise ValueError(f"Unknown trade type: {trade_type}")

    # Normalize weights
    return weights / np.sum(weights)


def draw_other_bonds(y_prev, y, bond_index, cov_matrix, tau_diff, K=1000):
    y_other_bonds = np.zeros((d - 1, K))
    for k in range(K):
        scale = np.delete(cov_matrix, bond_index, axis=1)

        mu = (
            np.delete(y_prev, bond_index, axis=0)[:, k]
            + (y - y_prev[bond_index])[k]
            * scale[bond_index, bond_index]
            / cov_matrix[bond_index, bond_index]
        )

        scale = np.delete(scale, bond_index, axis=0)
        y_other_bonds[:, k] = np.random.normal(mu, scale * tau_diff)
    return y_other_bonds


x_estimates = np.array([sample_K_x0(d, K=K)])  # (m, d, K) = (1, d, K)
y_estimates = np.array([sample_K_y0(d, loc=price_0, K=K)])  # (m, d, K) = (1, d, K)
result = np.zeros((len(trades), d, K))
result[0] = y_estimates[-1]
print("x_estimates", x_estimates.shape)
print("y_estimates", y_estimates.shape)

for time_index in range(1, len(trades)):
    ### Extracting the relevant information from the trades dataframe
    bond_index = trades.loc[time_index, "bond_index"]
    ytb = trades.loc[time_index, "YtB"]
    prev_ytb = trades.loc[time_index - 1, "YtB"]
    trade_type = trades.loc[time_index, "type"]
    tau_diff = trades.loc[time_index, "time"] - trades.loc[time_index - 1, "time"]
    print("Type of trade", trade_type)

    # ------------ Step 1: Drawing half bid-ask spreads ------------ #
    print("STEP 1")
    # Les xt sont iid gaussiens, donc x_hat peut être tirés selon une N(-9,0.83)
    # Pas d'effet de tau m+1 - tau m car indep des xt
    # x_hat = np.array(
    #     [
    #         np.random.multivariate_normal(
    #             np.dot(np.exp(-A * tau_diff), x_estimates[-1, :, k]),
    #             gamma(tau_diff, A, V),
    #         )
    #         for k in range(K)
    #     ]
    # ).reshape(
    #     -1, d, K
    # )  # (1, d, K)
    x_hat = np.array(
        sample_K_x0(d, K=K)
    ).reshape(1, d, K)  # (1, d, K)
    def sample_K_x0(d, loc=-9, scale=0.83, K=1000):
        return np.random.normal(loc, scale, size=(d, K))
    
    phi_hat = Phi * np.exp(x_hat[-1])  # (d, K)
    print("phi_hat", phi_hat.shape)
    print("x_hat", x_hat.shape)
    print("-------------------")
    # ------------ Step 2: Computing weights ------------ #
    print("STEP 2")
    print("phi_hat", phi_hat)
    print("ytb", ytb)
    print("previous ytb", prev_ytb)
    print("cov_matrix", cov_matrix)
    print("sigmas_eps", sigmas_eps)
    weights = compute_weights(
        phi_hat,
        ytb,
        tau_diff,
        trade_type,
        bond_index,
        y_estimates[-1],
        cov_matrix,
        sigmas_eps,
        alpha_i=0.5,
    )  # (K,)
    print("weights", weights.shape)
    print(weights)
    print("-------------------")

    # # ------------ Step 3: Resampling ------------ #
    print("STEP 3")
    sampling = np.random.choice(K, size=K, replace=True, p=weights)
    x_resampled = x_estimates[:, :, sampling]
    phi_resampled = phi_hat[:, sampling]

    # Add new time step to the new x trajectory
    x_estimates = np.concatenate(
        [x_resampled, x_hat[:, :, sampling]], axis=0
    )  # (m+1, d, K)
    y_estimates = y_estimates[
        :, :, sampling
    ]  # (m, d, K), still need the m+1-th time step
    print("phi_resampled", phi_resampled.shape)
    print("x_estimates", x_estimates.shape)
    print("y_estimates", y_estimates.shape)
    print("-------------------")

    # ------------ Step 4: Drawing y_tilde ------------ #
    print("STEP 4")
    y_tilde = draw_y_tilde(
        ytb,
        y_estimates[-1],
        phi_resampled,
        trade_type,
        bond_index,
        tau_diff,
        cov_matrix,
        sigmas_eps,
    )  # (K,)
    print("y_tilde", y_tilde.shape)
    print("-------------------")

    # ------------ Step 5: Drawing the price y_i ------------ #
    print("STEP 5")
    sigma_i = cov_matrix[bond_index, bond_index]
    sigma_e = sigmas_eps[bond_index]
    loc = (
        sigma_i * tau_diff * y_tilde + sigma_e**2 * y_estimates[-1, bond_index]
    ) / (sigma_i * tau_diff + sigma_e**2)
    scale = (sigma_i**2 * sigma_e**2) / (sigma_i**2 * tau_diff + sigma_e**2)
    next_y = np.random.normal(
        loc=loc,
        scale=scale,
    )  # (K,)
    print("next_y", next_y.shape)
    print("-------------------")

    # ------------ Step 6: Drawing the price y_i' ------------ #
    # This step is used when we have several bonds (d > 1)
    print("STEP 6")
    if d > 1:
        y_other_bonds = draw_other_bonds(
            y_estimates[-1], next_y, bond_index, cov_matrix, tau_diff, K=K
        )
        print("y_other_bonds", y_other_bonds.shape)
        new_y = np.insert(y_other_bonds, bond_index, next_y, axis=0)
        print("new_y", new_y.shape)
        y_estimates = np.concatenate(
            [y_estimates, np.expand_dims(new_y, axis=0)], axis=0
        )
    else:
        y_estimates = np.concatenate([y_estimates, next_y], axis=0)

    print("y_estimates", y_estimates.shape)
    print("=====================================================")

    result[time_index] = y_estimates[-1]


# Plot the results
import matplotlib.pyplot as plt

# Plot trades
plt.figure(figsize=(12, 6))
plt.plot(trades["time"], trades["price"], "o", label="Trades")

# Plot estimated prices
plt.plot(trades["time"], result.mean(axis=1), label="Estimated price")

# Plot 95% interval
lower_bound = np.percentile(result, 2.5, axis=1)
upper_bound = np.percentile(result, 97.5, axis=1)
plt.fill_between(
    trades["time"],
    lower_bound,
    upper_bound,
    color="blue",
    alpha=0.2,
    label="95% Confidence Interval",
)


plt.legend()
plt.show()