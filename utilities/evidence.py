"""
This script computes the Bayesian evidence using different integration methods (TI, SS, GSS) for a given set of temperature folders.
It reads log-likelihood values and samples from result files, computes log evidence using the specified method, and handles Gaussian KDE for posterior samples.
It also provides a command-line interface for specifying the root folder, likelihood type, approach, and integration method.
"""
import os
import numpy as np
from pathlib import Path
import yaml
from scipy.special import logsumexp
from scipy.stats import gaussian_kde


def read_log_likelihoods(temp_path, month, lik, approach, temp, integration_method):
    """
    Reads the log-likelihood values and samples from a result file.

    Args:
        temp_path (Path): Path to the temperature folder.
        month (str): Current month identifier.
        lik (str): Likelihood type.
        approach (str): Method type.
        temp (int): Temperature index.
        integration_method (str): Integration method (e.g., "GSS").

    Returns:
        tuple: A tuple containing:
            - log_likelihoods (array): Log-likelihood values.
            - beta (float): Temperature value.
            - samples (array): Sampled parameter values (None if beta == 1.0).
    """
    file = temp_path / f"result_month{month}_{lik}_{approach}_temp{temp}.npz"
    result = np.load(file, allow_pickle=True)['posterior'][()]
    log_likelihoods = result['log_likelihood']
    beta = result['temperature']

    if integration_method == "GSS" and beta == 1.0:
        print("Posterior mode found at beta = 1.0")
        return log_likelihoods, beta, None

    samples = [result[key] for key in result.keys() if key not in ['log_likelihood', 'temperature']]
    samples = np.array(samples)
    return log_likelihoods, beta, samples


def compute_log_evidence_ti(log_likelihoods_by_beta, betas):
    """
    Compute log evidence using Thermodynamic Integration (TI).

    Args:
        log_likelihoods_by_beta (list): Log-likelihoods for each beta.
        betas (list): Beta (temperature) values.

    Returns:
        float: Log evidence.
    """
    logZ = 0.0
    for k in range(1, len(betas)):
        delta_beta = betas[k] - betas[k - 1]
        mean_logL_k = np.mean(log_likelihoods_by_beta[k])
        mean_logL_prev = np.mean(log_likelihoods_by_beta[k - 1])
        logZ += 0.5 * delta_beta * (mean_logL_k + mean_logL_prev)
    return logZ


def compute_log_evidence_ss(log_likelihoods_by_beta, betas):
    """
    Compute log evidence using Stepping Stone (SS) method.

    Args:
        log_likelihoods_by_beta (list): Log-likelihoods for each beta.
        betas (list): Beta (temperature) values.

    Returns:
        float: Log evidence.
    """
    logZ = 0.0
    for k in range(1, len(betas)):
        delta_beta = betas[k] - betas[k - 1]
        scaled_ll = delta_beta * log_likelihoods_by_beta[k - 1]
        log_mean = logsumexp(scaled_ll) - np.log(len(scaled_ll))
        logZ += log_mean
    return logZ


def find_posterior_folder(temp_path, month, lik, approach, temp):
    """
    Check if the posterior mode exists for a given temperature.

    Args:
        temp_path (Path): Path to the temperature folder.
        month (str): Current month identifier.
        lik (str): Likelihood type.
        approach (str): Method type.
        temp (int): Temperature index.

    Returns:
        bool: True if posterior mode exists (beta == 1.0), False otherwise.
    """
    file = temp_path / f"result_month{month}_{lik}_{approach}_temp{temp}.npz"
    result = np.load(file, allow_pickle=True)['posterior'][()]
    return result['temperature'] == 1.0


def compute_kde(temp_path, month, lik, approach, temp):
    """
    Compute a Gaussian KDE for the posterior samples.

    Args:
        temp_path (Path): Path to the temperature folder.
        month (str): Current month identifier.
        lik (str): Likelihood type.
        approach (str): Method type.
        temp (int): Temperature index.

    Returns:
        gaussian_kde: Gaussian KDE object for the posterior samples.
    """
    file = temp_path / f"result_month{month}_{lik}_{approach}_temp{temp}.npz"
    result = np.load(file, allow_pickle=True)['posterior'][()]
    samples = [result[key] for key in result.keys() if key not in ['log_likelihood', 'temperature']]
    samples = np.array(samples)
    return gaussian_kde(samples)


def compute_log_evidence_gss(log_likelihoods_by_beta, betas, **kwargs):
    """
    Compute log marginal likelihood using Generalized Steppingstone Sampling (GSS).

    Args:
        log_likelihoods_by_beta (list): Log-likelihoods for each beta.
        betas (list): Beta (temperature) values.
        **kwargs: Additional arguments, including:
            - log_prior (list): Log prior probabilities for each beta.

    Returns:
        float: Log evidence.
    """
    logZ = 0.0
    log_prior = kwargs.get('log_prior')
    for k in range(1, len(betas) - 1):
        delta_beta = betas[k] - betas[k - 1]
        ll = log_likelihoods_by_beta[k - 1]
        log_ref = log_prior[k - 1]
        log_weights = delta_beta * (ll - log_ref)
        log_rk = logsumexp(log_weights) - np.log(len(log_weights))
        logZ += log_rk
    return logZ


def compute_log_evidence(log_likelihoods_by_beta, betas, integration_method, **kwargs):
    """
    Compute log evidence using the specified integration method.

    Args:
        log_likelihoods_by_beta (list): Log-likelihoods for each beta.
        betas (list): Beta (temperature) values.
        integration_method (str): Integration method ("TI", "SS", "GSS").
        **kwargs: Additional arguments for specific methods.

    Returns:
        float: Log evidence.
    """
    if integration_method == "TI":
        return compute_log_evidence_ti(log_likelihoods_by_beta, betas)
    elif integration_method == "SS":
        return compute_log_evidence_ss(log_likelihoods_by_beta, betas)
    elif integration_method == "GSS":
        return compute_log_evidence_gss(log_likelihoods_by_beta, betas, **kwargs)
    else:
        raise ValueError("Invalid method. Choose 'TI', 'SS', or 'GSS'.")


def collect_and_compute(root_folder, lik, approach, integration_method, num_temps=20):
    """
    Collect data from temperature folders and compute Bayesian evidence.

    Args:
        root_folder (str): Root directory containing monthly data.
        lik (str): Likelihood type.
        approach (str): Method type.
        integration_method (str): Integration method ("TI", "SS", "GSS").
        num_temps (int): Number of temperature folders.

    Returns:
        dict: Log evidence for each month, keyed by month name.
    """
    root_folder = Path(root_folder)
    evidence_results = {}

    for month in sorted(root_folder.glob("month_*")):
        path = month / "hmc" / lik / approach
        if not path.exists():
            continue

        ind_month = month.name.split("_")[1]
        print(f"Processing month {ind_month}...")

        log_likelihoods_by_beta = []
        betas = []
        log_prior = []

        if integration_method == "GSS":
            for i in range(1, num_temps + 1):
                temp_path = path / f"temp_{i}"
                if find_posterior_folder(temp_path, ind_month, lik, approach, i):
                    kde = compute_kde(temp_path, ind_month, lik, approach, i)
                    num_temps -= 1
                    break

        for i in range(1, num_temps + 1):
            temp_path = path / f"temp_{i}"
            ll, beta, sample = read_log_likelihoods(temp_path, ind_month, lik, approach, i, integration_method)
            betas.append(beta)
            log_likelihoods_by_beta.append(ll)

            if integration_method == "GSS" and sample is not None:
                log_prior.append(kde.logpdf(sample))

        logZ = compute_log_evidence(log_likelihoods_by_beta, betas, integration_method, log_prior=log_prior)
        evidence_results[month.name] = logZ

    return evidence_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Bayesian evidence using Stepping Stone method.")
    parser.add_argument("--folder", type=str, default="first_year", help="Root directory containing monthly data.")
    parser.add_argument("--lik", type=str, choices=["Gamma", "Whittle"], required=True, help="Likelihood type.")
    parser.add_argument("--approach", type=str, choices=["cyclo", "stat"], required=True, help="Method type.")
    parser.add_argument("--integration", default="TI", type=str, choices=["TI", "SS", "GSS"], help="Integration method.")
    args = parser.parse_args()

    results = collect_and_compute(args.folder, args.lik, approach = args.approach, integration_method = args.integration)
    for month, logZ in results.items():
        print(f"{month}: log evidence = {logZ:.2f}")

