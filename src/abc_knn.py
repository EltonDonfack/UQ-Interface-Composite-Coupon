import numpy as np
import pandas as pd
import openturns as ot

def ABC_per_curve(F_obs_list, Y_obs_list, theta_df, k_nn=50, sigma=None,
                  prior_ot=None, proposal_pdf_fn=None, eps_safe=1e-300):
    """
    Performs curve-by-curve Approximate Bayesian Computation (ABC-kNN) inference
    with optional prior/proposal density weighting (Importance Sampling).

    Parameters:
    -----------
    F_obs_list : list of numpy.ndarray
        List of simulated output ensembles for each test curve. 
        Each element is a 2D array of shape (N_simulations, N_time_or_disp_points).
    Y_obs_list : list of numpy.ndarray
        List of experimental target vectors for each test curve.
        Each element is a 1D or 2D array representing the target curve observations.
    theta_df : pandas.DataFrame or numpy.ndarray
        Prior parameter matrix of shape (N_simulations, N_parameters).
    k_nn : int, default=50
        Number of nearest neighbors to retain per experimental curve.
    sigma : float, optional
        Bandwidth parameter for the Gaussian kernel. If None, it defaults to 
        the maximum L2 distance among the k-nearest neighbors.
    prior_ot : openturns.Distribution, optional
        Multivariate OpenTURNS prior distribution object. Used to compute prior densities.
    proposal_pdf_fn : callable or openturns.Distribution, optional
        Proposal density function q(theta). If provided alongside prior_ot, 
        an importance sampling correction weight (prior / proposal) is applied.
    eps_safe : float, default=1e-300
        Small numerical floor to prevent zero-division and numerical underflow.

    Returns:
    --------
    theta_samples_list : list of numpy.ndarray
        Accepted parameter samples (k-nearest neighbors) for each test curve.
    weights_list : list of numpy.ndarray
        Normalized importance weights corresponding to the accepted parameter samples.
    idx_knn_list : list of numpy.ndarray
        Indices of the retained k-nearest neighbor samples in the candidate pool.
    """
    theta_samples_list = []
    weights_list = []
    idx_knn_list = []

    for idx_curve, (F_k, Y_obs_k) in enumerate(zip(F_obs_list, Y_obs_list)):
        # Compute L2 distance between simulated curves and experimental observation
        distances = np.linalg.norm(F_k - Y_obs_k, axis=1)
        idx_knn = np.argsort(distances)[:k_nn]

        # Extract k-nearest parameters
        if isinstance(theta_df, pd.DataFrame):
            theta_knn = theta_df.iloc[idx_knn].to_numpy()
        else:
            theta_knn = theta_df[idx_knn]

        distances_knn = distances[idx_knn]

        # Set bandwidth parameter for kernel density weighting
        sigma_curve = sigma if sigma is not None else np.max(distances_knn)

        # Base Gaussian kernel weighting
        weights = np.exp(-0.5 * (distances_knn / sigma_curve)**2)

        # Apply prior/proposal probability density correction (Importance Sampling)
        if prior_ot is not None:
            prior_vals = np.array([
                max(prior_ot.computePDF(list(theta)), eps_safe)
                for theta in theta_knn
            ])
            
            if proposal_pdf_fn is not None:
                proposal_vals = np.array([
                    max(proposal_pdf_fn.computePDF(list(theta)), eps_safe)
                    for theta in theta_knn
                ])
                imp_factor = prior_vals / proposal_vals
            else:
                imp_factor = 1.0

            weights *= imp_factor

        # Numerical safety: avoid negative weights and normalize to sum to 1
        weights = np.maximum(weights, 0.0)
        sum_weights = np.sum(weights)
        
        if sum_weights > 0:
            weights = weights / sum_weights
        else:
            weights = np.ones(k_nn) / k_nn  # Fallback to uniform weighting if underflow occurs

        theta_samples_list.append(theta_knn)
        weights_list.append(weights)
        idx_knn_list.append(idx_knn)

    return theta_samples_list, weights_list, idx_knn_list


# =============================================================================
# Usage Example
# Note: Real prior bounds have been anonymized/masked for confidentiality.
# =============================================================================
if __name__ == "__main__":
    # Define OpenTURNS prior distributions for random field hyperparameters
    # (Dummy bounds used below to protect proprietary material values)
    Law_G1c = ot.Uniform(0.0, 1.0)       # Masked value (original proprietary parameter hidden)
    Law_Std_G1c = ot.Uniform(0.0, 0.1)   # Masked value (original proprietary parameter hidden)
    Law_Sn = ot.Uniform(0.0, 100.0)      # Masked value (original proprietary parameter hidden)
    Law_Std_Sn = ot.Uniform(0.0, 10.0)   # Masked value (original proprietary parameter hidden)
    Law_lc = ot.Uniform(0.0, 10.0)       # Masked value (original proprietary parameter hidden)

    # Multivariate prior distribution
    prior_dist = ot.ComposedDistribution([Law_lc, Law_G1c, Law_Std_G1c, Law_Sn, Law_Std_Sn])

    # Extract parameters as NumPy array
    theta_array = theta_df[["Lc", "G1c", "StdG1c", "Sn", "StdSn"]].values

    # Execute ABC-kNN parameter identification per experimental curve
    theta_samples_list, weights_list, idx_knn_list = ABC_per_curve(
        F_obs_list=F_obs_list,
        Y_obs_list=Y_obs_list_filtered,
        theta_df=theta_array,
        k_nn=200,
        prior_ot=prior_dist,
        proposal_pdf_fn=joint_kde
    )

    # Output verification
    for i, (theta_s, w) in enumerate(zip(theta_samples_list, weights_list)):
        print(f"Test Curve {i+1}: Retained theta shape = {theta_s.shape}, Sum of weights = {np.sum(w):.4f}")
