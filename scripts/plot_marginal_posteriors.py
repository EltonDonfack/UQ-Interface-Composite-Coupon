import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_marginal_posteriors(theta_samples_list, weights_list, N_test=5, N_total=7897, k_neighbors=200, save_path=None):
    """
    Plots the 1D marginal posterior probability density functions (PDFs)
    for Mode I random field hyperparameters compared against prior distributions.

    Parameters:
    -----------
    theta_samples_list : list of numpy.ndarray
        List of accepted parameter arrays for each test curve.
    weights_list : list of numpy.ndarray
        List of normalized importance weights for each test curve.
    N_test : int, default=5
        Number of experimental coupon tests used.
    N_total : int, default=7897
        Total number of prior Monte Carlo simulations executed.
    k_neighbors : int, default=200
        Number of nearest neighbors retained in ABC-kNN.
    save_path : str, optional
        File path to save the generated figure.
    """
    # Parameter names formatted in LaTeX
    theta_columns = [
        r"$\ell_c^{I}$",
        r"$\mu_{G_{Ic}}$",
        r"$\mathrm{S}_{G_{Ic}}$",
        r"$\mu_{\sigma_{n,c}}$",
        r"$\mathrm{S}_{\sigma_{n,c}}$"
    ]

    # Prior bounds (Dummy / Masked values used to protect proprietary material data)
    prior_bounds = [
        [0.0, 10.0],  # Dummy bounds for lc (Masked)
        [0.0, 1.0],   # Dummy bounds for mu_GIc (Masked)
        [0.0, 0.1],   # Dummy bounds for S_GIc (Masked)
        [0.0, 100.0], # Dummy bounds for mu_sigman (Masked)
        [0.0, 10.0]   # Dummy bounds for S_sigman (Masked)
    ]

    n_curves = len(theta_samples_list)
    n_theta = len(theta_columns)

    # Figure setup
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    # Color palette
    posterior_colors = plt.cm.Greys(np.linspace(0.4, 0.7, n_curves))
    mean_color = "#1f77b4"
    prior_color = "black"

    handles_list = []
    labels_list = []

    for i, col in enumerate(theta_columns):
        ax = axes[i]
        p_min, p_max = prior_bounds[i]

        # Normalize sample values to [0, 1] for visualization
        norm_samples = [
            (theta_samples_list[k][:, i] - p_min) / (p_max - p_min)
            for k in range(n_curves)
        ]

        theta_grid_norm = np.linspace(0, 1, 1000)
        kde_values = []

        # ---------------------------------------------------------------------
        # 1. Uniform Prior Representation
        # ---------------------------------------------------------------------
        prior_density = np.ones_like(theta_grid_norm)
        lbl_prior = "Prior" if i == 0 else None
        line_prior, = ax.plot(theta_grid_norm, prior_density, color=prior_color,
                              linestyle='--', lw=2, label=lbl_prior)

        # Draw vertical bounds for the uniform prior
        ax.plot([0, 0], [0, 1], color=prior_color, linestyle='--', lw=1.5, alpha=0.8)
        ax.plot([1, 1], [0, 1], color=prior_color, linestyle='--', lw=1.5, alpha=0.8)

        if i == 0:
            handles_list.append(line_prior)
            labels_list.append(lbl_prior)

        # ---------------------------------------------------------------------
        # 2. Individual Test Posteriors (Curve-by-Curve)
        # ---------------------------------------------------------------------
        indiv_handle = None
        for k_idx in range(n_curves):
            w = np.asarray(weights_list[k_idx])
            w /= np.sum(w)

            kde = gaussian_kde(norm_samples[k_idx], weights=w, bw_method='silverman')
            values = kde(theta_grid_norm)
            kde_values.append(values)

            lbl = "Individual posteriors" if (i == 0 and k_idx == 0) else None
            line_indiv, = ax.plot(theta_grid_norm, values, color=posterior_colors[k_idx],
                                  lw=1.8, alpha=0.8, label=lbl)
            if lbl is not None:
                indiv_handle = line_indiv

        if i == 0:
            handles_list.append(indiv_handle)
            labels_list.append("Individual posteriors")

        # ---------------------------------------------------------------------
        # 3. Global Mean Posterior (ABC-kNN Ensemble Aggregation)
        # ---------------------------------------------------------------------
        mean_density = np.mean(kde_values, axis=0)
        mean_density /= np.trapz(mean_density, theta_grid_norm)

        lbl_mean = "Posterior mean (ABC-kNN)" if i == 0 else None
        line_mean, = ax.plot(theta_grid_norm, mean_density, color=mean_color, lw=3, label=lbl_mean)
        if i == 0:
            handles_list.append(line_mean)
            labels_list.append(lbl_mean)

        # ---------------------------------------------------------------------
        # Axes & Labels formatting
        # ---------------------------------------------------------------------
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks([0, 1])
        ax.set_xlabel("Normalized", fontsize=20)
        ax.set_title(f"Marginal density of {col}", fontsize=25)
        ax.set_ylabel("Density", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=15)

    # Remove empty 6th subplot
    for k_idx in range(n_theta, len(axes)):
        fig.delaxes(axes[k_idx])

    # Global Title and Legend
    info_text = rf"Bayesian calibration results: $N_{{test}} = {N_test}$, $N = {N_total}$, $k = {k_neighbors}$"
    fig.legend(handles_list, labels_list, loc="upper center", ncol=3, fontsize=25,
               frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.5, 1.06, info_text, ha='center', fontsize=22, weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# Execution Example
# =============================================================================
if __name__ == "__main__":
    # Call the plotting function (expects ABC-kNN output variables)
    plot_marginal_posteriors(
        theta_samples_list=theta_samples_list,
        weights_list=weights_list,
        N_test=5,
        N_total=7897,
        k_neighbors=200,
        save_path="bayesian_calibration_results_normal.png"
    )
