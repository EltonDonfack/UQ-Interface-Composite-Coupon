import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D


def plot_posterior_correlation_matrix(theta_samples_list, weights_list, save_path=None):
    """
    Plots a 5x5 corner matrix displaying 1D marginal posterior distributions on the 
    diagonal and 2D pairwise joint posterior scatter plots on the lower triangle.

    Parameters:
    -----------
    theta_samples_list : list of numpy.ndarray
        List of accepted parameter arrays for each test curve.
    weights_list : list of numpy.ndarray
        List of normalized importance weights for each test curve.
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

    # -------------------------------------------------------------------------
    # 1. Aggregate and Normalize Weighted Samples
    # -------------------------------------------------------------------------
    all_samples = []
    all_weights = []

    for k_idx in range(n_curves):
        all_samples.append(theta_samples_list[k_idx])
        w = np.asarray(weights_list[k_idx], dtype=float)
        all_weights.append(w / np.sum(w))

    global_samples = np.vstack(all_samples)
    global_weights = np.concatenate(all_weights)
    global_weights /= np.sum(global_weights)

    global_samples_norm = np.zeros_like(global_samples)
    for i in range(n_theta):
        p_min, p_max = prior_bounds[i]
        global_samples_norm[:, i] = (global_samples[:, i] - p_min) / (p_max - p_min)

    # -------------------------------------------------------------------------
    # 2. Build 5x5 Plot Grid
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(n_theta, n_theta, figsize=(16, 16))
    theta_grid_norm = np.linspace(0, 1, 500)

    mean_color = "#1f77b4"
    scatter_color = "gray"

    for row in range(n_theta):
        for col in range(n_theta):
            ax = axes[row, col]

            # -----------------------------------------------------------------
            # CASE 1: DIAGONAL (1D Global Marginal Densities)
            # -----------------------------------------------------------------
            if row == col:
                kde_values = []
                for k_idx in range(n_curves):
                    p_min, p_max = prior_bounds[col]
                    norm_indiv = (theta_samples_list[k_idx][:, col] - p_min) / (p_max - p_min)
                    w = np.asarray(weights_list[k_idx], dtype=float)
                    w /= np.sum(w)

                    kde = gaussian_kde(norm_indiv, weights=w, bw_method='silverman')
                    kde_values.append(kde(theta_grid_norm))

                mean_density = np.mean(kde_values, axis=0)
                mean_density /= np.trapz(mean_density, theta_grid_norm)

                ax.plot(theta_grid_norm, mean_density, color=mean_color, lw=2.5)
                ax.fill_between(theta_grid_norm, mean_density, color=mean_color, alpha=0.1)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(0, np.max(mean_density) * 1.1)

            # -----------------------------------------------------------------
            # CASE 2: LOWER TRIANGLE (2D Joint Scatter Samples)
            # -----------------------------------------------------------------
            elif row > col:
                stride = max(1, len(global_samples_norm) // 2000)
                ax.scatter(global_samples_norm[::stride, col], global_samples_norm[::stride, row],
                           color=scatter_color, alpha=0.2, s=4, zorder=2)

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)

            # -----------------------------------------------------------------
            # CASE 3: UPPER TRIANGLE (Hide Subplots)
            # -----------------------------------------------------------------
            else:
                ax.set_axis_off()
                continue

            # -----------------------------------------------------------------
            # Formatting Labels and Axes Ticks
            # -----------------------------------------------------------------
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.tick_params(axis='both', labelsize=11)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])

            if row == n_theta - 1:
                ax.set_xlabel(f"Normalized {theta_columns[col]}", fontsize=18, labelpad=8)
            else:
                ax.set_xticklabels([])

            if col == 0 and row > 0:
                ax.set_ylabel(f"Normalized {theta_columns[row]}", fontsize=18, labelpad=8)
            elif row != col:
                ax.set_yticklabels([])

    # -------------------------------------------------------------------------
    # 3. Figure Title and Legend
    # -------------------------------------------------------------------------
    fig.suptitle("Posterior Cross-Correlation", fontsize=18, fontweight='bold', y=0.98)

    legend_handles = [
        Line2D([0], [0], color=mean_color, lw=2.5, label='Global Posterior Mean (1D Marginal)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=scatter_color,
               markersize=8, alpha=0.4, label='ABC Posterior Samples (2D Joint)')
    ]

    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.95, 0.92),
               fontsize=14, frameon=True, shadow=False, facecolor='white', edgecolor='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# Execution Example
# =============================================================================
if __name__ == "__main__":
    # Call the plotting function (expects ABC-kNN output variables)
    plot_posterior_correlation_matrix(
        theta_samples_list=theta_samples_list,
        weights_list=weights_list,
        save_path="parameter_identifiability_matrix.png"
    )
