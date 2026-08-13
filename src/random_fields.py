import numpy as np
import openturns as ot
import matplotlib.pyplot as plt

def field_simple_plot(nx, ny, lc, type_corr_func="Gaussian"):
    """
    Generates and displays a realization of a stationary random field
    on a regular unit grid.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions (number of points in X and Y directions).
    lc : float
        Correlation length (scale parameter).
    type_corr_func : str
        Kernel type: "Gaussian", "Exp", "Matern3/2", "Matern5/2".
        
    Returns:
    --------
    sample : numpy.ndarray
        Array of shape (ny, nx) containing the simulated field values.
    """
    
    # 1. Spatial mesh construction (unit step size between nodes)
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    # Reshape into a coordinate list (N, 2) for distance calculation
    points = np.array([xx.ravel(), yy.ravel()]).T
    n_points = nx * ny

    # 2. Spatial correlation matrix calculation
    # Compute Euclidean distance matrix between all point pairs
    dist = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    
    # Apply the selected correlation kernel
    if type_corr_func == "Gaussian":
        corr_matrix = np.exp(-dist**2 / (2 * lc**2))
    elif type_corr_func == "Exp":
        corr_matrix = np.exp(-dist / lc)
    elif type_corr_func == "Matern3/2":
        corr_matrix = (1 + np.sqrt(3) * (dist / lc)) * np.exp(-np.sqrt(3) * dist / lc)
    elif type_corr_func == "Matern5/2":
        corr_matrix = (1 + np.sqrt(5) * (dist / lc) + 5 * (dist**2 / (3 * lc**2))) * np.exp(-np.sqrt(5) * dist / lc)
    else:
        raise ValueError(f"Unrecognized correlation function type: {type_corr_func}")

    # 3. OpenTURNS interface for probabilistic modeling
    # Create OpenTURNS correlation matrix object
    ot_corr_mat = ot.CorrelationMatrix(n_points)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            ot_corr_mat[i, j] = corr_matrix[i, j]

    # Construct Normal Copula (models spatial dependence)
    copula = ot.NormalCopula(ot_corr_mat)
    
    # Marginal distribution parameters (arbitrary Normal distribution)
    mu = 0.5
    std = 0.1
    marginal = ot.Normal(mu, std)
    
    # Create multivariate distribution (N correlated dimensions)
    distribution = ot.ComposedDistribution([marginal] * n_points, copula)

    # 4. Sample a single realization and reshape into grid matrix
    sample = np.array(distribution.getSample(1)).reshape(ny, nx)

    # 5. Graphical visualization with Matplotlib (imshow)
    plt.figure(figsize=(16, 4)) 
    
    # Set vmin/vmax to +/- 3 sigma to stabilize the color scale
    img = plt.imshow(sample, 
                     origin='lower', 
                     extent=[0, nx, 0, ny], 
                     cmap='magma', 
                     vmin=mu - 3 * std, 
                     vmax=mu + 3 * std, 
                     aspect='equal')

    # --- Plot aesthetics configuration ---
    
    # Adjust colorbar to field size
    cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
    cbar.set_label("Simulated Value", fontsize=15, weight='bold')
    cbar.ax.tick_params(labelsize=15)
    
    # Enlarge labels and title fonts for enhanced readability
    plt.xlabel("X", fontsize=15)
    plt.ylabel("Y", fontsize=15)
    plt.title(f"{type_corr_func} Simulation - Lc={lc}", fontsize=19, pad=10)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    plt.show()

    return sample

# --- Example Execution ---
# Field generation on an elongated grid with strong correlation length
field = field_simple_plot(nx=130, ny=25, lc=5, type_corr_func="Gaussian")
