"""Visualization functions for qualitatively evaluating
interpolation quality. Used primarily for generating PCA scatterplots and
line graphs to compare GP-interpolated_replay points to linearly-interpolated_replay points
and true points in holdout evaluation."""

# External Python packages
import numpy as np
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # Set style to make plots nicer

# Native Python packages
import os

# Custom Python packages/modules
from utils.execution.preprocessing_utils import check_dims


def plot_gp_params(model, B=256, env="HalfCheetah-v2", outdir="~/ray_results"):
    """Helper function to plot the hyperparameters of a Gaussian Process
    Regression model. These hyperparameters include:

        (i) Covariance noise (\sigma) in each target dimension
        (ii) Lengthscale (typically used with ARD) in each target dimension
        (iii) Outputscale in each target dimension
        (iv) Raw covariance noise in each target dimension

    Parameters:
        model (MaternKernel): A BatchedGP object that is a subclass of the
            ExactGP GPyTorch model.
        B (int): The batch size for training. Defaults to 256.
        env (str): The environment name used for plotting. Defaults to
            "HalfCheetah-v2".
        outdir (str): A string corresponding to the output path for saving
            these figures.
    """
    # Extract covariance noise
    covar_noise = model.likelihood.noise_covar.noise.cpu().detach().numpy()
    covar_noise_stacked = np.squeeze([covar_noise[i::B] for i in range(B)])

    # Extract lengthscale
    lengthscale = model.covar_module.base_kernel.lengthscale.cpu().detach().numpy()
    lengthscale_stacked = np.squeeze([lengthscale[i::B] for i in range(B)])

    # Extract outputscale
    outputscale = model.covar_module.outputscale.cpu().detach().numpy()
    outputscale_stacked = np.squeeze([outputscale[i::B] for i in range(B)])

    # Get raw covariance noise
    raw_noise = model.likelihood.noise_covar.raw_noise.cpu().detach().numpy()
    raw_noise_stacked = np.squeeze([raw_noise[i::B] for i in range(B)])

    # Extract param names
    param_names = ["Covariance Noise", "Lengthscale", "Outputscale", "Raw Noise"]
    param_list = [covar_noise_stacked, lengthscale_stacked,
                  outputscale_stacked, raw_noise_stacked]

    # Now loop over parameters and dimensions to extract histograms
    for d in range(outputscale_stacked.shape[-1]):

        for name, param in zip(param_names, param_list):
            plt.hist(param[:, d])
            plt.title("Distribution of {}, Dimension {} "
                      "for Environment {}".format(name, d, env))
            plt.xlabel("Param Value")
            plt.ylabel("Frequency")
            out_fname = "param_dist_env_{}_{}_dim_{}.png".format(env, name, d)
            outpath = os.path.join(outdir, out_fname)
            plt.savefig(outpath)
            plt.clf()


def plot_lengthscales(lengthscales, raw_lengthscales, dim=None):
    """Plotting utility function for plotting the lengthscale hyperparameters
    for Gaussian Process Regression.

    Parameters:
        lengthscales (torch.tensor): A tensor of lengthscales. This tensor
            should be detached from the computational graph as well as not on
            a CUDA device (so that we can use the NumPy bridge.
        raw_lengthscales (torch.tensor): A tensor of raw (learned) lengthscales.
            This tensor should be detached from the computational graph as well
            as not on a CUDA device (so that we can use the NumPy bridge.
        dim (int): The target dimension to visualize, if provided. If not
            provided (if dim is None), simply plots over all the lengthscales.
            Defaults to None.
    """
    # Extract NumPy array data for plotting
    lengthscales = np.squeeze(lengthscales.numpy())
    raw_lengthscales = np.squeeze(raw_lengthscales.numpy())

    # Determine what subset, if applicable, to plot
    if dim is None:
        lengthscales_plot_data = lengthscales.flatten()
        raw_lengthscales_plot_data = raw_lengthscales.flatten()
        dim_str = "All Dimensions"
    else:
        lengthscales_plot_data = lengthscales[dim, :].flatten()
        raw_lengthscales_plot_data = raw_lengthscales[dim, :].flatten()
        dim_str = "Dimension {}".format(dim)

    # Plot actual lengthscale used for parameterizing the GPR model
    plt.hist(lengthscales_plot_data)
    plt.title(
        "Histogram of Lengthscale For {} for ARD GPR Model".format(dim_str))
    plt.xlabel("Lengthscale")
    plt.ylabel("Frequency")
    plt.show()
    plt.clf()

    # Plot learned (raw) lengthscale obtained through optimization
    plt.hist(raw_lengthscales_plot_data)
    plt.title(
        "Histogram of Raw Lengthscale For {} for ARD GPR Model".format(dim_str))
    plt.xlabel("Lengthscale")
    plt.ylabel("Frequency")
    plt.show()
    plt.clf()


def plot_likelihood_weights_hist(weights, bins=10, env="HalfCheetah-v2"):
    """Helper visualization function for plotting the likelihood weights of
    the resulting samples produced by GPR.

    Parameters:
        weights (torch.Tensor): Tensor corresponding to the likelihood weights
            for a sampled minibatch produced by GPR.
        bins (int): The number of bins to use in the histogram.
        env (str): The name of the environment in which the agent is being
            trained.
    """
    # Create histogram for likelihood weights
    plt.hist(weights.numpy(), bins=bins)
    plt.xlabel("Weight Value")
    plt.ylabel("Relative Batch Frequency")
    plt.title("Unclipped Weights by Value, {}".format(env))
    plt.show()


def plot_compare_linear_gp(out_true, sample_indices, interp_indices, gp_pred,
                           idx=0, b=None, env="HalfCheetah-v2", use_pca=False):
    """Helper plotting function for validating the Gaussian Process approach.

    Parameters:
        out_true (np.array): The next states/rewards from the replay buffer
            that are used to compute the linear vs. GPR comparison.
        sample_indices (np.array/List): The list or array of indices we
            index into - this is typically the list of samples produced by
            a uniform sampling or PER.
        interp_indices (np.array/List): The list or array of indices we
            index into - this is typically the list of indices for points with
            which we interpolate with our sampled points.
        gp_pred (np.array): A 2D array of predicted values containing next states.
        idx (int): The dimension of next states we wish to index into.
            Note that an additional index for the GP predictions is added,
            because the first dimension of these predictions is composed of the
            reward. Defaults to 0.
        b (np.array): If using SMOTE, a series of values that define how far
            along the sample point to the neighboring point (on a scale of [0,1]).
            the interpolation is performed. Defaults to None, in which case
            interpolation at the arithmetic midpoint (b=0.5) is assumed.
        env (str): The name of the environment in which the agent is
            being trained in. Used for plotting - defaults to "HalfCheetah-v2".
        use_pca (bool): Whether to run plotting using PCA. Defaults to False.
    """
    # Check dimensions
    if (idx > min(out_true.shape[-1], gp_pred.shape[-1])):
        raise Exception("Requested idx is out of range for space.")

    # If using PCA, can only inspect first dimension
    if (idx > 0) and use_pca:
        raise Warning("Performing PCA, so only one dimension is given.")

    # See if we use all dimensions
    if use_pca:
        # Get data for plots using slicing and linear interpolation
        if b is None: # Interpolate at arithmetic mean point
            x_data = np.add(out_true[sample_indices],
                            out_true[interp_indices]) / 2
        else: # Interpolate with SMOTE or Mixup
            one_minus_b = np.subtract(np.ones(b.shape), b)
            x_data = np.add(np.multiply(b, out_true[sample_indices]),
                            np.multiply(one_minus_b, out_true[interp_indices]))
        # GPR predictions
        y_data = gp_pred

    else:
        # Get data for plots using slicing and linear interpolation
        if b is None:  # Interpolate at arithmetic mean point
            x_data = np.add(out_true[sample_indices, idx],
                            out_true[interp_indices, idx]) / 2
        else:  # Interpolate with SMOTE or Mixup

            # If only computing over 1D, untile b
            b = b[:, 0]

            # Form linear prediction
            one_minus_b = np.subtract(np.ones(b.shape), b)
            x_data = np.add(
                np.multiply(b, out_true[sample_indices, idx]),
                np.multiply(one_minus_b, out_true[interp_indices, idx])
            )
        # GPR predictions along a particular index
        y_data = gp_pred[:, idx]

    # Create PCA object if using PCA, and transform data
    if use_pca:
        features = np.vstack((x_data, y_data))
        D = PCA(n_components=1).fit_transform(features)
        N = y_data.shape[0]
        x_data, y_data = D[:N, :], D[N:, :]  # Different axes for linear test

    # Now create scatter plots for visualizing comparison between GP and linear
    plt.scatter(x_data, y_data)
    if use_pca:
        plt.title("(PCA d=1) GPR vs. Linear Interp Delta States, {}".format(env))
        plt.xlabel("PCA (d=1) of Linearly Interpolated Outputs")
        plt.ylabel("PCA (d=1) of GPR Interpolated Outputs")
    else:
        plt.title("GPR vs. Linear Interp Delta States, {}".format(env))
        plt.xlabel("Linearly Interpolated Next State")
        plt.ylabel("GPR Prediction for Next State")
    plt.show()


def pca_plot_all_batch(synthetic_batch, real_batch, sample_indices=None, env=None):
    """Scatter plot utility function for displaying samples using
    Principal Component Analysis (PCA). Calculates a 2D projection using PCA
    and then plots this data.

     Arguments:
         synthetic_batch (SampleBatch): Wrapped dictionary containing
            experience information that will be used for plotting the
            synthetically-generated data points.
         real_batch (SampleBatch): Wrapped dictionary containing
            experience information that will be used for plotting the
            real data points.
         sample_indices (np.array): Array of indices for which to take slice of
            for the real dataset. If None, takes the whole dataset.
         env (str): Name of the environment we are performing PCA on. Defaults
            to None, in which case an environment name is not added for plotting.
    """
    # Get observations, actions, rewards, and next states from samples
    X_syn = np.hstack((check_dims(synthetic_batch["obs"]),
                       check_dims(synthetic_batch["actions"]),
                       check_dims(synthetic_batch["rewards"]),
                       check_dims(synthetic_batch["new_obs"])))
    X_real = np.hstack((check_dims(real_batch["obs"]),
                        check_dims(real_batch["actions"]),
                        check_dims(real_batch["rewards"]),
                        check_dims(real_batch["new_obs"])))

    # Check if we take a slice of the real dataset
    if sample_indices is not None:
        X_real = X_real[sample_indices, :]  # Take slice corresponding to x

    # Use PCA to project feature matrices into two dimensions
    pca = PCA(n_components=2)
    Z = pca.fit_transform(np.vstack((X_syn, X_real)))

    # Now create scatter plot with real and interpolated_replay data
    N = Z.shape[0]

    # Extract interpolated_replay and real data
    x_interpolated, y_interpolated = Z[:N // 2, 0], Z[:N // 2, 1]
    x_real, y_real = Z[N // 2:, 0], Z[N // 2:, 1]

    # Plot interpolated_replay and real data using scatter plots
    plt.scatter(x_interpolated, y_interpolated, color="r", label="Interpolated")
    plt.scatter(x_real, y_real, color="b", label="Real")
    plt.legend()
    if env is not None:
        plt.title("{} PCA (dim = 2) of Real/Interpolated Points".format(env))
    else:
        plt.title("PCA (dim = 2) of Real/Interpolated Points")
    plt.show()
    plt.clf()


def pca_heatmap(synthetic_batch, real_batch, sample_indices=None,
                neighbor_indices=None, env=None):
    """Scatterplot utility for displaying samples using Principal Component
     Analysis. Calculates a 2D projection of the inputs using PCA and then
     plots this data, along with a 1D projection of the outputs using PCA as a
     heatmap color.

     Arguments:
         synthetic_batch (SampleBatch): Wrapped dictionary containing
            experience information that will be used for plotting the
            synthetically-generated data points.
         real_batch (SampleBatchpl): Wrapped dictionary containing
            experience information that will be used for plotting the
            real data points.
        sample_indices (np.array): Array of indices for which to take slice of
            for the real dataset. If None, takes the whole dataset.
        neighbor_indices (np.array): Array of indices for which to take slice of
            for the real dataset.  If None, takes the whole dataset.
        env (str): Name of the environment we are performing PCA on.  Defaults
            to None, in which case the environment name is not added when
            plotting.
     """
    # Get observations, actions, rewards, and next states from samples
    X_syn = np.hstack((check_dims(synthetic_batch["obs"]),
                       check_dims(synthetic_batch["actions"])))
    Y_syn = np.hstack((check_dims(synthetic_batch["rewards"]),
                       check_dims(synthetic_batch["new_obs"])))
    X_real = np.hstack((check_dims(real_batch["obs"]),
                        check_dims(real_batch["actions"])))
    Y_real = np.hstack((check_dims(real_batch["rewards"]),
                        check_dims(real_batch["new_obs"])))

    # Check if we take a slice of the real dataset
    idx = np.array([]).astype(np.int32)
    if sample_indices is not None:
        idx = np.concatenate((idx, sample_indices))
    if neighbor_indices is not None:
        idx = np.concatenate((idx, neighbor_indices))
    # Now take indices
    if (sample_indices is not None) or (neighbor_indices is not None):
        X_real = X_real[idx, :]  # Take slice corresponding to x

        Y_real = Y_real[idx, :]  # Take slice corresponding to y

    # Use PCA to project feature matrices into two dimensions
    pca_x = PCA(n_components=2)  # Use for plotting x
    pca_y = PCA(n_components=1)  # Use for heat map color

    # Fit PCA objects for x and y, and transform datasets
    Z_x = pca_x.fit_transform(np.vstack((X_syn, X_real)))  # Transformed x
    Z_y = pca_y.fit_transform(np.vstack((Y_syn, Y_real)))  # Transformed y

    # Get real and interpolated_replay split point
    N = Z_x.shape[0]
    if (sample_indices is not None) and (neighbor_indices) is not None:
        s = N // 3
    else:
        s = N // 2

    # Extract interpolated_replay and real data
    x_interpolated, y_interpolated = Z_x[:s, 0], Z_x[:s, 1]
    x_real, y_real = Z_x[s:, 0], Z_x[s:, 1]

    # Extract heat map values
    heatmap_interpolated = Z_y[:s]
    heatmap_real = Z_y[s:]

    # Now create scatter plots: real --> o, interpolated_replay --> v
    plt.scatter(x_interpolated, y_interpolated, c=heatmap_interpolated,
                marker="v", label="Interpolated")
    plt.scatter(x_real, y_real, c=heatmap_real, marker="o", label="Real")
    if env is not None:
        plt.title("{} PCA (dx = 2, dy = 1) of Interpolated/Real Points".format(env))
    else:
        plt.title("PCA (dx = 2, dy = 1) of Interpolated/Real Points")
    plt.legend()
    plt.show()


def pca_holdout(pred, true, env=None):
    """Scatter plot utility for displaying samples using Principal Component
     Analysis.

     This function is used to perform PCA on holdout data in order to visualize
     the differences between true holdout points and corresponding approximations
     of these points from interpolation (e.g. with GPR or KNN).  Allows for
     qualitatively evaluating holdout performance, and therefore overall
     interpolation quality.

     Calculates a 1D projection of the inputs using PCA, and then
     plots this data on the x-axis, along with a 1D projection of the outputs
     that is plotted along the y-axis. For the best fits, a roughly 45-degree
     line beginning from the origin should be seen.

     Arguments:
        pred (np.array): Array of predicted values of shape (N, Dy), where N
            is the batch size and Dy is the dimensionality of the outputs.
        pred (np.array): Array of true targets of shape (N, Dy), where N
            is the batch size and Dy is the dimensionality of the outputs.
        env (str): Name of the environment we are performing PCA on.  Defaults
            to None.
     """
    # Use PCA to project feature matrices into two dimensions
    pca = PCA(n_components=1)  # Use for plotting x

    # Fit PCA objects for x and y, and transform datasets
    Z_y = pca.fit_transform(np.vstack((pred, true)))  # Transformed y

    # Get real and interpolated_replay split point
    N = Z_y.shape[0]
    s = N // 2

    # Extract interpolated_replay and real data
    interpolated_points = Z_y[s:]
    real_points = Z_y[:s]

    # Now create scatter plot to plot real vs. true
    plt.scatter(interpolated_points, real_points)
    if env is not None:
        plt.title(
            "Holdout {} PCA of Interpolated (y) vs. True (x) Points".format(env))
    else:
        plt.title("Holdout PCA of Interpolated (y) vs. True (x) Points")
    plt.xlabel("PCA (d=1) of Linearly Interpolated Outputs")
    plt.ylabel("PCA (d=1) of GPR Interpolated Outputs")
    plt.show()


def pca_1D(synthetic_batch, real_batch, sample_indices=None,
           neighbor_indices=None, env=None):
    """Scatterplot utility for displaying samples using Principal Component
     Analysis.  Calculates a 1D projection of the inputs using PCA and then
     plots this data on the x-axis, along with a 1D projection of the outputs
     that is plotted along the y-axis.

     Arguments:
         synthetic_batch (SampleBatch): Wrapped dictionary containing
            experience information that will be used for plotting the
            synthetically-generated data points.
         real_batch (SampleBatchpl): Wrapped dictionary containing
            experience information that will be used for plotting the
            real data points.
        sample_indices (np.array): Array of indices for which to take slice of
            for the real dataset.  If None, takes the whole dataset.
        neighbor_indices (np.array): Array of indices for which to take slice of
            for the real dataset.  If None, takes the whole dataset.
        env (str): Name of the environment we are performing PCA on.  Defaults
            to None.
     """
    # Get observations, actions, rewards, and next states from samples
    X_syn = np.hstack((check_dims(synthetic_batch["obs"]),
                       check_dims(synthetic_batch["actions"])))
    Y_syn = np.hstack((check_dims(synthetic_batch["rewards"]),
                       check_dims(synthetic_batch["new_obs"])))
    X_real = np.hstack((check_dims(real_batch["obs"]),
                        check_dims(real_batch["actions"])))
    Y_real = np.hstack((check_dims(real_batch["rewards"]),
                        check_dims(real_batch["new_obs"])))

    # Check if we take a slice of the real dataset
    idx = np.array([]).astype(np.int32)
    if sample_indices is not None:
        idx = np.concatenate((idx, sample_indices))
    if neighbor_indices is not None:
        idx = np.concatenate((idx, neighbor_indices))

    # Now take indices
    if (sample_indices is not None) or (neighbor_indices is not None):
        X_real = X_real[idx, :]  # Take slice corresponding to x
        Y_real = Y_real[idx, :]  # Take slice corresponding to y

    # Use PCA to project feature matrices into two dimensions
    pca_x = PCA(n_components=1)  # Use for plotting x
    pca_y = PCA(n_components=1)  # Use for heat map color

    # Fit PCA objects for x and y, and transform datasets
    Z_x = pca_x.fit_transform(np.vstack((X_syn, X_real)))  # Transformed x
    Z_y = pca_y.fit_transform(np.vstack((Y_syn, Y_real)))  # Transformed y

    # Get real and interpolated_replay split point
    N = Z_x.shape[0]
    if (sample_indices is not None) and (neighbor_indices) is not None:
        s = N // 3
    else:
        s = N // 2

    # Sort the array indices
    idx_sorted = [i[0] for i in sorted(enumerate(Z_x[s:]), key=lambda x:x[1])]

    # Extract real and interpolated_replay data for creating scatter plots
    x_interpolated, y_interpolated = Z_x[:s], Z_y[:s]
    x_real, y_real = Z_x[s:], Z_y[s:]
    x_real_sorted, y_real_sorted = Z_x[s:][idx_sorted], Z_y[s:][idx_sorted]

    # Now create scatterplots - real --> o, interpolated_replay --> v
    plt.scatter(x_interpolated, y_interpolated, marker="v",
                label="Interpolated", color="g")
    plt.scatter(x_real, y_real, marker="o", label="Real", color="b")

    # Plot the real data in a sequential line using ordered points
    plt.plot(x_real_sorted, y_real_sorted, label="Real (Line)", color="r")
    if env is not None:
        plt.title(
            "{} PCA (dx = 1, dy = 1) of Interpolated/Real Points".format(env))
    else:
        plt.title("PCA (dx = 1, dy = 1) of Interpolated/Real Points")
    plt.legend()
    plt.show()
