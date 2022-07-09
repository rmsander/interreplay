"""Utilities for running analytic evaluation on environments defined for
robotics and reinforcement learning, such as OpenAI Pendulum-v2.

Specifically, these defined functions include utilities for:

1. Generating datasets:

    a. Clustered datasets
    b. Uniformly-sampled datasets
    c. Linearly/regularly-spaced datasets

2. Feature and target construction for the Pendulum environment:
    a. Analytic feature construction: [cos(th), sin(th), thdot, action]
    b. Analytic feature computation

3. Plotting utilities for plotting scatterplots in 3D, and heatmaps in 2D.
"""
# External Python packages
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets.samples_generator import make_blobs


def generate_cluster_data_pendulum(N, bounds, n_blobs=50, seed=42, std=1.0):
    """Function for generating cluster data for simulating "neighborhood" models
    with our GPR.

    Clusters are sampled within the bounds ranges in each dimension

    Parameters:
        N (int): The number of samples we generate using clusters. Note that
            this is separate from the number of clusters.
        bounds (list): List of tuples corresponding to numerical bounds to
            constrain the state and action space over the environment.
        n_blobs (int): The number of blobs (clusters) to generate. Defaults
            to 50. Note this is separate from number of points we generate.
        seed (int): Random seed used for generating clusters. Defaults to 42.
        std (float): High standard deviation value. Defaults to 1.0.

    Returns:
        X (np.array): NumPy array of points corresponding to input for
            constructing features (X_obs) and reward (Y) for GPR.
            Bounds span the state and action space of the Pendulum environment.
        Y (np.array): NumPy array of targets corresponding to the targets for
            GPR. This spans the reward space of the Pendulum environment.
        X_obs (np.array): NumPy array of features corresponding to the targets
            for GPR. This spans the observation/action space of the Pendulum
            environment.
    """
    # Extract bounds for sampling points in space
    [(low_s1, high_s1), (low_s2, high_s2), (low_a, high_a)] = bounds

    # Construct points for cluster centers/actions using random uniform sampling
    S1 = np.random.uniform(low=low_s1, high=high_s1, size=n_blobs)  # State dim1
    S2 = np.random.uniform(low=low_s2, high=high_s2, size=n_blobs)  # State dim2
    A = np.random.uniform(low=low_a, high=high_a, size=n_blobs)  # Action dim

    # Take cluster points as centers and sample cluster standard deviations
    centers = [(s1, s2, a) for s1, s2, a in zip(S1, S2, A)]
    cluster_std = np.random.uniform(low=0.1, high=std, size=n_blobs)

    # Create the blobs using the defined clusters
    X, y = make_blobs(n_samples=N, cluster_std=cluster_std, centers=centers,
                      n_features=3, random_state=seed)  # Features for GPR

    # Placeholder for reward
    Y = np.zeros(X.shape[0])  # Targets for GPR

    # Clip to ensure all observations fall within observation/action space
    lower_bounds = [low_s1, low_s2, low_a]
    upper_bounds = [high_s1, high_s2, high_a]
    X = np.clip(X, lower_bounds, upper_bounds)

    # Iterate over generated samples to compute analytic reward
    for k in range(X.shape[0]):
        Y[k] = compute_reward_pendulum(X[k])  # Compute analytic reward

    # Construct features from inputs
    X_obs = construct_features_pendulum(X)

    return X, Y, X_obs


def sample_uniformly_pendulum(N, bounds):
    """Utility function for sampling points i.i.d ~ U(bounds) from 3D space.

    Note this function does not sample from clusters as in the function above.
    This function can be used to evaluate analytical GPR performance.

    Parameters:
        N (int): The cubic root of the number of points to sample in each
            dimension, e.g. if N = 10, then 10^3 = 1000 points will be sampled.
        bounds (list): List of tuples corresponding to numerical bounds to
            constrain the state and action space over the environment.

    Returns:
        X (np.array): NumPy array of points corresponding to input for
            constructing features (X_obs) and reward (Y) for GPR.
            Bounds span the state and action space of the Pendulum environment.
        Y (np.array): NumPy array of targets corresponding to the targets for
            GPR. This spans the reward space of the Pendulum environment.
        X_obs (np.array): NumPy array of features corresponding to the targets
            for GPR. This spans the observation/action space of the Pendulum
            environment.
    """
    # Extract bounds for sampling points in space
    [(low_s1, high_s1), (low_s2, high_s2), (low_a, high_a)] = bounds
    N_cubed = N ** 3  # Number of points in each dimension

    # Now sample each dimension uniformly, i.i.d.
    s1 = np.random.uniform(low=low_s1, high=high_s1, size=N_cubed)
    s2 = np.random.uniform(low=low_s2, high=high_s2, size=N_cubed)
    a = np.random.uniform(low=low_a, high=high_a, size=N_cubed)

    # Now combine stacked, randomly-sampled features
    X = np.stack((s1, s2, a), axis=-1)
    Y = np.zeros(N_cubed)

    # Iterate over generated samples to compute analytic reward
    for k in range(N_cubed):
        Y[k] = compute_reward_pendulum(X[k])  # Compute analytic reward

    # Construct features from inputs
    X_obs = construct_features_pendulum(X)

    return X, Y, X_obs


def generate_lin_space_samples_pendulum(N, bounds):
    """Function for generating samples from a linearly-spaced interval.

    Note this function does not sample from clusters as in the function above.
    This function can be used to evaluate analytical GPR performance.

    Parameters:
        N (int): The cubic root of the number of points to sample in each
            dimension, e.g. if N = 10, then 10^3 = 1000 points will be sampled.
        bounds (list): List of tuples corresponding to numerical bounds to
            constrain the state and action space over the environment.

    Returns:
        X (np.array): NumPy array of points corresponding to input for
            constructing features (X_obs) and reward (Y) for GPR.
            Bounds span the state and action space of the Pendulum environment.
        Y (np.array): NumPy array of targets corresponding to the targets for
            GPR. This spans the reward space of the Pendulum environment.
        X_obs (np.array): NumPy array of features corresponding to the targets
            for GPR. This spans the observation/action space of the Pendulum
            environment.
    """
    # Extract bounds for sampling points in space
    [(low_s1, high_s1), (low_s2, high_s2), (low_a, high_a)] = bounds

    # Get step sizes over linear space according to bounds and number of samples
    step_s1 = (high_s1 - low_s1) / N
    step_s2 = (high_s2 - low_s2) / N
    step_a = (high_a - low_a) / N

    # Create regularly-spaced grid using the step sizes for each dimension
    X = np.mgrid[low_s1:high_s1:step_s1,
        low_s2:high_s2:step_s2,
        low_a:high_a:step_a].reshape(3, -1).T

    # Iterate over generated samples to compute analytic reward
    N_cubed = N ** 3
    Y = np.zeros(N_cubed)
    for k in range(N_cubed):
        Y[k] = compute_reward_pendulum(X[k])  # Compute analytic reward

    # Construct features from inputs
    X_obs = construct_features_pendulum(X)

    return X, Y, X_obs


def generate_datasets_pendulum(n_train, n_test, bounds,
                               dataset_types=["uniform", "uniform"],
                               num_batches=1, seed=42, std=1.0):
    """Utility to function to create datasets for analytic_evaluation.

    Specifically, this function is used to generate training and testing
    datasets to evaluate the in-sample and out-of-sample function approximation
    capabilities of GPR. These results can give insight into the performance
    of GPR models in other RL environments/settings.

    Parameters:
        n_train (int): The number of samples to generate for the training set.
        n_test (int): The number of samples to generate for the training set.
        bounds (list): List of tuples corresponding to numerical bounds to
            constrain the state and action space over the environment.
        dataset_types (list): A list of strings corresponding to the types of
            datasets generated for training and analytic_evaluation,
            respectively. Defaults to ["uniform", "uniform"]. Other options
            include "cluster", for clustered datasets, and "grid", for regular
            grid datasets.
        num_batches (int): The number of batches used for training this GPR
            model in a batched setting with GPyTorch. Defaults to 1.
        seed (int): Random seed used for generating clusters. Defaults to 42.
        std (float): High standard deviation value for generating cluster data.
            Defaults to 1.0; not relevant unless cluster data is used.

    Returns:
        D_train (tuple): A tuple consisting of X_train, Y_train, and X_obs_train.
            For information on these arrays, please see the docstrings in the
            dataset generation functions. This corresponds to the training
            dataset for the GPR model. Note that X_train, Y_train, and
            X_obs_train are all PyTorch tensors.
        D_test (tuple): A tuple consisting of X_test, Y_test, and X_obs_test.
            For information on these arrays, please see the docstrings in the
            dataset generation functions. This corresponds to the testing
            dataset for the GPR model. Note that X_test, Y_test, and
            X_obs_test are all PyTorch tensors.
    """
    # Create training and testing datasets
    datasets = []
    num_points = [n_train, n_test]
    for n, t in zip(num_points, dataset_types):
        if t == "uniform":
            datasets.append(sample_uniformly_pendulum(n, bounds))
        elif t == "cluster":
            datasets.append(generate_cluster_data_pendulum(
                n, bounds, n_blobs=num_batches, seed=seed, std=std))
        elif t == "grid":
            datasets.append(generate_lin_space_samples_pendulum(n, bounds))

    # Get device depending on CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'

    # Get training and analytic_evaluation datasets
    (X_train, Y_train, X_obs_train) = datasets[0]
    (X_test, Y_test, X_obs_test) = datasets[1]

    # Preprocess train data as torch tensors
    X_train_np_shape = (num_batches,) + (X_train.shape[0] // num_batches,
                                        X_train.shape[1])
    X_train_np = X_train.reshape(X_train_np_shape).astype(np.float64)
    X_train = torch.tensor(X_train_np, device=device).double()

    # Preprocess train observations as torch tensors
    X_obs_train_np_shape = (num_batches,) + (X_obs_train.shape[0] // num_batches,
                                            X_obs_train.shape[1])
    X_obs_train_np = X_train.reshape(X_obs_train_np_shape).astype(np.float64)
    X_obs_train = torch.tensor(X_obs_train_np, device=device).double()

    # Preprocess train targets as torch tensors
    Y_train_np_shape = (num_batches,) + (Y_train.shape[0] // num_batches,)
    Y_train_np = Y_train.reshape(Y_train_np_shape).astype(np.float64)
    Y_train = torch.tensor(Y_train_np, device=device).double()
        
    # Preprocess test data as torch tensors
    X_test_np_shape = (num_batches,) + (X_test.shape[0] // num_batches, 
                                        X_test.shape[1])
    X_test_np = X_test.reshape(X_test_np_shape).astype(np.float64)
    X_test = torch.tensor(X_test_np, device=device).double()
    
    # Preprocess test observations as torch tensors
    X_obs_test_np_shape = (num_batches,) + (X_obs_test.shape[0] // num_batches, 
                                            X_obs_test.shape[1])
    X_obs_test_np = X_test.reshape(X_obs_test_np_shape).astype(np.float64)
    X_obs_test = torch.tensor(X_obs_test_np, device=device).double()
    
    # Preprocess test targets as torch tensors
    Y_test_np_shape = (num_batches,) + (Y_test.shape[0] // num_batches,)
    Y_test_np = Y_test.reshape(Y_test_np_shape).astype(np.float64)
    Y_test = torch.tensor(Y_test_np).double()

    return (X_train, Y_train, X_obs_train), (X_test, Y_test, X_obs_test)


def angle_normalize(th):
    """Utility function to normalize an angle between [-pi, pi].

    Parameters:
        th (float): Float corresponding to an angle to be normalized to range
            [-pi, pi].

    Returns:
        normalized_th (float): Float corresponding to angle normalized to range
            [-pi, pi].
    """
    return ((th + np.pi) % (2 * np.pi)) - np.pi


def compute_reward_pendulum(x):
    """Analytic function to compute reward given states and action.

    Parameters:
        x (np.array/list): Array of shape (3,) corresponding to a single input
            for a state-action vector. This state-action vector determines the
            analytic reward for the Pendulum environment.

    Returns:
        reward (float): Analytic reward from the Pendulum-v2 environment.
    """
    # Get components of state-action vector
    th, thdot, u = x

    # Compute analytic reward
    reward = -(angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2))
    return reward


def compute_reward_pendulum_batched(X):
    """Analytic function to compute reward given states and action in a batched
    setting. This function is used when batch sizes > 1 are allowed.

    Parameters:
        X (np.array): Array of shape (N, 3) corresponding to a single batch
            of state-action vectors. This state-action vector determines the
            analytic reward for the Pendulum environment in a batched setting.

    Returns:
        reward (np.array): Batched analytic reward from the Pendulum-v2
            environment. Has shape (N,).
    """
    # Create placeholder for reward
    Y = np.zeros(X.shape[:-1])

    # Compute reward for all elements in a batch
    for i, x in enumerate(X):
        th, thdot, u = np.squeeze(x)
        Y[i] = -(angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2))
    return Y


def construct_features_pendulum(X):
    """Helper function to construct the analytic observations from a set of
    inputs. Specifically, computes the observations according to the
    observations for the Pendulum-v2 environment.

    Parameters:
        X (np.array): NumPy array of points corresponding to input for
            constructing features (X_obs) and reward (Y) for GPR.
            Bounds span the state and action space of the Pendulum environment.

    Returns:
        X_obs (np.array): NumPy array of features corresponding to the targets
            for GPR. This spans the observation/action space of the Pendulum
            environment.
    """
    # Get observations (features for GPR) from input points
    X1, X2, A = X[:, 0], X[:, 1], X[:, 2]  # Separate features
    X_cos = np.squeeze(np.cos(X1))  # Cosine of angular position
    X_sin = np.squeeze(np.sin(X1))  # Sine of angular position
    X_thdot = np.squeeze(X2)  # Derivative of angular position
    X_a = np.squeeze(A)  # Action

    # Construct observations as stacked features
    X_obs = np.stack((X_cos, X_sin, X_thdot, X_a), axis=-1)
    return X_obs


def plot_points(X, Y, title="Analytic Reward Over State-Action Space"):
    """Utility function for plotting points. Plots as a 3D scatter plot.

    Parameters:
        X (np.array): NumPy array of points corresponding to input for
            constructing features (X_obs) and reward (Y) for GPR.
            Bounds span the state and action space of the Pendulum environment.
        Y (np.array): NumPy array of targets corresponding to the targets for
            GPR. This spans the reward space of the Pendulum environment.
        title (str): The title of the plot when constructing this figure.
    """
    # Create 3D figure and labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Theta (th)')
    ax.set_ylabel('Theta dot (thdot)')
    ax.set_zlabel('Action (u)')

    # Add 3D scatter plot using data
    x1, x2, a = X[:, 0], X[:, 1], X[:, 2]
    img = ax.scatter(x1, x2, a, c=Y, cmap=plt.hot())
    fig.colorbar(img, orientation="horizontal", pad=0.03)
    fig.tight_layout()
    plt.title(title)
    plt.show()
    plt.clf()


def plot_predictions(X_test, Y_test, Y_pred):
    """Utility to function to plot predictions and true (analytic) values.

    Parameters:
      X_test (np.array): NumPy array of test features. Should be of shape
        (N, Dx), where N is the number of points, and Dx is the dimension of the
        features.
      Y_test (np.array): NumPy array of test features. Should be of shape
        (N, Dy), where N is the number of points, and Dy is the dimension of the
        targets. This the ground truth array of analytic test targets.
      Y_pred (np.array): NumPy array of predicted features. Should be of shape
        (N, Dy), where N is the number of points, and Dy is the dimension of the
        targets. This is the predictions array of test targets.
    """
    # Plot the true analytic values
    plot_points(X_test, Y_test, title="Analytic Reward Over State-Action Space")

    # Plot predictions
    plot_points(X_test, Y_pred, title="Predicted Reward Over State-Action Space")
