"""Test script for training batched Gaussian Process Regression models using
BoTorch. Specifically, this script uses BoTorch to predict how well a sine
function can be approximated in a batched, multi-dimensional setting. Note that
each dimension of X and corresponding dimension of Y is a direct sine mapping.

To adjust the parameters to observe the effect of modifying the batch size,
number of training samples, or dimensions/distribution of the features/targets,
please adjust the parameters at the top of the main function.
"""

# External Python packages
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
import matplotlib.pyplot as plt
import numpy as np

# Native Python packages
import time

# Custom Python packages/modules
from utils.gpytorch.gpytorch_utils import standardize, standardize_manual, \
    unstandardize

def main():
    """Main scripting function for analytic testing of Gaussian Process
     Regression (GPR) with the BoTorch library."""
    # Specify dimensions as parameters
    B = 1  # "batch" dimension
    N = 1000  # "num points" dimension
    Dx = 2  # "num features" dimension
    SIGMA = 5  # Standard deviation of dataset

    # Create features, and normalize them
    X = SIGMA * torch.rand(B, N, Dx)
    X, x_std, x_mu = standardize(X)

    # Create targets from normalized features
    Y = torch.sin(X)

    # Everything to GPU, if available
    if torch.cuda.is_available():
        x_std = x_std.cuda()
        x_mu = x_mu.cuda()

    # Create test data and standardize it with moments of x
    test_x = SIGMA * torch.rand(B, N, Dx).cuda()  # "Perturb" with high variance
    test_x = standardize_manual(test_x, x_std, x_mu)  # Standardize perturbation

    # Specify likelihood, model, and mll objective
    shape = torch.Size([B])
    likelihood = GaussianLikelihood(batch_shape=shape)
    base_kernel = MaternKernel(batch_shape=shape, ard_num_dims=Dx)
    gp = SingleTaskGP(X, Y, likelihood=likelihood, covar_module=base_kernel)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    # Put model and mll objective on cuda, if available
    if torch.cuda.is_available():
        mll = mll.cuda()
        gp = gp.cuda()

    # Train using BoTorch training function
    time_start = time.time()
    fit_gpytorch_model(mll)  # BoTorch training
    time_end = time.time()
    print("TIME TO TRAIN: {}".format(time_end - time_start))

    # Make predictions/perform inference on test dataset
    gp.eval()  # Places GP model in posterior mode
    with torch.no_grad():
        preds = gp.likelihood(gp(test_x))

    # Extract data to plot, and plot results
    plot_dim = 0  # Indicates dimension to visualize
    plot_x = test_x.cpu().numpy()[0, :, plot_dim]  # From first batch
    plot_y = preds.mean[0, ...].cpu().numpy().T[:, plot_dim]  # From first batch

    # Scatter plot of predicted data
    plt.scatter(plot_x, plot_y, color="r", label="Predicted")

    # Scatter plot of analytic data
    x_ordered = sorted(plot_x)  # Create ordered set from plotted features
    plt.plot(x_ordered, np.sin(x_ordered), color="b", label="Analytic")

    # Generate figure
    plt.xlabel("Features, Dim {}".format(plot_dim))
    plt.ylabel("Targets, Dim {}".format(plot_dim))
    plt.title("Learning an Analytic Sine Function Using BoTorch")
    plt.legend()
    plt.show()

    # Display error
    y_hat = torch.transpose(preds.mean, 2, 1).cpu().numpy()
    y_true = np.sin(test_x.cpu().numpy())
    mse = np.mean(np.square(np.linalg.norm(y_true - y_hat, ord=2, axis=-1)))
    print("Average Test MSE Error: {}".format(mse))


if __name__ == "__main__":
    main()
