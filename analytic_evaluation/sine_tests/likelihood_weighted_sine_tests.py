"""Test script for evaluating likelihood weighting with Gaussian Process
Regression. By training and testing a GPR model on sinusoidal data, likelihoods
can be calculated at the mean predicted points using the posterior estimated
variance values at the test points.

To adjust the parameters to observe the effect of modifying the batch size,
number of training samples, or dimensions/distribution of the features/targets,
please adjust the parameters at the top of the main function.
"""
# External Python packages
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

# Native Python packages
import time

# Custom Python packages/modules
from utils.gpytorch.gpytorch_utils import gpr_train_batch, preprocess_eval_inputs, \
    format_preds
from utils.execution.performance_utils import determine_device_torch


def main():
    """Main tester function."""
    # Set parameters
    B = 256  # "Batch size"
    N = 100  # Number of data points in each batch
    D = 3  # Dimension of features and targets
    EPOCHS = 5  # Number of training iterations of GPyTorch training
    USE_CUDA = torch.cuda.is_available()  # Whether to use GPU
    MEAN = 0  # Mean of input data
    SCALE = 1  # Standard deviation of input data
    COMPOSITE_KERNEL = False  # Whether to use a composite GPR kernel
    USE_ARD = True  # Whether to use Automatic Relevance Determination (ARD)
    LR = 0.5  # Learning rate for GPR

    # Specify device
    device = determine_device_torch()  # Get device

    # Create training data and targets
    train_x_np = np.random.normal(loc=MEAN, scale=SCALE, size=(B, N, D))
    train_x = torch.tensor(train_x_np, device=device)
    train_y = torch.sin(train_x)

    # GPyTorch training
    model, likelihood = gpr_train_batch(
        train_x, train_y, use_cuda=USE_CUDA, epochs=EPOCHS, lr=LR,
        thr=-math.inf, use_ard=USE_ARD, composite_kernel=COMPOSITE_KERNEL,
        kernel="rbf", mean_type="zero", global_hyperparams=False,
        model_hyperparams=None, est_lengthscales=False, fp64=True)

    # Place model and likelihood in posterior mode
    model.eval()
    likelihood.eval()

    # Create test features for prediction
    test_x_np = np.random.normal(loc=MEAN, scale=SCALE, size=(B, 1, D))
    test_x = preprocess_eval_inputs(test_x_np, train_y_np.shape[-1])

    # Place on CUDA, if available
    if USE_CUDA:
        test_x = test_x.cuda()

    # Make predictions without gradients
    with torch.no_grad():

        # Predict on test features
        observed_pred = likelihood(model(test_x))  # Forward inference
        mean = observed_pred.mean  # Get predicted mean at test_x
        variance = observed_pred.variance  # Get predicted variance at test_x

        # Reformat variance block
        variance_block = format_preds(
            variance, B, single_model=False)

        # Now compute the predicted likelihood of the predicted mean
        # (i) Compute the likelihoods as a block of shape (B, 1, D)
        likelihoods_block = torch.reciprocal(
            torch.sqrt(2 * np.pi * variance_block))

        # (ii) Compute log likelihoods of block for numerical stability
        log_likelihoods_block = torch.log(likelihoods_block)

        # (iii) Take sum along samples to take product in log domain
        weights_log = torch.sum(torch.tensor(log_likelihoods_block), dim=1)

        # (iv) Exponentiate to compute joint squashed likelihood
        weights = torch.exp(weights_log)

        # (v) Normalize and clip
        # Compute normalizing factor and scale
        normalizing_factor = B / torch.sum(weights)
        scaled_weights = (weights * normalizing_factor).cpu().numpy()

    # Now plot
    if D > 1:  # Greater than one dimension
        plt.hist(scaled_weights)
        plt.show()

    for DIM, c in zip([0, 1, 2], ["r", "g", "b"]):
        # Compute mean and unstack
        output = mean.cpu().detach().numpy()
        out_y = np.squeeze(np.array([output[i::B] for i in range(B)]))

        # Reformat, get analytic y, and plot
        x_plot = np.squeeze(test_x_np)
        y_plot_gt = np.sin(x_plot)
        plt.scatter(x_plot[:, DIM], out_y[:, DIM])
        plt.title("Test X vs. Predicted Y, Dimension {}".format(DIM))
        plt.show()
        plt.clf()

        # Creating color map
        my_cmap_analytic = plt.get_cmap('hot')
        my_cmap_predicted = plt.get_cmap('cool')

        # Plot ground truth
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x_plot[..., 0], x_plot[..., 1], y_plot_gt[..., DIM],
                        antialiased=True, cmap=my_cmap_analytic)
        plt.title("Analytic, Ground Truth Surface, Dimension {}".format(DIM))
        plt.show()
        plt.clf()

        # Plot the predicted values
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x_plot[..., 0], x_plot[..., 1], out_y[..., DIM],
                        antialiased=True, cmap=my_cmap_predicted)
        plt.title("Predicted Surface, Dimension {}".format(DIM))
        plt.show()
        plt.clf()

    # Compute average RMSE
    y_true = np.squeeze(np.sin(test_x_np))
    y_pred = out_y
    rmse = np.sqrt(mse(y_true, y_pred))
    print("RMSE: {}".format(rmse))
    return rmse


if __name__ == '__main__':
    all_rmses = []
    for i in range(1):
        all_rmses.append(main())
    print("AVERAGED RMSE: {}".format(np.mean(all_rmses)))
