"""A set of utility functions for performing and assessing analytic validation
of different interpolation schemes on analytic functions. These analytic
functions include sine functions, test functions such as Ackley and Rastrigin
functions, and reward functions from RL environments."""

# External Python packages
import torch
from torch.distributions import MultivariateNormal
import gpytorch
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models import SingleTaskGP
from botorch.test_functions.synthetic import Rosenbrock, Ackley, Griewank, \
    Rastrigin
import numpy as np
import matplotlib.pyplot as plt

# Native Python packages
import math
import warnings
warnings.filterwarnings("ignore")

# Custom Python packages/modules
from utils.gpytorch.gpytorch_utils import gpr_train_batch, unstandardize


def create_synthetic_normal(N, D):
    """Utility to function to create N samples of a D-dimensional normal
    distribution.

    NOTE: The dimensions of these MVN samples are independent of one another
          (note the diagonal covariance).

    Parameters:
        N (int): The number of samples to create for the normal dataset.
        D (int): The dimensionality of the samples for the dataset.

    Returns:
        X (torch.Tensor): Tensor representing samples drawn from a MVN
            distribution. Has shape (1, N, D), and samples across dimensions
            are independent from one another.
    """
    # Sample mean and diagonal covariance
    mean = torch.rand(D)
    cov = torch.diag(torch.rand(D))

    # Draw MVN samples and add dimension such that X has dimensions (1, N, D)
    X = torch.normal(mean, cov, size=(N, D)).unsqueeze(0)
    return X


def create_and_sample_normal(N, mean, std):
    """Utility to function to create N samples of a D-dimensional normal
    distribution.

    NOTE: The dimensions of these MVN samples are independent of one another
          (note the diagonal covariance).

    Parameters:
        N (int): The number of samples to create for the normal dataset.
        mean (torch.Tensor): A vector of shape (D, ) corresponding to the
            means of the MVN distribution we sample from.
        std (torch.Tensor): A vector of shape (D, ) corresponding to the
            diagonals of the covarianc of the MVN distribution we sample from.
            Note the covariance of the MVN is diagonal by default.

    Returns:
        X (torch.Tensor): Tensor representing samples drawn from a MVN
            distribution. Has shape (1, N, D), and samples across dimensions
            are independent from one another.
        """
    # Create MVN distribution using means and diagonal constructed from std
    D = MultivariateNormal(mean, torch.diag(std))

    # Sample from MVN
    samples = D.sample((N, ))  # Returns (N, D)
    return samples


def generate_mean_and_std(D):
    """Utility function to generate means and standard deviations for datasets.

    Parameters:
        D (int): The dimensionality of the data for the samples/datasets
            generated.

    Returns:
        mean (torch.Tensor): Tensor object corresponding to the set of means
            for the dataset. Expected mean value is 0.
        std (torch.Tensor): Tensor object corresponding to the set of standard
            deviations for the dataset. Note these values > 0.
    """
    mean = torch.rand(D) - 0.5  # Mean zero, since torch.rand ~ U([0, 1])
    std = torch.rand(D)  # Strictly non-negative
    return mean, std


def create_function_data(shape, mean, std, fn=Ackley, Dy=1):
    """Create analytic test function datasets to test GPyTorch/BoTorch models.

    Parameters:
        shape (tuple): Tuple of dimensions given by (N, D), where N is the
            number of samples, and D is the number of dimensions.
        mean (torch.Tensor): Tensor object corresponding to the set of means
            for the dataset. Expected mean value is 0.
        std (torch.Tensor): Tensor object corresponding to the set of standard
            deviations for the dataset. Note these values > 0.
        fn (optimization_test_functions.test_functions.synthetic.<Fn>): A
            BoTorch test function object used for producing an analytic value
            for Y that can be compared directly to predicted values to assess
            function approximation performance.
        Dy (int): The dimensionality of the targets. Defaults to 1.

    Returns:
        X (torch.Tensor): Tensor corresponding to the features used for
            Gaussian Process Regression. Has shape (1, N, D), and is produced
            synthetically from a MVN distribution. The first dimension is a
            batch dimension.
        Y (torch.Tensor): Tensor corresponding to the targets used for
            Gaussian Process Regression. Has shape (N, Dy), and is produced
            by evaluating the features as inputs to the given test function fn.
    """
    # Create the inputs from a sampled normal distribution
    N, D = shape
    X = create_and_sample_normal(N, mean, std).unsqueeze(dim=0)  # (1, N, D)

    # Compute the outputs according to the test function
    if fn == Ackley:
        func = fn(dim=2, noise_std=None, negate=False)
    else:
        func = fn(noise_std=None, negate=False)

    # Compute the outputs according to the dimension of the targets
    if Dy > 1 and (fn in [Ackley, Griewank, Rosenbrock, Rastrigin]):
        Y = torch.stack([func.evaluate_true(X) for i in range(Dy)], dim=-1)
    else:
        Y = func.evaluate_true(X).unsqueeze(dim=-1)

    return X, Y


def plot_data(x_plot, y_plot):
    """Utility function to plot data in 3D.

    Parameters:
        x_plot (np.array): Array corresponding to input features. Has shape
            (N, 2), i.e. two dimensions.
        y_plot (np.array): Array corresponding to output features. Has shape
            (N, ), i.e. a single dimension.
    """
    # Create heat map
    my_cmap_analytic = plt.get_cmap('hot')

    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate heat map plot and plot figure
    ax.plot_trisurf(x_plot[:, 0], x_plot[:, 1], y_plot, cmap=my_cmap_analytic)
    plt.show()


def plot_evaluation(x, y_true, y_pred, function_name="Ackley"):
    """Generate comparative analytic and predicted surfaces for the given
    test function evaluated on a generated dataset. Used for qualitatively
    evaluating the quality of interpolation/function approximation.

    Parameters:
        x (np.array): Array of features for which there exist corresponding
            predicted and analytic values predicted/evaluated at these
            points in feature space.
        y_true (np.array): Array of true analytic targets evaluated at x.
        y_pred (np.array): Array of predicted targets predicted at x.
        function_name (str): Name of the test function that maps features x to
            analytic targets y_pred.
    """
    # Creating color map for analytic surfaces and predictions
    my_cmap_analytic = plt.get_cmap('gnuplot')
    my_cmap_predicted = plt.get_cmap('cool')

    # Plot analytic ground truth
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_true[0, :, 0], x[0, :, 1], y_true[0, :, 0],
                    antialiased=True, cmap=my_cmap_analytic)
    plt.title("Test Function {}, Analytic".format(n))
    plt.savefig("test_function_{}_analytic.png".format(n))
    plt.show()
    plt.clf()

    # Plot the predicted values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_true[0, :, 0], x[0, :, 1], y_pred[:, 0],
                    antialiased=True, cmap=my_cmap_predicted)
    plt.title("Test Function {}, Predicted".format(function_name))
    plt.savefig("test_function_{}_predicted.png".format(function_name))
    plt.clf()


def botorch_create_model(X, Y):
    """Function for running Gaussian Process Regression training via BoTorch.

    NOTE: Training is run in BoTorch, rather than GPyTorch.

    Parameters:
        X (torch.Tensor): Tensor corresponding to the features used for
            Gaussian Process Regression. Expects shape (B, N, Dx), where B is
            the batch size, N is the number of samples, and Dx is the is the
            dimension of the features.
        Y (torch.Tensor): Tensor corresponding to the targets used for
            Gaussian Process Regression. Expects shape (B, N, Dx), where B is
            the batch size, N is the number of samples, and Dx is the is the
            dimension of the features.
    """
    shape = torch.Size(X.size()[0])
    likelihood = GaussianLikelihood(batch_shape=shape)
    kernel = MaternKernel(batch_shape=shape, ard_num_dims=Dx)
    gpr_model = SingleTaskGP(X, Y, likelihood=likelihood, base_kernel=kernel)
    mll = ExactMarginalLogLikelihood(gpr_model.likelihood, gpr_model)

    # Send model/mll to cuda, if available
    if torch.cuda.is_available():
        mll = mll.cuda()
        gpr_model = gpr_model.cuda()

    # Begin timing and fit model
    time_start = time.time()
    fit_gpytorch_model(mll)  # BoTorch model fit function
    time_end = time.time()
    print("TIME TO TRAIN: {}".format(time_end - time_start))
    return gpr_model


def create_model(X, Y, epochs):
    """Wrapper for training a GPyTorch batched model using defined GPyTorch
    functions for training batched GPR models.

    Parameters:
        X (torch.Tensor): Tensor corresponding to the features used for
            Gaussian Process Regression. Expects shape (B, N, Dx), where B is
            the batch size, N is the number of samples, and Dx is the is the
            dimension of the features.
        Y (torch.Tensor): Tensor corresponding to the targets used for
            Gaussian Process Regression. Expects shape (B, N, Dx), where B is
            the batch size, N is the number of samples, and Dx is the is the
            dimension of the features.
        epochs (int): The number of training epochs.

    Returns:
        model (BatchedGP): A GPR model of BatchedGP type with which to generate
            synthetic predictions of rewards and next states.
    """
    # Train the GPR model using batched GPyTorch training
    use_cuda = torch.cuda.is_available()
    gpr_model, _ = gpr_train_batch(
        X, Y, use_cuda=use_cuda, epochs=epochs, lr=0.1, thr=-math.inf,
        use_ard=True, composite_kernel=False, kernel="rbf", mean_type="zero",
        global_hyperparams=False, model_hyperparams=None,
        est_lengthscales=False, fp64=True)
    return gpr_model


def evaluate_model(model, shape, Dy, Y_std, Y_mean, X_std, X_mean,
                   standardize=True, fn=Ackley):
    """Function for evaluating a trained model on test data with targets
    analytically defined by a supplied test function.

    Parameters:
        model (BatchedGP): A GPR model of BatchedGP type with which to generate
            synthetic predictions of rewards and next states.
        shape (tuple): Tuple of shape (N, D), where N is the number of samples,
            and D is the dimension of the data.
        Dy (int): The dimension of the targets.
        Y_std (torch.tensor): Standard deviation of the targets across a given
            dimension. This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.
        Y_mean (torch.tensor): Mean of the targets across a given dimension.
            This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.
        X_std (torch.tensor): Standard deviation of the features across a given
            dimension. This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.
        X_mean (torch.tensor): Mean of the features across a given dimension.
            This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.
        standardize (bool): Whether to standardize the features and targets of
            the dataset. Defaults to True.
        fn (optimization_test_functions.test_functions.synthetic.<Fn>):
            A BoTorch test function object used for producing an analytic value
            for Y that can be compared directly to predicted values to assess
            function approximation performance.

    Returns:
        X_test (torch.Tensor): Tensor corresponding to the test features used for
            Gaussian Process Regression. Expects shape (B, N, Dx), where B is
            the batch size, N is the number of samples, and Dx is the
            dimension of the features.
        Y_test (torch.Tensor): Tensor corresponding to the analytic test targets
            used for Gaussian Process Regression. Expects shape (B, N, Dx),
            where B is the batch size, N is the number of samples, and Dx is
            the dimension of the features.
        Y_pred (torch.Tensor): Tensor corresponding to the test targets used for
            Gaussian Process Regression. Expects shape (B, N, Dx), where B is
            the batch size, N is the number of samples, and Dx is the is the
            dimension of the features.
    """
    # Place model in posterior mode
    model.eval()

    # Generate test dataset for evaluating model performance
    X_test, Y_test = create_function_data(shape, X_mean, X_std, fn=fn, Dy=Dy)

    # Make predictions without gradients and LOVE
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        # Inference
        predictions = model.likelihood(model(X_test)).mean.T

        # Unstandardize, if used
        if standardize:
            Y_pred = unstandardize(predictions, Y_std, Y_mean)
        else:
            Y_pred = predictions

    # Compute metrics for dataset (MAE, MSE, RMSE)
    MAE = torch.mean(torch.abs(Y_pred-Y_test))  # Mean Absolute Error
    MSE = torch.mean(torch.square(Y_pred-Y_test))  # Mean Squared Error
    RMSE = torch.sqrt(MSE)  # Root-Mean Squared Error

    # Display metrics
    print("__________________")
    print("MAE: {}".format(MAE))
    print("MSE: {}".format(MSE))
    print("RMSE: {}".format(RMSE))

    return X_test, Y_test, predictions


def extract_lengthscales(model):
    """Utility function for extracting parameters.

    Parameters:
        model (BatchedGP): A GPR model of BatchedGP type with which to generate
            synthetic predictions of rewards and next states.

    Returns:
        lengthscales_np (np.array): Tensor corresponding to the lengthscales of
            the (possibly batched) GPyTorch model.
    """
    # Get detached torch Tensor of lengthscales
    lengthscales_torch = model.covar_module.base_kernel.lengthscale.cpu().detach()

    # Convert to np array and take transpose
    lengthscales_np = np.squeeze(lengthscales_torch).T
    return lengthscales_np
