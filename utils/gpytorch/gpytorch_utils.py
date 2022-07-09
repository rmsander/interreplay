"""Functions and utility functions for creating and training Gaussian Process
Regression objects through GPyTorch. Namely, this module contains functions for:

1. Training batched GPR models (`gpr_train_batch`): Implements GPR training
    for batched applications. Because GPyTorch does not allow for modeling batched,
    multi-output models, each outcome dimension is modeled independently. This
    function can be configured for hyperparameter initialization, instantiation
    of priors, use of a variety of different kernels, and different optimization
    routines (e.g. Adam and L-BFGS.)
2. Training non-batched GPR models (`train_gp_modellist`): Implements GPR
    training for non-batched applications. Considerably slower than the above,
    and impractical for reinforcement learning applications.
3. Standardizing data
4. Pre and post-processing data for use in evaluation/inference.
"""

# External Python Packages
from gpytorch.models import IndependentModelList
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList, \
    MultitaskGaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from botorch.fit import fit_gpytorch_model
import torch
import numpy as np

# Native Python packages
import math
import gc

# Custom packages/modules
from utils.gpytorch.models import BatchedGP, CompositeBatchedGP, MultitaskGPModel
import parameters

def gpr_train_batch(Zs, Ys, epochs=10, lr=0.1, thr=0,
                    use_ard=False, composite_kernel=False, kernel='matern',
                    mean_type='zero', matern_nu=2.5, ds=None, device="cpu",
                    global_hyperparams=False, model_hyperparams=None,
                    use_botorch=False, use_lbfgs=False, use_priors=False,
                    est_lengthscales=False, cluster_heuristic_lengthscales=None,
                    lengthscale_constant=2.0, lengthscale_prior_std=0.1,
                    fp64=True, early_stopping=False, update_hyperparams=False,
                    holdout=False):
    """Fits a batched Gaussian Process Regression (GPR) model using the GPyTorch
     package. Each target dimension is modeled as a single scalar outcome, thus
     allowing for batching over both model batches and target dimensions.

    Specifically, this function is designed to meet the computational needs of
    training and performing inference on a batch of Gaussian Process Regression
    models simultaneously. The steps of this function are outlined below:

    1. The training features (Zs) and targets (Ys) are casted to the appropriate
        precision, reshaped, and tiled.
    2. Models and likelihood objects are created that support batching.
    3. If desired, models and likelihood objects can be initialized with a batch
        set of hyperparameters.
    4. Models and likelihood objects are trained using either Adam or L-BFGS
        on the training set, in order to optimize their hyperparameters.

    Parameters:
        Zs (torch.tensor): Tensor of inputs of expanded shape (B, N, XD), where B is
            the size of the minibatch, N is the number of data points in each
            GP (the number of neighbors we consider in IER), and XD is the
            dimensionality of the state-action space of the environment.
        Ys (torch.tensor): Tesor of predicted values of shape (B, N, YD), where B is the
            size of the minibatch and N is the number of data points in each
            GP (the number of neighbors we consider in IER), and YD is the
            dimensionality of the state-reward space of the environment.
        device (str): The name of the device to assign tensors to. Defaults to
            "cpu".
        epochs (int): The number of epochs to train the batched GPs over.
            Defaults to 10.
        lr (float): The learning rate to use for the Adam optimizer to train
            the batched GPs.
        thr (float): The mll threshold at which to stop training. Defaults to 0.
        use_ard (bool): Whether to use Automatic Relevance Determination (ARD)
            for the lengthscale parameter, i.e. a weighting for each input dimension.
            Defaults to False.
        composite_kernel (bool): Whether to use a composite kernel that computes
            the product between states and actions to compute the variance of y.
            Defaults to False.
        kernel (str): Type of kernel to use for optimization. Defaults to "
            "Matern kernel ('matern'). Other options include RBF
            (Radial Basis Function)/SE (Squared Exponential) ('rbf'), and RQ
            (Rational Quadratic) ('rq').
        mean_type (str): Type of mean function to use for Gaussian Process.
            Defaults to zero mean ('zero'). Other options: linear
            ('linear'), and constant ('constant').")
        matern_nu (float): Value in set if {1/2, 3/2, 5/2} that denotes the power
            to raise the matern kernel evaluation to. Smaller values allow for
            greater discontinuity. Only relevant if kernel is matern. Defaults to
            2.5.
        ds (int): If using a composite kernel, ds specifies the dimensionality of
            the state. Only applicable if composite_kernel is True.
        global_hyperparams (bool): Whether to use a single set of hyperparameters
            over an entire model.
        model_hyperparams (dict): A dictionary of hyperparameters to use for
            initializing a model. Defaults to None.
        use_botorch (bool): Whether to optimize with L-BFGS using Botorch.
        use_lbfgs (bool): Whether to use second-order gradient optimization with
            the L-BFGS PyTorch optimizer. Note that this requires a closure
            function, as defined below.
        use_priors (bool): Whether to use prior distribution over lengthscale and
            outputscale hyperparameters. Defaults to False.
        est_lengthscales (bool): Whether to estimate the lengthscales of each
            cluster of neighborhood by finding the farthest point in each dimension.
            Defaults to False.
        cluster_heuristic_lengthscales (np.array): If computed, an array of clustered
            lengthscales used for estimation.
        matern_nu (float): Value in set if {1/2, 3/2, 5/2} that denotes the power
            to raise the matern kernel evaluation to. Smaller values allow for
            greater discontinuity. Only relevant if kernel is matern. Defaults to
            2.5.
        lengthscale_constant (float): Value which we multiply estimated lengthscales
            by. Defaults to 1.0.
        lengthscale_prior_std (float): Value for the standard deviation of the
            lengthscale prior. Defaults to 0.1.
        fp64 (bool): Whether to perform GPR operations in double-precision, i.e.
            torch.float64. Defaults to True.
        early_stopping (bool): Whether to stop GPR training if loss begins to
            increase. Defaults to False.
        holdout (bool): Whether or not holdout evaluation is being performed.
            If True, do not calculate covariance matrix statistics. Defaults to
            False.

    Returns:
        model (BatchedGP/CompositeBatchedGP): A GPR model of BatchedGP type with
            which to generate synthetic predictions of rewards and next states.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): A likelihood
            object used for training and predicting samples with the BatchedGP
            model.
    """
    # Preprocess batch data
    B, N, XD = Zs.shape
    YD = Ys.shape[-1] if len(Ys.shape) > 2 else 1  # Note: Should always be > 2
    batch_shape = B * YD

    # Perform tiling/reshaping
    if YD > 1:
        train_x = Zs.repeat((YD, 1, 1))  # Tile dimensions of x
        train_y = torch.vstack([Ys[..., i] for i in range(YD)])

    # No need to perform tiling
    else:
        train_x = Zs
        train_y = torch.squeeze(Ys, -1)

    # Initialize likelihood for model
    likelihood = GaussianLikelihood(
        batch_shape=torch.Size([batch_shape]), noise_prior=None,
        noise_constraint=GreaterThan(parameters.MIN_INFERRED_NOISE_LEVEL))

    # Determine desired precision of training data (fp32 or fp64)
    if fp64:  # Double-precision
        train_x = train_x.double()
        train_y = train_y.double()
        likelihood = likelihood.double()
    else:
        train_x = train_x.float()
        train_y = train_y.float()
        likelihood = likelihood.float()

    # Select composite or concatenated type of kernel for model
    if composite_kernel:
        model = CompositeBatchedGP(
            train_x, train_y, likelihood, batch_shape, device,
            use_ard=use_ard, ds=ds, use_priors=use_priors, kernel=kernel,
            mean_type=mean_type, matern_nu=matern_nu,
            heuristic_lengthscales=cluster_heuristic_lengthscales)
    else:
        model = BatchedGP(
            train_x, train_y, likelihood, batch_shape, device,
            use_ard=use_ard, use_priors=use_priors, kernel=kernel,
            mean_type=mean_type, matern_nu=matern_nu,
            heuristic_lengthscales=cluster_heuristic_lengthscales,
            lengthscale_prior_std=lengthscale_prior_std)

    # Convert the model to the desired precision (fp32 or fp64)
    if fp64:  # Double-precision
        model = model.double()
    else:
        model = model.float()

    # Determine if model is initialized with hyperparameters
    if model_hyperparams is not None:

        # Stack the hyperparams
        stacked_model_hyperparams = {}
        for k, v in model_hyperparams.items():
            stacked_model_hyperparams[k] = torch.vstack([v[:, i, ...] for i in range(YD)])

        # Set model to eval mode
        model.initialize(**model_hyperparams)

    # Determine if we need to optimize hyperparameters
    if global_hyperparams:

        # Make sure model and likelihood are in "posterior" mode after training
        model.eval()
        likelihood.eval()

        if parameters.CUDA_AVAILABLE:  # Send everything to GPU for training

            # Empty the cache from GPU
            torch.cuda.empty_cache()
            gc.collect()  # NOTE: Critical to avoid GPU leak

            # Put model onto GPU
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Before returning, log metrics for the tensors
        if parameters.GLOBAL_STEP_COUNT % (parameters.REPLAY_RATIO * parameters.LOG_INTERVAL) == 0 \
                and not holdout:
            compute_covariance_metrics(model.covar_module, train_x,
                                       parameters.TB_WRITER)

        # Remove to prevent memory leaks
        del train_x, train_y, Zs, Ys

        # Return model and likelihood
        return model, likelihood

    # Determine which optimizer to use
    if use_lbfgs:
        opt = torch.optim.LBFGS
        lr = 1.0
        epochs = 2
    else:
        opt = torch.optim.Adam

    # Model in prior mode
    model.train()
    likelihood.train()
    optimizer = opt(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Send everything to GPU for training
    # NOTE: train_x and train_y should already be on GPU!
    if parameters.CUDA_AVAILABLE:
        model = model.cuda()
        likelihood = likelihood.cuda()
        mll = mll.cuda()

    def closure():
        """Helper function, specifically for L-BFGS."""
        optimizer.zero_grad()
        output = model(train_x)  # Forward pass through model
        loss = -mll(output, train_y).sum()  # Compute ind. losses + aggregate
        loss.backward()  # Backpropagate gradients
        gc.collect()  # NOTE: Critical to avoid GPU leak
        return loss

    def epoch_train():
        """Helper function for running training in the optimization loop. Note
        that the model and likelihood are updated outside of this function as
        well.
        """
        optimizer.zero_grad(set_to_none=None)  # Zero gradients
        output = model(train_x)  # Forward pass
        loss = -mll(output, train_y).sum()  # Compute ind. losses + aggregate
        loss.backward()  # Backpropagate gradients
        optimizer.step()  # Update weights
        gc.collect()  # NOTE: Critical to avoid GPU leak

    # Optimize with BoTorch
    if use_botorch:
        fit_gpytorch_model(mll)  # BoTorch optimization with L-BFGS
        gc.collect()  # Perform garbage collection operation to avoid mem leak

    # Optimize with ADAM or LFBGS, if using GPyTorch
    else:
        # Initialize variables to keep track of learning performance
        if early_stopping:
            patience_count = 0
            losses = [math.inf]
            prev_loss = math.inf

        # Run the optimization loop
        for i in range(epochs):

            # Optimize with L-BFGS
            if use_lbfgs:
                loss = optimizer.step(closure).item()

            # Optimize with ADAM
            else:
                # Train for an epoch
                epoch_train()

                # Terminate training error
                if early_stopping:
                    if loss_i > prev_loss:
                        patience_count += 1
                        print("EARLY STOPPING")
                        break
                    prev_loss = loss_i
                    losses.append(loss_i)

            # When finished, perform a garbage collection run
            gc.collect()

    # Make sure model and likelihood are in "posterior" mode after training
    model.eval()
    likelihood.eval()

    if parameters.CUDA_AVAILABLE:  # Send everything to GPU for training

        # Empty the cache from GPU
        torch.cuda.empty_cache()
        gc.collect()  # NOTE: Critical to avoid GPU leak

        # Put model onto GPU
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Before returning, log metrics for the tensors
    if parameters.GLOBAL_STEP_COUNT % (
            parameters.REPLAY_RATIO * parameters.LOG_INTERVAL) == 0 \
            and not holdout:
        compute_covariance_metrics(model.covar_module, train_x,
                                   parameters.TB_WRITER)
    return model, likelihood


def train_gpytorch_modellist(Zs, Ys, use_cuda=False, epochs=10, lr=0.1, thr=-1e5):
    """Computes a Gaussian Process object using GPyTorch. Rather than training
    in a batched fashion, this approach trains Gaussian Process Regression
    models iteratively in a list.

    NOTE: This approach does not leverage the same parallelism capabilities
        offered by the function above, which makes use of batching in both the
        batch and output dimensions.

    Parameters:
        Zs (np.array): Array of features of expanded shape (B, N, XD), where B
            is the size of the minibatch, N is the number of data points in each
            GP (the number of neighbors we consider in IER), and XD is the
            dimensionality of the state-action space of the environment.
        Ys (np.array): Array of predicted values of shape (B, N, YD), where B is
            the size of the minibatch and N is the number of data points in each
            GP (the number of neighbors we consider in IER), and YD is the
            dimensionality of the state-reward space of the environment.
        use_cuda (bool): Whether to use CUDA for GPU acceleration with PyTorch
            during the optimization step. Defaults to False.
        epochs (int):  The number of epochs to train the batched GPs over.
            Defaults to 10.
        lr (float):  The learning rate to use for the Adam optimizer to train
            the batched GPs.
        thr (float):  The mll threshold at which to stop training. Defaults to
            1e-5.

    Returns:
        model (BatchedGP): A GPR model of BatchedGP type with which to generate
            synthetic predictions of rewards and next states.
        likelihood (GaussianLikelihood): A likelihood object used for training
            and predicting samples with the BatchedGP model.
    """
    # Create GP Models using a Modellist object
    likelihoods = [MultitaskGaussianLikelihood(
        num_tasks=Ys.shape[-1]) for i in range(Ys.shape[0])]
    models = [
        MultitaskGPModel(Zs[i], Ys[i], likelihoods[i], num_tasks=Ys.shape[-1])
        for i in range(len(likelihoods))]
    likelihood = LikelihoodList(*[model.likelihood for model in models])

    # Create the aggregated model
    model = IndependentModelList(*models)

    # Create marginal log likelihood object
    mll = SumMarginalLogLikelihood(likelihood, model)

    # Ensure model and likelihood are trainable
    model.train()
    likelihood.train()

    # Send everything to GPU for training
    if use_cuda:
        model = model.cuda()
        likelihood.cuda()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)

    # Initialize loss to infinitely high value
    loss_val = math.inf
    i = 0  # Counter
    max_iters = epochs

    # Optimization loop
    while loss_val > thr and i < max_iters:
        optimizer.zero_grad(set_to_none=False)  # Zero gradients
        output = model(*model.train_inputs)  # Forward pass
        loss = -mll(output, model.train_targets)  # Compute objective
        loss.backward()  # Backpropagate gradients
        optimizer.step()  # Update parameters with gradients

        # Increment i and update loss tracker
        i += 1
        loss_val = loss.item()

    return model, model.likelihood


def preprocess_eval_inputs(Zs, d_y, device="cpu", fp64=True):
    """Helper function to preprocess inputs for use with training
    targets and evaluation.

    Parameters:
        Zs (np.array): Array of features of expanded shape (B, N, XD), where B is
            the size of the minibatch, N is the number of data points in each
            GP (the number of neighbors we consider in IER), and XD is the
            dimensionality of the state-action space of the environment.
        d_y (int):  The dimensionality of the targets of GPR.
        device (str):  The name of the device to assign tensors to. Defaults to
            "cpu".
        fp64 (bool):  Whether to perform GPR operations in double-precision, i.e.
            torch.float64. Defaults to True.

    Returns:
        eval_x (torch.tensor):  Torch tensor of shape (B * YD, N, XD). This
            tensor corresponding to a tiled set of inputs is used as input for
            the inference model in FP32 format.
    """
    # Preprocess batch data
    eval_x = torch.tensor(Zs, device=device)
    if fp64:
        eval_x = eval_x.double()
    else:
        eval_x = eval_x.float()
    eval_x = eval_x.repeat((d_y, 1, 1))
    return eval_x


def standardize(Y):
    """Standardizes to zero-mean gaussian. Preserves batch and output dimensions.

    Parameters:
        Y (torch.tensor): Tensor corresponding to the targets to be standardized.
            Expects shape (batch_shape, num_samples, dim_y) if y is
            multidimensional or (batch_shape, num_samples) if y is one dimension.

    Returns:
        Y_norm (torch.tensor):  Standard-normal standardized data. Shape is
            preserved.
        Y_std (torch.tensor): Standard deviation of the dataset across a given
            dimension. This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.
        Y_mean (torch.tensor):  Mean of the dataset across a given dimension.
            This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.
    """
    stddim = -1 if Y.dim() < 2 else -2
    Y_std = Y.std(dim=stddim, keepdim=True)
    Y_std = Y_std.where(Y_std >= 1e-9, torch.full_like(Y_std, 1.0))
    Y_mean = Y.mean(dim=stddim, keepdim=True)
    return (Y - Y_mean / Y_std), Y_std, Y_mean


def standardize_manual(Y, Y_std, Y_mean):
    """Performs manual standardization given a tensor, standard deviation, and
    mean. Useful if the variables being standardized are being standardized
    with different mean and standard deviation moments.

    Parameters:
        Y (torch.tensor): Tensor corresponding to the data to be standardized.
            Expects shape (batch_shape, num_samples, dim_y) if y is
            multidimensional or (batch_shape, num_samples) if y is one dimension.
        Y_std (torch.tensor): Standard deviation of the dataset across a given
            dimension. This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.
        Y_mean (torch.tensor): Mean of the dataset across a given dimension.
            This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.

    Returns:
        Y_norm (torch.tensor): Standard-normal standardized data. Shape is
            preserved.
    """
    return (Y - Y_mean) / Y_std


def unstandardize(Y, Y_std, Y_mean):
    """Unstandardizes targets. Relies on having pre-computed standard deivations
    and mean moments.

    Parameters:
        Y (torch.tensor): Tensor corresponding to the targets to be standardized.
            Expects shape (batch_shape, num_samples, dim_y) if y is
                multidimensional or (batch_shape, num_samples) if y is
                one dimensional.
        Y_std (torch.tensor): Standard deviation of the targets across a given
            dimension. This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.
        Y_mean (torch.tensor):  Mean of the targets across a given dimension.
            This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.

    Returns:
        Y (torch.tensor):  Unstandardized data, using the previously-computed
            moments. Shape is preserved.
    """
    tile_dim = Y.size()[-2]
    Y_std_tiled = Y_std.repeat((1, tile_dim, 1))
    Y_mean_tiled = Y_mean.repeat((1, tile_dim, 1))
    return torch.multiply(Y_std_tiled, Y) + Y_mean_tiled


def normalize_features_standardize_targets(
        Zs, Ys, keepdim_x=True, keepdim_y=True, verbose=False, device="cpu"):
    """Method to map the features and targets for data used in GPR to a
    standard domain.

    Parameters:
        Zs (torch.tensor): Input tensor of shape (B, N, D_x), where B is the
            batch size, N is the number of samples, and D_x is the dimension
            of the feature set. Min-max normalized in this function.
        Ys (torch.tensor): Input tensor of shape (B, N, D_y), where B is the
            batch size, N is the number of samples, and D_y is the dimension
            of the target set. Standardized to N(0, 1) in this function.
        keepdim_x (bool): Whether to preserve the inner dimension of the
            features. Defaults to True.
        keepdim_y (bool): Whether to preserve the inner dimension of the
            targets. Defaults to True.
        verbose (bool): Whether to print to console. Suggested for debugging,
            but not for production/debugged experiments. Defaults to False.
        device (str): The name of the device to assign tensors to. Defaults to
            "cpu".

    Returns:
        Z_normalized (torch.tensor): A min-max normalized representation of
            the input features. Shape is (B, N, D_x).
        Z_max (torch.tensor): The maximum batched values along batches and
            dimension. If keepdim_x = True, shape is (B, 1, D_x), else
            shape is (B, D_x).
        Z_min (torch.tensor): The minimum batched values along batches and
            dimension. If keepdim_x = True, shape is (B, 1, D_x), else
            shape is (B, D_x).
        Y_standardized (torch.tensor): A z-score standardized representation
            of the target features. Shape is (B, N, D_y).
        Y_mean (torch.tensor): The mean batched values along batches and
            dimension. If keepdim_y = True, shape is (B, 1, D_y), else
            shape is (B, D_x).
        Y_std (torch.tensor): The standard deviation of batched values along
            batches and dimension. If keepdim_y = True, shape is (B, 1, D_y),
            else shape is (B, D_x).
    """
    # Compute mean and std for Y, and ensure std is not zero
    Y_mean = Ys.mean(dim=1, keepdim=keepdim_y)  # Mean of each cluster
    Y_std = Ys.std(dim=1, keepdim=keepdim_y)  # Standard dev of each cluster
    idx_no_var = Y_std <= 0.0  # Find clusters with no standard dev
    Y_std[idx_no_var] = 1.0  # Setting does not affect variables

    # Perform standardization of targets
    Y_standardized = (Ys - Y_mean) / Y_std

    # Compute min and max
    Z_max = Zs.max(dim=1, keepdim=keepdim_x).values
    Z_min = Zs.min(dim=1, keepdim=keepdim_x).values

    # Compute any degenerate indices
    degenerate_idx = torch.eq(Z_max, Z_min)
    Z_max[degenerate_idx] = 1.0  # Setting does not affect variables
    Z_min[degenerate_idx] = 0.0  # Setting does not affect variables

    # Now perform min-max normalization
    Z_normalized = (Zs - Z_min) / (Z_max - Z_min)

    return Z_normalized, Z_max, Z_min, Y_standardized, Y_mean, Y_std


def standardize_features_standardize_targets(
        Zs, Ys, keepdim_x=True, keepdim_y=True, verbose=False,
        device="cpu", eps=1e-6):
    """Method to map the features and targets for data used in GPR to a
    standard domain.

    Parameters:
        Zs (torch.tensor): Input tensor of shape (B, N, D_x), where B is the
            batch size, N is the number of samples, and D_x is the dimension
            of the feature set. Min-max normalized in this function.
        Ys (torch.tensor): Input tensor of shape (B, N, D_y), where B is the
            batch size, N is the number of samples, and D_y is the dimension
            of the target set. Standardized to N(0, 1) in this function.
        keepdim_x (bool): Whether to preserve the inner dimension of the
            features. Defaults to True.
        keepdim_y (bool): Whether to preserve the inner dimension of Zs, the
            targets. Defaults to True.
        verbose (bool): Whether to print to console. Suggested for debugging,
            but not for production/debugged experiments. Defaults to False.
        device (str): The name of the device to assign tensors to. Defaults to
            "cpu".
        eps (1e-6): A small standard deviation value to add to each point.

    Returns:
        Z_standardized (torch.tensor): A min-max normalized representation of
            the input features. Shape is (B, N, D_x).
        Z_mean (torch.tensor): The mean batched values along batches and
            dimension. If keepdim_x = True, shape is (B, 1, D_x), else
            shape is (B, D_x).
        Z_std (torch.tensor): The standard dev batched values along batches and
            dimension. If keepdim_x = True, shape is (B, 1, D_x), else
            shape is (B, D_x).
        Y_standardized (torch.tensor): A z-score standardized representation
            of the target features. Shape is (B, N, D_y).
        Y_mean (torch.tensor): The mean batched values along batches and
            dimension. If keepdim_y = True, shape is (B, 1, D_y), else
            shape is (B, D_x).
        Y_std (torch.tensor): The standard deviation of batched values along
            batches and dimension. If keepdim_y = True, shape is (B, 1, D_y),
            else shape is (B, D_x).
    """
    # Compute mean and std for Y, and ensure std is not zero
    Y_mean = Ys.mean(dim=1, keepdim=keepdim_y)  # Mean of each cluster
    Y_std = Ys.std(dim=1, keepdim=keepdim_y)  # Add jitter

    # Set locations of zero std to have std 1
    Y_std[Y_std <= 0.0] = 1.0  # Setting does not affect variables

    # Perform standardization of targets
    Y_standardized = (Ys - Y_mean) / Y_std

    # Compute mean and std for Y, and ensure std is not zero
    Z_mean = Zs.mean(dim=1, keepdim=keepdim_x)  # Mean of each cluster
    Z_std = Zs.std(dim=1, keepdim=keepdim_x)  # Add jitter

    # Set locations of zero std to have std 1
    Z_std[Z_std <= 0.0] = 1.0  # Setting does not affect variables

    # Perform standardization of targets
    Z_standardized = (Zs - Z_mean) / Z_std

    return Z_standardized, Z_mean, Z_std, Y_standardized, Y_mean, Y_std


def compute_dist(X):
    """Utility function for computing the furthest distance from X.

    Parameters:
        X (np.array):  Array of shape (N, D), where N is the number of samples,
            and D is the dimensionality of the data. Note that X[0, :] should
            correspond to the center sample of the dataset.

        dist (np.array):  An array of distances corresponding to the maximum
            distance from each center to its furthest points. Each element of
            this array corresponds to a different distance value.
    """
    # Slice to extract sample point and neighborhood
    x = X[:, 0:1, :]  # Center point
    neighborhood = X[:, 1:, :]  # Neighborhood
    return torch.max(torch.abs(torch.subtract(x, neighborhood)), axis=1).values


def compute_avg_dist(X):
    """Utility function for computing the average distance from X.

    Parameters:
        X (np.array):  Array of shape (B, N, D), where N is the number of samples,
            and D is the dimensionality of the data. Note that X[0, :] should
            correspond to the center sample of the dataset.

        dist (np.array):  An array of distances corresponding to the maximum
            distance from each center to its furthest points. Each element of
            this array corresponds to a different distance value.
    """
    # Slice to extract sample point and neighborhood
    x = X[:, 0:1, :]  # Center point
    neighborhood = X[:, 1:, :]   # Neighborhood

    # Determine whether to process in torch or numpy
    if hasattr(X, "numpy"):  # Torch tensors
        return torch.mean(torch.abs(torch.subtract(x, neighborhood)), axis=1)
    elif hasattr(X, "shape"):  # NumPy arrays
        return np.mean(np.abs(np.subtract(x, neighborhood)), axis=1)


def heuristic_hyperparams(train_x, dy, lengthscale_constant=2.0,
                          mc_clustering=False):
    """Function to estimate hyperparameters for use as global hyperparameter
    priors. Specifically, approximates the lengthscales for setting a prior by
    computing the average mean distance over clusters in each dimension.

    Parameters:
        train_x (torch.tensor):  Tensor of inputs of shape (B, N, D), where B
            is the batch size, N is the number of points, and D is the dimension
            of the data.
        dy (int): The dimension of the outputs.
        lengthscale_constant (float): The value we multiply the mean estimates
            for the lengthscales.
        mc_clustering (bool): Whether or not the parameters presented are
                derived from a set of clusters.
    """
    # Compute mean distances
    x = train_x[:, 0:1, :]  # Center point
    K = train_x[:, 1:, :]  # Neighborhood
    mean_dist_over_clusters = np.mean(np.abs(np.subtract(x, K)), axis=1)

    # Now tile as needed
    if mc_clustering:  # Use clustered data
        scaled_lengthscales = lengthscale_constant * torch.tensor(mean_dist_over_clusters)
        cluster_heuristic_lengthscales = torch.unsqueeze(scaled_lengthscales, 1)
        cluster_heuristic_lengthscales = cluster_heuristic_lengthscales.repeat((dy, 1, 1)).double()
        return cluster_heuristic_lengthscales

    else:  # Use non-clustered data
        aggregate_mean_dist = np.mean(mean_dist_over_clusters, axis=0)
        tiled_aggregate_mean_dist = torch.tensor(np.tile(aggregate_mean_dist, (dy, 1)))
        return lengthscale_constant * tiled_aggregate_mean_dist


def format_preds(Y, B, single_model=False):
    """Function to format a tensor outputted from a GPR, either by tiling over
    intervals or transposition.

    Parameters:
        Y (torch.tensor):  Tensor corresponding to predicted outputs. If:

            1. single_model is False, expects a tensor of shape (B * Yd, 1),
            where B is the batch size, and Yd is the dimension of the output.
            Outputs of the ith element of the batch are given as i + B*j, where
            j is the dimension beginning at 0.

            2. single_model is True, expects a tensor of shape (Yd, B), where B
            is the batch size and Yd is the output. In this case, transposition
            is performed.

        B (int): The batch size (used for splitting).

        single_model (bool):  Whether a single model is used for prediction, or
            prediction is made in a batched format.

    Returns:
        Y_tiled (torch.tensor):  An output tensor of shape (B, Yd), where B is
        the batch size, and Yd is the dimension of the outputs. Each row
        corresponds to an output, and each column corresponds to a different
        predicted feature.
    """
    # Reformat the likelihoods to compute weights
    if single_model:
        return torch.transpose(Y, 0, 1)
    else:
        return torch.squeeze(torch.stack(torch.split(Y, B, dim=0), 1))


def format_preds_unsqueezed(Y, B, single_model=False):
    """Function to format a tensor outputted from a GPR, either by tiling over
    intervals or transposition.

    Compared to the function above, leaves the inner dimension unsqueezed.

    Parameters:
        Y (torch.tensor):  Tensor corresponding to predicted outputs. If:

            1. single_model is False, expects a tensor of shape (B * Yd, 1),
            where B is the batch size, and Yd is the dimension of the output.
            Outputs of the ith element of the batch are given as i + B*j, where
            j is the dimension beginning at 0.

            2. single_model is True, expects a tensor of shape (Yd, B), where B
            is the batch size and Yd is the output. In this case, transposition
            is performed.

        B (int): The batch size (used for splitting).

        single_model (bool):  Whether a single model is used for prediction, or
            prediction is made in a batched format.

    Returns:
        Y_tiled (torch.tensor):  An output tensor of shape (B, Yd), where B is
        the batch size, and Yd is the dimension of the outputs. Each row
        corresponds to an output, and each column corresponds to a different
        predicted feature.
    """
    # Reformat the likelihoods to compute weights
    if single_model:
        return torch.transpose(Y, 0, 1)
    else:
        return torch.stack(torch.split(Y, B, dim=0), 1)


def format_preds_precomputed(Y, B, single_model=False):
    """Function to format a tensor outputted from a GPR, either by tiling over
    intervals or transposition. Allows for repetition with test points.

    Parameters:
        Y (torch.tensor):  Tensor corresponding to predicted outputs. If:

            1. single_model is False, expects a tensor of shape (B * Yd, 1),
            where B is the batch size, and Yd is the dimension of the output.
            Outputs of the ith element of the batch are given as i + B*j, where
            j is the dimension beginning at 0.

            2. single_model is True, expects a tensor of shape (Yd, B), where B
            is the batch size and Yd is the output. In this case, transposition
            is performed.

        B (int): The batch size (used for splitting).

        single_model (bool):  Whether a single model is used for prediction, or
            prediction is made in a batched format.

    Returns:
        Y_tiled (torch.tensor):  An output tensor of shape (B, Yd), where B is
        the batch size, and Yd is the dimension of the outputs. Each row
        corresponds to an output, and each column corresponds to a different
        predicted feature.
    """
    # Reformat the likelihoods to compute weights
    if single_model:
        return torch.transpose(Y, 0, 1)
    else:
        return torch.squeeze(torch.stack(torch.split(Y, B, dim=0), -1))


def compute_covariance_metrics(kernel, x_train, writer):
    """Helper function to compute covariance metrics.

    Parameters:
        kernel (gpytorch.kernels.Kernel): A kernel object with a forward method
            for torch tensors.
        x_train (torch.Tensor):  A torch tensor of shape (B, N, D), where B
            is the batch size, N is the number of points, and D is the dimension
            of the data.
        writer (torch.utils.tensorboard.SummaryWriter): A summary writer object
            for logging metrics to tensorboard.
    """
    # Compute the covariance for the training data
    Kxx = kernel(x_train).evaluate()  # .evaluate() maps LazyTensor --> Tensor

    # Compute the condition number, p=2 norm gives \sigma_max / \sigma_min
    condition_number = torch.linalg.cond(Kxx, p=2)

    # Compute the log determinant as a stability metric
    log_det = torch.logdet(Kxx)

    # Compute the mean and variance of each metric, since models are batched
    vals = [condition_number, log_det]
    names = ["Condition Number", "Log Determinant"]
    for val, name in zip(vals, names):  # Loop jointly

        # Count number of NaNs
        num_nans = torch.sum(torch.isnan(val))

        # Get binary mask of indices where not NaN
        not_nan = ~(torch.isnan(val))

        # Get mean, maximum, and minimum
        mean_val = torch.mean(val[not_nan])
        max_val = torch.max(val[not_nan])
        min_val = torch.min(val[not_nan])
        types = [mean_val, max_val, min_val]
        name_types = ["Mean", "Max", "Min"]

        # Log the number of NaNs
        writer.add_scalar("Covariance/{} Number of NaNs".format(name),
                          num_nans, global_step=parameters.GLOBAL_STEP_COUNT)

        # Loop through the metrics to log
        for t, n in zip(types, name_types):
            writer.add_scalar("Covariance/{} {}".format(name, n), t,
                              global_step=parameters.GLOBAL_STEP_COUNT)
