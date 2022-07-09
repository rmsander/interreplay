"""Script for running hyperparameter tuning using analytic motion and reward
functions from the OpenAI Gym Pendulum-v2 environment. Though these functions
do not interface with the Pendulum-v2 environment directory, the models used
are identical to those used in the environment.

This model allows for investing the fit performance of Gaussian Process
Regression models to simple environments.
"""
# External Python packages
import gpytorch

# Native Python packages
import copy

# Custom Python packages/modules
from utils.gpytorch.gpytorch_utils import gpr_train_batch, preprocess_eval_inputs
from utils.execution.execution_utils import set_seeds
from analytic_evaluation.hyperparameter_tuning.data_gen_utils import *
import parameters


def run_inference(model, likelihood, X_test, Y_test):
    """Function for running inference and forward pass.

    Parameters:
        model (BatchedGP/CompositeBatchedGP): A GPR model of BatchedGP type with
            which to generate synthetic predictions of rewards and next states.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): A likelihood
            object used for training and predicting samples with the BatchedGP
            model.
        X_test (torch.tensor): Tensor of shape (B, 1, Dx), where B is the batch
            size, and Dx is the dimension of the features. This is the tensor
            of test inputs for the model.
        Y_test (torch.tensor): Tensor of shape (B, 1, Dy), where B is the batch
            size, and Dy is the dimension of the targets. This is the tensor of
            analytic test targets for the model.

    Returns:
        means (np.array): Array of predicted mean moments at the test points.
            For the Pendulum-v2 environment, this corresponds to the predicted
            analytic reward at the points X_test.
    """
    # Perform inference without gradients and using LOVE
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        # Place model and likelihood into posterior mode
        model.eval()
        likelihood.eval()

        # Send to cuda, if available
        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Perform forward pass on trained model
        preds = likelihood(model(X_test))

        # Get mean moments from predictions
        means = preds.mean.cpu().numpy()

    # Flatten arrays for computing error metrics
    Y_test = np.squeeze(Y_test)
    means = np.squeeze(means)

    # Now compute error metrics (MAE, MSE, RMSE)
    MAE = np.mean(np.abs(means - Y_test))
    MSE = np.mean(np.square(means - Y_test))
    RMSE = np.sqrt(MSE)
    print("__________________")
    print("(Mean Absolute Error) MAE: {}".format(MAE))
    print("(Mean Squared Error) MSE: {}".format(MSE))
    print("(Root Mean Squared Error) RMSE: {}".format(RMSE))

    return means


def main():
    """Main script to run GPR function approximation in a supervised learning
    evaluation setting using the analytic observation and reward models of the
    OpenAI Gym Pendulum-v2 environment."""
    # Generate training and testing datasets
    DATASET_TYPES = ["uniform", "cluster"]
    SAMPLES_TRAIN = 10
    K = 200  # Number of neighbors
    STD = 1.0  # Standard deviation (for clustered data)
    BATCHED = True  # Use batching
    NUM_BATCHES = 256 if BATCHED else 1
    if DATASET_TYPES[1] == "cluster":
        SAMPLES_TEST = K * NUM_BATCHES if BATCHED else 500
    else:
        SAMPLES_TEST = 10
    BOUNDS = [(-np.pi, np.pi), (-8, 8), (-2, 2)]

    # Set seeds and repeats for reproducability and consistency
    SEED = 42  # Seed
    set_seeds(SEED)  # Set seed for RNGs
    REPEATS = 10  # Number of trials

    # GPR Kernel parameters
    USE_PRIORS = False  # Prior distributions over GPR parameters
    MEAN_TYPE = "zero"  # Mean type for GPR model
    KERNEL = "rbf"  # Kernel type for GPR model
    COMPOSITE_KERNEL = False  # Use composite kernel over states/actions

    # Optimization
    EPOCHS = 100  # Iterations of GPR training
    LR = 0.1  # Learning rate for GPR

    # Interpolation
    ALPHA = 0.75  # Coefficient for Mixup sampling

    # Generate training and testing datasets (torch.Tensor type)
    D_train, D_test = generate_datasets_pendulum(
        SAMPLES_TRAIN, SAMPLES_TEST, BOUNDS, dataset_types=DATASET_TYPES,
        num_batches=NUM_BATCHES, std=STD)

    # Parse training/analytic_evaluation data
    X_train, Y_train, X_obs_train = D_train
    X_test, Y_test, X_obs_test = D_test

    # Train GPR model on training dataset
    model, likelihood = gpr_train_batch(
        X_obs_train, Y_train, epochs=EPOCHS, lr=LR,
        use_cuda=parameters.CUDA_AVAILABLE, thr=-1e5, use_ard=True,
        use_priors=USE_PRIORS, use_lbfgs=False, mean_type=MEAN_TYPE,
        composite_kernel=COMPOSITE_KERNEL, kernel=KERNEL)

    # Extract hyperparameters and run fitting for test dataset/testing
    if BATCHED:
        model_hyperparameters = extract_and_construct_hyperparams(
            model, NUM_BATCHES, mean_type=MEAN_TYPE,
            composite_kernel=COMPOSITE_KERNEL)
        model_batched, likelihood_batched = gpr_train_batch(
            X_obs_test, Y_test, epochs=EPOCHS, lr=LR,
            use_cuda=parameters.CUDA_AVAILABLE, thr=-1e5, use_ard=True,
            use_priors=USE_PRIORS, use_lbfgs=False, mean_type=MEAN_TYPE,
            composite_kernel=COMPOSITE_KERNEL, kernel=KERNEL,
            model_hyperparams=model_hyperparameters, global_hyperparams=True)

    # Remove for memory
    del X_train, Y_train, X_obs_train

    # Run inference - either batched over clusters or unbatched
    if BATCHED:

        X_repeat = copy.deepcopy(X_test)
        X_plot = np.zeros((REPEATS * NUM_BATCHES, 3))
        Y_plot = np.zeros(REPEATS * NUM_BATCHES)
        M_plot = np.zeros(REPEATS * NUM_BATCHES)

        # Run several trials and append samples
        for i in range(REPEATS):
            # Take indices for interpolation query points
            sample_indices = np.random.randint(
                low=0, high=K, size=NUM_BATCHES)
            neighbor_indices = np.random.randint(
                low=0, high=K, size=NUM_BATCHES)
            is_equal = neighbor_indices == sample_indices
            neighbor_indices[is_equal] = neighbor_indices[is_equal] - 1

            # Take slices of dataset for sample and neighbor points
            X_obs_test = X_repeat.cpu().numpy()
            Xs_np = torch.tensor(
                [X[s, :] for X, s in zip(X_obs_test, sample_indices)])
            Xn = torch.tensor(
                [X[n, :] for X, n in zip(X_obs_test, neighbor_indices)])

            # Perform Mixup Sampling
            b = np.random.beta(ALPHA, ALPHA, size=NUM_BATCHES)
            B = torch.tensor(
                np.repeat(b, X_obs_test.shape[-1]).reshape((NUM_BATCHES, -1)))
            one_minus_B = torch.tensor(
                np.subtract(np.ones(B.shape), B))
            Xq = torch.add(torch.multiply(B, Xs),
                           torch.multiply(one_minus_B, Xn))
            Yq = compute_reward_pendulum_batched(Xq)

            # Get components of X
            Xq1 = Xq[..., 0]  # First dimension of test features
            Xq2 = Xq[..., 1]  # Second dimension of test features
            Aq = Xq[..., 2]  # Action dimension of test features

            # Construct interpolated_replay features
            X_cos = np.squeeze(np.cos(Xq1))  # Cosine
            X_sin = np.squeeze(np.sin(Xq1))
            X_thdot = np.squeeze(Xq2)
            X_a = np.squeeze(Aq)

            # Stack the interpolated_replay features
            X_q_obs = np.stack((X_cos, X_sin, X_thdot, X_a), axis=-1)

            # Add an additional placeholder dimension for prediction
            X_q_obs.resize((NUM_BATCHES, 1) + X_q_obs.shape[1:])

            # Perform tiling on interpolated_replay test points
            X_q_obs = preprocess_eval_inputs(X_q_obs, 1)

            # Place interpolated_replay features on CUDA, if available
            if torch.cuda.is_available():
                X_q_obs = X_q_obs.cuda()

            # Perform inference on the interpolated_replay test points
            means = run_inference(
                model_batched, likelihood_batched, X_q_obs, Yq).flatten()

            # Plot comparative predictions vs. true
            Xq = np.squeeze(Xq)
            idx_start = NUM_BATCHES * i
            idx_end = NUM_BATCHES * (i+1)
            X_plot[idx_start:idx_end, :] = Xq
            Y_plot[idx_start:idx_end] = Yq
            M_plot[idx_start:idx_end] = means

        # Now plot all predictions
        plot_predictions(X_plot, Y_plot, M_plot)

    # Run without batching
    else:
        # Perform inference on the interpolated_replay test points
        means = run_inference(model, likelihood, X_obs_test, Y_test).flatten()

        # Plot comparative predictions vs. true
        X_test = np.squeeze(X_test.cpu().numpy())
        plot_predictions(X_test, Y_test, means)


def extract_and_construct_hyperparams(model, batch_size, mean_type="constant",
                                      composite_kernel=False):
    """Function to extract and construct hyperparameters for local models.

    This function is used primarily for approximating the hyperparameters of a
    local model using a Monte Carlo average of other hyperparameters.

    Parameters:
        model (gpytorch.models.ExactGP):  A GPyTorch model from which we
            will extract hyperparameters.
        batch_size (int): The number of batches to tile the hyperparameters over.
        mean_type (str): Type of mean function to use for Gaussian Process.
            Defaults to constant mean ('constant'). Other options: linear
            ('linear'), and zero ('zero').")
        composite_kernel (bool):  Whether to use a composite kernel that computes
            the product between states and actions to compute the variance of y.
            Defaults to False.

    Returns:
        model_hyperparams (dict): A dictionary of model hyperparameters for
            initializing the batched model.
    """
    # Extract hyperparameters
    if composite_kernel:  # Use a factored kernel for GPR (affects lengthscale)
        # Hyperparams
        lengthscale_state = model.state_base_kernel.lengthscale.cpu().detach()
        lengthscale_action = model.action_base_kernel.lengthscale.cpu().detach()

        # Raw hyperparams
        raw_lengthscale_state = model.state_base_kernel.raw_lengthscale.cpu().detach()
        raw_lengthscale_action = model.action_base_kernel.raw_lengthscale.cpu().detach()

    else:  # Use a single kernel (affects lengthscale)
        lengthscale = model.covar_module.base_kernel.lengthscale.cpu().detach()
        raw_lengthscale = model.covar_module.base_kernel.raw_lengthscale.cpu().detach()

    # Extract kernel factorization-independent hyperparameters
    covar_noise = model.likelihood.noise_covar.noise.cpu().detach()
    outputscale = model.covar_module.outputscale.cpu().detach()
    raw_noise = model.likelihood.noise_covar.raw_noise.cpu().detach()

    # Set all but lengthscale
    names = ['likelihood.noise_covar.noise',
             'covar_module.outputscale',
             'likelihood.noise_covar.raw_noise']
    params = [covar_noise, outputscale, raw_noise]  # Values

    # Determine what mean parameters to use
    if mean_type == "linear":
        names.extend(['mean_module.bias', 'mean_module.weights'])
        params.extend([model.mean_module.bias, model.mean_module.weights])

    elif mean_type == "constant":
        names.append('mean_module.constant')
        params.append(model.mean_module.constant)

    # Set lengthscale according to factorization of kernel
    if composite_kernel:
        names.extend(["state_base_kernel.lengthscale",
                      "state_base_kernel.raw_lengthscale",
                      "action_base_kernel.lengthscale",
                      "action_base_kernel.raw_lengthscale"])
        params.extend([lengthscale_state, raw_lengthscale_state,
                       lengthscale_action, raw_lengthscale_action])
    else:
        names.extend(['covar_module.base_kernel.lengthscale',
                      'covar_module.base_kernel.raw_lengthscale'])
        params.extend([lengthscale, raw_lengthscale])

    # Outputs of hyperparameter tiling
    model_hyperparams = {}

    # Now tile params
    print("MODEL HYPERPARAMETER SUMMARY")
    print("_____________________________________________________________________")
    for name, param in zip(names, params):  # Loop over hyperparameters
        print("PARAMETER : {} \n{} \n_______________".format(name, param))
        # Tile parameters according to their dimension
        if param.dim() > 2:
            model_hyperparams[name] = param.repeat(
                (batch_size, 1, 1))  # Tile
        else:
            model_hyperparams[name] = param.repeat(
                (batch_size, 1))  # Tile
    print("_____________________________________________________________________")

    # Now return the tiled hyperparameters
    return model_hyperparams


if __name__ == '__main__':
    main()
