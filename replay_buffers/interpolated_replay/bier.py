"""Module for Bayesian Interpolated Experience Replay (BIER) class without
pre-computed samples. This module completely defines the replay functions used
for BIER.
"""
# External Python packages
import gpytorch
import numpy as np
import torch
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Native Python packages
import time
import gc

# Ray/RLlib/CUDA
_ALL_POLICIES = "__all__"
USE_CUDA = torch.cuda.is_available()

# Custom Python packages/modules
from replay_buffers.interpolated_replay.ier_base import InterpolatedReplayBuffer
import utils.gpytorch.gpytorch_utils as gputils
from utils.visualization.visualization import plot_compare_linear_gp, pca_holdout
from parameters import GP_EPOCHS, GP_LR
import parameters


class BIER(InterpolatedReplayBuffer):
    """Performs Bayesian Interpolated Experience Replay (BIER) without
    pre-computing samples.

    Class for BIER without pre-computed samples. Inherits from
    InterpolatedReplayBuffer; parent class for PreCompBIER. Python class for
    Bayesian Interpolated Experience Replay with Gaussian Process
    Regression-based interpolation, implemented in the library
    GPyTorch. This class performs interpolation by fitting local environment
    models defined over the (s, a, r, s') transition space of the replay buffer.

    Some notes and recommendations on this model:

    1. Each of these local models is defined around a given sampled point,
        selected by Importance Sampling with Prioritized Experience Replay, and
        its K-nearest neighbors.
    2. For predicting interpolated_replay points, this paper applies Mixup sampling
        (convex linear combinations of points), and queries trained,
        locally-defined GPR models to predict the targets at the interpolated_replay
        point(s).
    3. It is strongly recommended to use standardization (N(0, 1)) transform)
        for targets, and min-max ([0, 1] transform) normalization for inputs.
        This is done in order to ensure that the GPyTorch default priors are
        consistent with the domain of inputs and targets.
    4. If using this for RL applications, it is recommended to predict
        (reward, \delta obs), where \delta obs = next obs - current obs,
        rather than (reward, next obs). Predicting delta states empirically
        results in better performance, and effectively solves the same problem.
    5. It is also recommended to use a zero mean function.
    6. For runtime considerations, it is recommended to use Monte Carlo-based
        hyperparameters for all models, as well as Automatic Relevance
        Determination(ARD). For ARD options, please see utils/gpytorch/models.py.
    7. This model is a subclass of InterpolatedReplayBuffer, and is a parent
        class of PreCompBIER.
    8. The parameters for all types of interpolated_replay experience replay are given
        in the InterpolatedReplayBuffer class.
    """
    def __init__(self, *args, **kwargs):

        # Call constructor of superclass (InterpolatedReplayBuffer)
        super().__init__(*args, **kwargs)

        # Select precision dtype for GPR
        if self.fp32:
            self.gpr_dtype = torch.float32
        else:
            self.gpr_dtype = torch.float64

        # Display feature preprocessing
        print("Normalize? {}".format(self.normalize))
        print("Standardizing Features? {}".format(self.standardize_gpr_x))

    def interpolate_samples(self):
        """Function for generating interpolated_replay samples using BIER.

        Method for interpolating samples using selected sample indices, query
        point selection using Mixup sampling, and interpolation of targets
        using localized Gaussian Process Regression models. Overview of method:

        1. Selected samples are generated using Importance Sampling and
            Prioritized Experience Replay.
        2. Neighbors are computed, and local Gaussian Process Regression models
            are fit around the neighborhoods.
        3. Query points are generated from the sampled points using Mixup
            sampling.
        4. Target predictions are made on the query points using trained local
            Gaussian Process Regression models.
        5. Interpolated trajectories are concatenated and passed to the
            InterpolatedReplayBuffer parent class.

        Returns:
            X_interp (np.array): A tuple composed of
                (obs, actions, rewards, next obs) of the training batch returned
                by the replay buffer.
            weights (None): Likelihood weights corresponding to the likelihood
                of the interpolated points using predicted variances from
                Gaussian Process Regression interpolation.
            interp_indices (list): Indices for the samples. If not needed,
                i.e. the interpolated priority update is not used, this is set
                to None.
            bs (np.array/list): An array of interpolation coefficients.
            neighbor_priorities (None): Indices for the neighbors. If not
                needed, i.e. the interpolated priority update is not used, this
                is set to None.
            sample_priorities (None): The priorities of the sample points. Only
                needed if the interpolated priority update is used, else can be
                set to None.
        """
        # Preprocess, sample points, and query neighbors
        self.sample_and_query()

        # Check if it's time to train one of our models
        if self.current_replay_count == 0:

            # USE A GLOBAL SET OF HYPERPARAMETERS
            if self.global_hyperparams:

                # Fit a global set of hyperparameters
                self.fit_global_hyperparameters()

            # EXECUTED IF WE HAVE A SINGLE GLOBAL MODEL
            if self.single_model:

                # Format training and analytic_evaluation data
                num_training_points = min(
                    self.total_sample_count, self.train_size)
                training_indices = np.random.choice(self.total_sample_count,
                    size=num_training_points, replace=False)

                # TODO(rms): If single model used, convert to torch
                # Create training data for GPR
                Z_train = Z[training_indices, :]
                Z_train.resize((1,) + Z_train.shape)
                Y_train = Y[training_indices, :]

                # Train model and likelihood
                self.model, self.likelihood = gputils.gpr_train_batch(
                    Z_train, Y_train, device=self.device,
                    epochs=GP_EPOCHS[self.env], lr=GP_LR[self.env],
                    use_ard=self.use_ard, composite_kernel=self.composite_kernel,
                    kernel=self.kernel, mean_type=self.mean_type,
                    matern_nu=self.matern_nu, ds=self.d_s,
                    use_priors=self.use_priors, use_lbfgs=self.use_lbfgs,
                    fp64=not self.fp32)

        # Time loop
        time_start_interpolation = time.time()

        # CREATE BATCHED GP MODELS
        if not self.single_model:

            # Track time to create tensors
            time_start_gpr_tensor_creation = time.time()

            # Create training data - including sample points
            Zs = torch.tensor([self.X[k] for k in self.nearest_neighbors_indices],
                              device=self.device)  # Training features
            Ys = torch.tensor([self.Y[k] for k in self.nearest_neighbors_indices],
                              device=self.device)  # Training labels

            # Cast to appropriate precision
            if self.fp32:  # Single-precision, i.e. fp32
                Zs = Zs.float()
                Ys = Ys.float()
            else:  # Double-precision, i.e. fp64
                Zs = Zs.double()
                Ys = Ys.double()

            time_end_gpr_tensor_creation = time.time()
            self.gpr_tensor_creation_time += time_end_gpr_tensor_creation - \
                                             time_start_gpr_tensor_creation

            # Normalize
            if self.normalize:

                # Normalization timing
                time_start_gpr_normalization = time.time()

                # Map to standardized features and targets
                if self.standardize_gpr_x:
                    Zs, Z_mean, Z_std, Ys, Y_mean, Y_std = \
                        gputils.standardize_x_standardize_y(
                            Zs, Ys, keepdim_x=True, keepdim_y=True,
                            verbose=self.verbose, device=self.device)

                # Map to min-max normalized features and standardized targets
                else:
                    Zs, Z_max, Z_min, Ys, Y_mean, Y_std = \
                        gputils.normalize_x_standardize_y(
                            Zs, Ys, keepdim_x=True, keepdim_y=True,
                            verbose=self.verbose, device=self.device)

                time_end_gpr_normalization = time.time()
                self.gpr_normalization_time += time_end_gpr_normalization - \
                                               time_start_gpr_normalization

            # If we have sampled new points, retrain the model.
            if (self.current_replay_count < self.retrain_interval * self.replay_ratio) or \
                    (not self.multi_sample):  # Retrain

                # Train model and likelihood
                if not hasattr(self, "model_hyperparams"):
                    self.model_hyperparams = None

                # Track time to create model
                time_start_gpr_model_creation = time.time()

                # Train model via GPR hyperparameter optimization
                self.model, self.likelihood = gputils.gpr_train_batch(
                    Zs, Ys, device=self.device, epochs=GP_EPOCHS[self.env],
                    lr=GP_LR[self.env], use_ard=self.use_ard,
                    composite_kernel=self.composite_kernel, kernel=self.kernel,
                    mean_type=self.mean_type, matern_nu=self.matern_nu,
                    global_hyperparams=self.global_hyperparams,
                    model_hyperparams=self.model_hyperparams,
                    ds=self.d_s, use_priors=self.use_priors,
                    use_lbfgs=self.use_lbfgs,
                    est_lengthscales=self.est_lengthscales,
                    lengthscale_constant=self.est_lengthscale_constant,
                    fp64=not self.fp32)

                time_end_gpr_model_creation = time.time()
                self.gpr_model_formation_time += time_end_gpr_model_creation - \
                                                 time_start_gpr_model_creation

        # If hyperparameters have been estimated, log them
        if self.step_count % (parameters.LOG_INTERVAL * self.replay_ratio)  == 0 and \
                hasattr(self, "model_hyperparams") and self.global_hyperparams:
            if self.composite_kernel:
                keys = ["state_base_kernel.lengthscale",
                        "state_base_kernel.raw_lengthscale",
                        "action_base_kernel.lengthscale",
                        "action_base_kernel.raw_lengthscale"]
            else:
                keys = ['covar_module.base_kernel.lengthscale',
                        'covar_module.base_kernel.raw_lengthscale']
            for key in keys:
                estimated_lengthscales = torch.flatten(self.model_hyperparams[key].cpu())
                self.writer.add_histogram(key, estimated_lengthscales, self.step_count)

        # Ensure model is no longer in training mode after training
        # Can adjust decomposition size in second context manager setting
        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            # Time inference preprocessing
            time_start_gpr_inference_preprocessing = time.time()

            # Place model and likelihood into evaluation mode
            self.model.eval()
            self.likelihood.eval()

            # Create query points for GPR
            X_s = torch.tensor(self.X[self.sample_indices], device=self.device)

            # Determine how neighbor(s) selected
            if self.multi_sample:  # Sample from all neighbors
                interp_indices = [k[1+self.current_replay_count] for k in self.nearest_neighbors_indices]
            else:  # Select neighbors at random
                interp_indices = [np.random.choice(k[1:self.furthest_neighbor])
                                  for k in self.nearest_neighbors_indices]

            # Get neighbor weights, if we perform an interpolated_replay update
            if self.interp_prio_update:
                neighbor_priorities = self.replay_buffers[self.policy_id].priorities[interp_indices]
                sample_priorities = self.replay_buffers[self.policy_id].priorities[self.sample_indices]

            # Slice dataset for interpolation points
            X_n = torch.tensor(self.X[interp_indices], device=self.device)

            # Determine how query points are interpolated_replay
            if self.use_smote or self.use_mixup:

                # Determine how to select coefficient
                if self.use_smote:
                    b = torch.tensor(
                        np.random.random(size=X_s.shape[0]), device=self.device)
                elif self.use_mixup:
                    b = torch.tensor(
                        np.random.beta(self.mixup_alpha, self.mixup_alpha,
                                       size=X_s.shape[0]), device=self.device)

                # Log coefficient to tensorboard
                if self.step_count % (parameters.LOG_INTERVAL * self.replay_ratio)  == 0:
                    self.writer.add_histogram("Interpolation Coefficient", b,
                                              self.step_count)

                # Tile coefficient along feature dimensions
                B = torch.transpose(b.repeat(self.d_s + self.d_a, 1), 0, 1)
                one_minus_B = torch.subtract(
                    torch.ones(B.shape, device=self.device), B)

                # Linearly interpolate to generate query point
                X_q = torch.add(torch.multiply(B, X_s),
                                     torch.multiply(one_minus_B, X_n))

            else:  # Interpolate with arithmetic mean
                X_q = torch.add(X_s, X_n) / 2
                b = torch.ones(X_s.shape[0]) * 0.5

            # Process features for inference
            if self.single_model:
                X_q_tensor = torch.unsqueeze(X_q, 0)  # Pred on single model
            else:
                X_q_tensor = torch.unsqueeze(X_q, 1)  # Pred on diff models

            # Normalize, if applicable
            if self.normalize:

                # Check whether to normalize or standardize features
                # Standardize features
                if self.standardize_gpr_x:
                    X_q_tensor = (X_q_tensor - Z_mean) / Z_std

                # Normalize features
                else:
                    X_q_tensor = (X_q_tensor - Z_min) / (Z_max - Z_min)

            # Cast to desired precision
            if self.fp32:
                X_q_tensor = X_q_tensor.float()
            else:
                X_q_tensor = X_q_tensor.double()

            # Now tile
            X_q_tensor = X_q_tensor.repeat((self.d_r + self.d_s, 1, 1))

            time_end_gpr_inference_preprocessing = time.time()
            self.gpr_inference_preprocessing_time += time_end_gpr_inference_preprocessing - \
                                                     time_start_gpr_inference_preprocessing

            # Perform inference
            time_start_gpr_inference = time.time()  # Start timing
            observed_pred = self.likelihood(self.model(X_q_tensor))  # Inference
            time_end_gpr_inference = time.time()  # End timing
            self.gpr_inference_time += time_end_gpr_inference - \
                                       time_start_gpr_inference

            # Take mean of each prediction for interpolated_replay batch
            samples = observed_pred.mean

            # Start gpr likelihood weights time
            time_start_gpr_likelihood_weights_time = time.time()

            if self.weighted_updates:  # Compute weights for updating losses

                # Take variance of each prediction
                variance = observed_pred.variance

                # Reformat the likelihoods to compute weights
                variance_block = gputils.format_preds(
                    variance, self.replay_batch_size,
                    single_model=self.single_model)

                if self.normalize:
                    variance_block = variance_block * torch.square(torch.squeeze(Y_std))

                # (i) Compute likelihood
                likelihoods = torch.reciprocal(torch.sqrt(2 * np.pi * variance_block))

                # (ii) Take squashed log probabilities for numerical stability
                log_likelihoods = torch.log(likelihoods)

                # (iii) Take sum along samples to take product in log domain
                log_weights_summed = torch.sum(log_likelihoods, dim=-1)

                # (iv) Exponentiate to compute joint squashed likelihood
                weights = torch.exp(log_weights_summed)

                # (v) Take geometric mean of likelihood
                geometric_mean_weights = torch.pow(weights, 1 / (self.d_r + self.d_s))

                # (vi) Squash weights between [1/2, 1] for stability
                squashed_weights = torch.sigmoid(geometric_mean_weights).cpu().numpy()

                # (vii) Normalize weights such that they have mean 1
                normalizing_factor = self.replay_batch_size / np.sum(squashed_weights)
                robust_weights = squashed_weights * normalizing_factor
                """
                # (v) Clip and normalize
                # Clip weights to self.wll_max multiplied by the median
                median = np.median(weights)
                clipped_weights = np.clip(weights, 0.0, self.wll_max * median)

                # Compute normalizing factor, scale, clip to [0, self.wll_max]
                normalizing_factor = \
                    self.replay_batch_size / np.sum(clipped_weights)

                # Now normalize the weights
                robust_weights = clipped_weights * normalizing_factor

                # Lastly, clip one more time to ensure we have no outliers
                robust_weights = np.clip(robust_weights, 0, self.wll_max)
                """

                # Log weights to tensorboard writer
                if self.step_count % (parameters.LOG_INTERVAL * self.replay_ratio)  == 0:
                    self.writer.add_histogram('likelihood_weights',
                                              robust_weights, self.step_count)

                    # Log the ratio of min: max
                    min_weight = np.min(robust_weights)
                    max_weight = np.max(robust_weights)
                    min_max_ratio = max_weight / min_weight
                    labels = ["Min Weight", "Max Weight", "Max:Min Ratio"]
                    values = [min_weight, max_weight, min_max_ratio]
                    for l, v in zip(labels, values):
                        self.writer.add_scalar(
                            "Likelihood Weights/{}".format(l),
                            v, global_step=self.step_count)


            time_end_gpr_likelihood_weights_time = time.time()
            self.gpr_likelihood_weights_time += time_end_gpr_likelihood_weights_time - \
                                                time_start_gpr_likelihood_weights_time

        # Reshape samples and normalize, if applicable
        out_y = gputils.format_preds(samples, self.replay_batch_size,
                                     single_model=self.single_model)

        # Unstandardize predicted targets
        if self.normalize:
            out_y = ((out_y * torch.squeeze(Y_std)) + torch.squeeze(Y_mean))

        # Move inputs and predictions from GPU -> CPU simultaneously to amortize
        out_x = X_q.detach().cpu().numpy()  # Query point
        out_y = out_y.detach().cpu().numpy()  # Predicted point
        b = b.cpu().numpy()  # Interpolation coefficients

        # Extract outputs from interpolated_replay x and predicted y
        Xn = out_x[:, :self.d_s]  # States
        An = out_x[:, self.d_s:]  # Actions
        Rn = np.squeeze(out_y[:, :self.d_r])  # Rewards

        # Choose which interpolation task
        if self.use_delta:
            Xn1 = Xn + out_y[:, self.d_r:]  # state + delta state
        else:
            Xn1 = out_y[:, self.d_r:]  # next state only

        # Check if prob_interpolation < 1
        if self.prob_interpolation < 1.0:  # TODO(rms): Integrate with interpolated priority update

            # Create mask to use vanilla replay over
            vanilla_mask = np.random.binomial(
                n=1, p=(1-self.prob_interpolation), size=Xn.shape[0])

            # Now set relevant interpolated_replay values to their vanilla values
            vanilla_idx = self.sample_indices[vanilla_mask]
            Xn[vanilla_mask, :] = self.X[vanilla_idx, :self.d_s]
            An[vanilla_mask, :] = self.X[vanilla_idx, self.d_s:]
            Rn[vanilla_mask, :] = np.squeeze(self.Y[vanilla_idx, :self.d_r])
            Xn1[vanilla_mask, :] = self.Y[vanilla_idx, self.d_r:]

        # Increment replay counter and reset if needed
        self.current_replay_count += 1
        if (self.current_replay_count >= self.retrain_interval * self.replay_ratio) \
                and (not self.single_model):
            self.current_replay_count = 0
            del self.model, self.likelihood  # Ensures these must be retrained

        # Delete model + likelihood to avoid GPU memory leak, and clear cache
        if self.use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Visualize
        if self.debug_plot:
            if self.step_count % parameters.LOG_INTERVAL == 0:
                plot_y = out_y

                # Tile b for when we perform PCA
                B = np.tile(b, (self.d_r + self.d_s, 1)).T
                plot_compare_linear_gp(self.Y, self.sample_indices,
                                       interp_indices, plot_y, idx=0,
                                       b=B, env=self.env, use_pca=True)

        # Perform holdout evaluation to evaluate interpolation
        if (self.perform_holdout_eval) and \
                (self.step_count % (self.holdout_freq * self.replay_ratio) == 0):
            self.perform_holdout_evaluation()

        # Compute GPR-specific metrics
        if self.step_count % (parameters.LOG_INTERVAL * self.replay_ratio) == 0:

            # Tile b for comparing linear to GPR
            B = np.tile(b, (self.d_r + self.d_s, 1)).T

            # Compare GPR to linear
            self.compute_delta_linear_gp(
                self.Y, self.sample_indices, interp_indices, out_y, b=B)

        # Delete tensors to avoid GPU memory leak
        del X_q, X_q_tensor, qout_y, out_x, observed_pred, samples

        # Group into a tuple for wrapping into a SampleBatch
        X_interpolated = (Xn, An, Rn, Xn1)

        # Check if priority sampling
        if not self.interp_prio_update:
            neighbor_priorities = None
            sample_priorities = None

        # Set variables to None if they don't exist
        if not (self.weighted_updates and self.gaussian_process):
            robust_weights = None

        # Increment counter
        if int(self.step_count % self.replay_ratio) == 0:
            self.transitions_sampled += 1

        # Stop timing interpolation
        time_end_interpolation = time.time()
        self.interpolation_time += time_end_interpolation - time_start_interpolation
        self.average_interpolation_time = self.interpolation_time / self.step_count

        return X_interpolated, robust_weights, interp_indices, b, \
               sample_priorities, neighbor_priorities

    def extract_and_construct_hyperparams(self, model, mc_clustering=False,
                                          batch_size=None):
        """Helper function to extract/construct hyperparameters for local models.

        This function can be used to extract hyperparameter and hyperparameter
        Monte Carlo estimates (across sample means of batched parameters), as
        well as construct torch.Tensor objects of appropriate shape and size
        for initializing batched models with a given set of hyperparameters.

        Parameters:
            model (gpytorch.models.ExactGP): A GPyTorch model from which we
                will extract hyperparameters.
            mc_clustering (bool): Whether or not the parameters presented are
                derived from a set of clusters.
            batch_size (int): The number of times to run repeated hyperparameter
                optimization. If None, defaults to replay batch size.
        """
        # Get the parameter tensors and string names
        params, names = self._extract_hyperparams(model, as_dict=False)

        # Outputs of parameter tiling
        self.model_hyperparams = {}
        self.model_untiled_hyperparams = {}
        self.avg_hyperparams = {}

        # Check batch size
        if batch_size is None:
            batch_size = self.replay_batch_size

        # If using Monte Carlo tiling, average parameters across the batch axis
        if mc_clustering:
            for i, (name, param) in enumerate(zip(names, params)):
                param_reshaped = gputils.format_preds(
                    param, self.replay_batch_size, single_model=self.single_model)
                self.model_untiled_hyperparams[name] = torch.squeeze(
                    param_reshaped.mean(dim=0))  # Take mean along batch dim

            # Now tile params - loop over hyperparameters
            for name, mean_param in self.model_untiled_hyperparams.items():

                # Tile parameters according to their dimension
                if mean_param.dim() >= 2:
                    self.model_hyperparams[name] = torch.unsqueeze(
                        mean_param.repeat((batch_size, 1)), 1)  # Tile
                else:
                    # Tile with mean of parameter
                    self.model_hyperparams[name] = mean_param.repeat(batch_size)

                    # Add additional dimension between batch_dim and target_dim
                    if name != "covar_module.outputscale":
                        self.model_hyperparams[name] = torch.unsqueeze(
                            self.model_hyperparams[name], -1)

                if self.warmstart_global:  # Retrain large model from scratch
                    self.avg_hyperparams[name] = mean_param  # For "warmstarts"

                # Log parameters to tensorboard
                plot_param = torch.flatten(param)
                self.writer.add_histogram(name, plot_param, self.step_count)

        # If not using Monte Carlo tiling, only need to perform tiling
        else:
            for name, param in zip(names, params):  # Loop over hyperparameters
                # Tile parameters according to their dimension
                if param.dim() > 2:
                    self.model_hyperparams[name] = param.repeat(
                        (batch_size, 1, 1))  # Tile
                else:
                    self.model_hyperparams[name] = param.repeat(
                        (batch_size, 1))  # Tile
                if self.warmstart_global:  # Retrain large model from scratch
                    self.avg_hyperparams[name] = param  # For warmstarts

                # Log parameters to tensorboard
                plot_param = torch.flatten(param)
                self.writer.add_histogram(name, plot_param, self.step_count)

    def _extract_hyperparams(self, model, as_dict=False):
        """Helper function to extract the hyperparameters of a model according
        to its kernel and mean structure.

        Parameters:
            model (gpytorch.model.ExactGP): An instance of the ExactGP class.
                Can be batched, but this is not a requirement.
            as_dict (bool): Whether to return the parameters and names together
                as key: value pairs in the dictionary (where keys are names,
                and values are params). Else, returns a list of params, and a
                list of names. Defaults to False.

        Returns:
            params (list): A list of torch.Tensor objects corresponding to the
                hyperparameters to be used for downstream tasks.
            names (list): A list of names corresponding to the hyperparameters
                used by the model.
        """
        # Extract hyperparameters
        if self.composite_kernel:  # Separate lengthscales for states/actions

            # Parameters used in accordance with model constraints
            lengthscale_state = model.state_base_kernel.lengthscale.cpu().detach()
            lengthscale_action = model.action_base_kernel.lengthscale.cpu().detach()

            # Raw parameters learned via -mll minimization
            raw_lengthscale_state = model.state_base_kernel.raw_lengthscale.cpu().detach()
            raw_lengthscale_action = model.action_base_kernel.raw_lengthscale.cpu().detach()

        else:  # Single lengthscales for states/actions (concatenated)
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

        # Determine what RQ parameters to use, if RQ kernel is used
        if self.kernel == "rq":
            if self.composite_kernel:
                names.extend(["covar_module.state_base_kernel.alpha",
                              "covar_module.action_base_kernel.alpha"])
                params.extend([model.covar_module.state_base_kernel.alpha,
                               model.covar_module.action_base_kernel.alpha])
            else:
                names.append("covar_module.base_kernel.alpha")
                params.append(model.covar_module.base_kernel.alpha)

        # Determine what mean parameters to use based off of mean type
        if self.mean_type == "linear":
            names.extend(['mean_module.bias', 'mean_module.weights'])
            params.extend([model.mean_module.bias, model.mean_module.weights])

        elif self.mean_type == "constant":
            names.append('mean_module.constant')
            params.append(model.mean_module.constant)

        # Set lengthscale according to factorization of kernel
        if self.composite_kernel:
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

        # Return as a co-referenced dictionary
        if as_dict:
            param_name_dict = {n: p for p, n in zip(params, names)}
            return param_name_dict

        # Return as two co-referenced lists
        else:
            return params, names

    def perform_holdout_evaluation(self, holdout_points=5):
        """Quantitative evaluation of interpolation quality.

        Method to perform holdout evaluation to assess the quality of
        interpolation. Though not an exact metric of interpolation quality for
        interpolated_replay samples, this method performs train/validation splits, and
        then assesses predictive performance on the holdout samples to determine
        interpolation accuracy of local GPR models in a batched setting.

        Parameters:
            holdout_points (int): The number of test points (in each cluster) to
                use in the holdout evaluation. Defaults to 5.
        """
        # Query neighbors using FAISS
        if self.use_faiss:
            neighbor_idx = self.faiss.query(
                self.X_norm[self.sample_indices, :])

        # Query neighbors using KD-Tree
        elif self.use_kd_tree:
            neighbor_idx = self.kd_tree.query(
                self.X_norm[self.sample_indices, :], k=self.kneighbors,
                return_distance=False, dualtree=True, sort_results=True)
        # Query neighbors using KNN
        else:
            neighbor_idx = self.knn.kneighbors(
                self.X_norm[self.sample_indices, :], return_distance=False)

        # Shuffle training indices
        # Only running interpolation on closest points in neighborhood
        if self.furthest_neighbor is not None:

            # Check if we have more holdout neighbors than sampling points
            if holdout_points > self.furthest_neighbor:
                holdout_points = self.furthest_neighbor
                print("Warning: Number of holdout points > Number of neighbors."
                      "Setting Number of holdout points equal to Number "
                      "of Neighbors.")

            # Get neighbor ranges and separate into "close" and "far" neighbors
            neighbor_range = np.arange(self.kneighbors)
            closest_neighbors = neighbor_range[:self.furthest_neighbor]
            furthest_neighbors = neighbor_range[self.furthest_neighbor:]

            # Shuffle indices
            idx_shuffled = np.array([np.concatenate(
                (shuffle(closest_neighbors), shuffle(furthest_neighbors))
            ) for _ in range(self.replay_batch_size)])

        # Running interpolation on all points in neighborhood
        else:
            idx_shuffled = np.array([shuffle(np.arange(self.kneighbors))
                for _ in range(self.replay_batch_size)])

        # Shuffle the indices for training and holdout
        idx_train = idx_shuffled[:, holdout_points:]
        idx_test = idx_shuffled[:, :holdout_points]

        # Split the neighbor indices according to training and holdout
        K_train = [k[train] for k, train in zip(neighbor_idx, idx_train)]
        K_test = [k[test] for k, test in zip(neighbor_idx, idx_test)]

        # Get training data using K_train neighbors
        Z_train = torch.tensor([(self.X[k_train]) for k_train in K_train],
                               device=self.device)  # Features
        Y_train = torch.tensor([(self.Y[k_train]) for k_train in K_train],
                               device=self.device)  # Targets

        # Get analytic test data using K_test neighbors
        Z_test = torch.tensor([self.X[k_test] for k_test in K_test],
                              device=self.device)
        Y_test = np.array([self.Y[k_test] for k_test in K_test])

        # Determine whether to perform normalization of the data
        if not self.normalize:
            if holdout_points == 1:  # Only add dimension if single test point
                Z_test = torch.unsqueeze(Z_test, 1)

        # Cast to desired precision (fp32 or fp64)
        if self.fp32:  # Single-precision (fp32)
            Z_train = Z_train.float()
            Y_train = Y_train.float()
            Z_test = Z_test.float()
        else:  # Double-precision (fp64)
            Z_train = Z_train.double()
            Y_train = Y_train.double()
            Z_test = Z_test.double()

        # Min-max normalize features (Z_train) and standardize (N(0,1)) targets
        if self.normalize:

            # Map to standardized features and targets
            if self.standardize_gpr_x:
                Z_train, Z_mean, Z_std, Y_train, Y_mean, Y_std = \
                    gputils.standardize_features_standardize_targets(
                        Z_train, Y_train, keepdim_x=True, keepdim_y=True,
                        verbose=self.verbose, device=self.device)

            # Map to min-max normalized features and standardized targets
            else:
                Z_train, Z_max, Z_min, Y_train, Y_mean, Y_std = \
                    gputils.normalize_features_standardize_targets(
                        Z_train, Y_train, keepdim_x=True, keepdim_y=True,
                        verbose=self.verbose, device=self.device)

        # Load model hyperparameters to be used for the model
        if hasattr(self, "store_hyperparameters"):
            model_hyperparams = self.load_hyperparams(self.sample_indices)
            global_hyperparams = False

        else:
            if not hasattr(self, "model_hyperparams"):
                self.model_hyperparams = None
            else:
                print("GLOBAL HYPERPARAMS FOUND FOR HOLDOUT...")
            model_hyperparams = self.model_hyperparams
            global_hyperparams = self.global_hyperparams

        # Train holdout model via GPR hyperparameter optimization
        holdout_model, holdout_likelihood = gputils.gpr_train_batch(
            Z_train, Y_train, device=self.device, epochs=GP_EPOCHS[self.env],
            lr=GP_LR[self.env], use_ard=self.use_ard,
            composite_kernel=self.composite_kernel,
            kernel=self.kernel, mean_type=self.mean_type,
            matern_nu=self.matern_nu,
            global_hyperparams=global_hyperparams,
            model_hyperparams=model_hyperparams,
            ds=self.d_s, use_priors=self.use_priors, use_lbfgs=self.use_lbfgs,
            est_lengthscales=self.est_lengthscales,
            lengthscale_constant=self.est_lengthscale_constant,
            fp64=not self.fp32, update_hyperparams=False, holdout=True)

        # Standardize test inputs for prediction
        if self.normalize:
            if holdout_points == 1:  # Only add dimension if single test point
                Z_test = torch.unsqueeze(Z_test, 1)

            # Check whether to normalize or standardize the features
            # Standardize features
            if self.standardize_gpr_x:
                Z_test = (Z_test - Z_mean) / Z_std

            # Normalize features
            else:
                Z_test = (Z_test - Z_min) / (Z_max - Z_min)

        # Place model in posterior mode with fast predictive variances (LOVE)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            # Place model and likelihood into evaluation mode
            holdout_model.eval()
            holdout_likelihood.eval()

            # Cast to desired precision (fp32 or fp64)
            if self.fp32:
                Z_test = Z_test.float()
            else:
                Z_test = Z_test.double()

            # Preprocess evaluation data via tiling by dim(rewards)+dim(new obs)
            Z_test = Z_test.repeat((self.d_r + self.d_s, 1, 1))

            # Compute predictions using holdout model
            observed_pred = holdout_likelihood(holdout_model(Z_test))
            samples = observed_pred.mean  # Take mean of pred as estimate

        # Format the output predictions according to dimension
        if holdout_points > 1:
            out_y = gputils.format_preds_precomputed(
                samples, self.replay_batch_size,
                single_model=self.single_model)
        else:
            out_y = gputils.format_preds(
                samples, self.replay_batch_size,
                single_model=self.single_model)

        # Unstandardize test targets for prediction
        if self.normalize:
            if holdout_points > 1:  # Don't need squeeze for test points > 1
                out_y = (out_y * Y_std) + Y_mean
            else:  # Need to squeeze dimensions if single test point
                out_y = (out_y * torch.squeeze(Y_std)) + torch.squeeze(Y_mean)

        # Convert to cpu + numpy
        out_y = out_y.cpu().numpy()

        # Format predictions into predicted new states and rewards
        predicted_states = out_y[:, self.d_r:]  # Could be \delta states
        predicted_rewards = out_y[:, :self.d_r]  # squeezing not applied here
        Y_test_hat = np.hstack((predicted_rewards, predicted_states))

        # Compute error between predictions and ground truth on holdout points
        E = np.subtract(Y_test_hat, Y_test)

        # Get axes to apply operations over
        if holdout_points > 1:
            axes = (0, 1)
        else:
            axes = 0

        # Now compute error between predicted and true for each dimension
        holdout_rmse_by_dimension = np.sqrt(np.mean(np.square(E), axis=axes))
        holdout_mae_by_dimension = np.mean(np.abs(E), axis=axes)

        # Log dimension-wise MSE
        for i, d_mse in enumerate(holdout_rmse_by_dimension):
            tag = "Holdout Evaluation-RMSE/Average RMSE, Dimension {}".format(i)
            self.writer.add_scalar(tag, d_mse, self.step_count)

        # Log dimension-wise MAE
        for i, d_mae in enumerate(holdout_mae_by_dimension):
            tag = "Holdout Evaluation-MAE/Average MAE, Dimension {}".format(i)
            self.writer.add_scalar(tag, d_mae, self.step_count)

        # Compute L1/L2 norms of errors across dimensions (same shape as batch)
        holdout_l2_error = np.linalg.norm(E, ord=2, axis=-1)
        holdout_l1_error = np.linalg.norm(E, ord=1, axis=-1)

        if self.debug_plot:  # Plot errors against batches

            # Compute holdout error by batch
            holdout_l2_error_by_batch = np.mean(holdout_l2_error, axis=1)

            # X-axes
            xs = np.arange(holdout_l2_error_by_batch.size)
            xs_mod_dy = xs % (self.d_s + self.d_r)

            # Compute correlation coefficients
            corrcoef_idx_l2_err = np.corrcoef(
                holdout_l2_error_by_batch, xs)[0, 1]
            corrcoef_modidx_l2_err = np.corrcoef(
                holdout_l2_error_by_batch, xs_mod_dy)[0, 1]

            # Plot errors against batch idx
            plt.scatter(xs, holdout_l2_error_by_batch)
            plt.xlabel("Batch idx")
            ply.ylabel("L2 Error")
            plt.title("Mean Batch Errors By Batch")
            plt.show()

        # Compute average L1 and L2 norms of errors across batch
        holdout_avg_l2_error = np.mean(holdout_l2_error, axis=axes)
        holdout_avg_l1_error = np.mean(holdout_l1_error, axis=axes)

        # Compute maximum L1 and L2 norms of errors across batch
        holdout_max_l2_error = np.max(holdout_l2_error, axis=axes)
        holdout_max_l1_error = np.max(holdout_l1_error, axis=axes)

        # Compute minimum L1 and L2 norms of errors across batch
        holdout_min_l2_error = np.min(holdout_l2_error, axis=axes)
        holdout_min_l1_error = np.min(holdout_l1_error, axis=axes)

        # Compute likelihood weights for predicted points
        if self.weighted_updates:  # Compute weights for updating losses

            # Take variance of each prediction
            variance = observed_pred.variance

            # Format variance into block
            if holdout_points > 1:  # Need to squeeze dimensions
                variance_block = gputils.format_preds_precomputed(
                    variance, self.replay_batch_size,
                    single_model=self.single_model)

            else:  # No need to squeeze dimensions
                variance_block = gputils.format_preds(
                    variance, self.replay_batch_size,
                    single_model=self.single_model)

            if self.normalize:
                if holdout_points > 1:  # Don't need squeeze for test points > 1
                    variance_block = variance_block * torch.square(Y_std)
                else:  # N
                    variance_block = variance_block * torch.square(torch.squeeze(Y_std))

            # (i) Compute likelihood
            likelihoods = torch.reciprocal(torch.sqrt(2 * np.pi * variance_block))

            # (ii) Take squashed log probabilities for numerical stability
            log_likelihoods = torch.log(likelihoods)

            # (iii) Take sum along samples to take product in log domain
            log_weights_summed = torch.sum(log_likelihoods, dim=-1)

            # (iv) Exponentiate to compute joint squashed likelihood
            weights = torch.exp(log_weights_summed)

            # (v) Take geometric mean of likelihood
            geometric_mean_weights = torch.pow(weights, 1 / (self.d_r + self.d_s))

            # (vi) Squash weights between [1/2, 1] for stability
            squashed_weights = torch.sigmoid(geometric_mean_weights).cpu().numpy()

            # (vii) Normalize weights such that they have mean 1
            normalizing_factor = (self.replay_batch_size * holdout_points) / np.sum(squashed_weights)
            robust_weights = squashed_weights * normalizing_factor

            """
            # (v) Clip and normalize
            # Clip weights to self.wll_max multiplied by the median
            median = np.median(weights)
            clipped_weights = np.clip(weights, 0.0, self.wll_max * median)

            # Compute normalizing factor, scale, clip to [0, self.wll_max]
            normalizing_factor = (self.replay_batch_size * holdout_points) / np.sum(clipped_weights)
            robust_weights = (clipped_weights * normalizing_factor)

            # Lastly, clip one more time to ensure we have no outliers
            robust_weights = np.clip(robust_weights, 0, self.wll_max)
            """
            # Flatten errors and weights to compute 1D correlation
            # (B, N) --> (B x N)
            holdout_l1_error_flattened = holdout_l1_error.reshape(-1)
            holdout_l2_error_flattened = holdout_l2_error.reshape(-1)
            robust_weights_flattened = robust_weights.reshape(-1)


            if self.verbose:
                ratio = np.max(robust_weights_flattened) / np.min(robust_weights_flattened)
                print("RATIO OF MIN: MAX IS: {}".format(ratio))
                print("MAX IS: {}".format(np.max(robust_weights_flattened)))

            # Compute correlation coefficient between weights and norm errors
            coercoef_weights_l1 = np.corrcoef(robust_weights_flattened,
                                              holdout_l1_error_flattened)[0, 1]
            coercoef_weights_l2 = np.corrcoef(robust_weights_flattened,
                                              holdout_l2_error_flattened)[0, 1]

            # Compute correlation between neighbor number and L1 and L2 error
            # (B, N) --> (B x N)
            idx_test_flattened = idx_test.reshape(-1)
            coercoef_idx_l1 = np.corrcoef(idx_test_flattened,
                                          holdout_l1_error_flattened)[0, 1]
            coercoef_idx_l2 = np.corrcoef(idx_test_flattened,
                                          holdout_l2_error_flattened)[0, 1]

            # Compute the average likelihood weights
            avg_likelihood_weights = np.mean(robust_weights_flattened)

        # Consolidate metrics for iteratively logging to tensorboard
        metrics = [holdout_avg_l1_error, holdout_avg_l2_error,
                   holdout_max_l1_error, holdout_max_l2_error,
                   holdout_min_l1_error, holdout_min_l2_error]
        names = ["Holdout Evaluation-Correlation-Norm/Average L1 Error",
                 "Holdout Evaluation-Correlation-Norm/Average L2 Error",
                 "Holdout Evaluation-Correlation-Norm/Maximum L1 Error",
                 "Holdout Evaluation-Correlation-Norm/Maximum L2 Error",
                 "Holdout Evaluation-Correlation-Norm/Minimum L1 Error",
                 "Holdout Evaluation-Correlation-Norm/Minimum L2 Error"]

        if self.weighted_updates:
            metrics.extend([coercoef_weights_l1, coercoef_weights_l2,
                            coercoef_idx_l1, coercoef_idx_l2,
                            avg_likelihood_weights])
            names.extend([
                "Holdout Evaluation-Correlation-Norm/Correlation "
                "Coefficient L1 vs. Weights",
                "Holdout Evaluation-Correlation-Norm/Correlation "
                "Coefficient L2 vs. Weights",
                "Holdout Evaluation-Correlation-Norm/Correlation "
                "Coefficient L1 vs. Neighbor Idx",
                "Holdout Evaluation-Correlation-Norm/Correlation "
                "Coefficient L2 vs. Neighbor Idx",
                "Holdout Evaluation-Correlation-Norm/Average "
                "Likelihood Weights"])

        # Loop over metrics and log to tb
        for name, metric in zip(names, metrics):

            # Log these quantities to tensorboard
            self.writer.add_scalar(name, metric, self.step_count)
            print("{}: {}".format(name, metric))

        # Finally, plot PCA of predicted point relative to its true value
        if self.debug_plot:

            # Transform 3d arrays into 2d arrays
            b, n_holdout, d_y = Y_test.shape

            # Perform PCA on holdout data with multiple holdout points
            if n_holdout > 1:  # Flatten batches + samples, but not dimensions
                Y_test_hat_2d = Y_test_hat.reshape((-1, d_y))  # Predictions
                Y_test_2d = Y_test.reshape((-1, d_y))  # Ground truth

                # Perform PCA on flattened data
                pca_holdout(Y_test_hat_2d, Y_test_2d, env=self.env)

            # Perform PCA on holdout data with only one holdout point
            else:
                pca_holdout(Y_test_hat, Y_test, env=self.env)

        # Delete tensors/arrays
        del Z_train, Y_train, Z_test, Y_test, weights, robust_weights, metrics

    def fit_global_hyperparameters(self, batch_size=None):
        """Fits hyperparameters using a subset of training data that can be
        applied to future models.

        Method to fit the global hyperparameters of the GPR model. This method
        can be applied to

        Parameters:
            batch_size (int): The number of times to run repeated hyperparameter
                optimization. If None, defaults to replay batch size.
        """
        # Log time
        start_time_training = time.time()

        # Format training and analytic_evaluation data
        if self.mc_hyper:
            num_training_points = min(self.total_sample_count,
                                      self.replay_batch_size)
        else:
            num_training_points = min(self.total_sample_count,
                                      self.train_size)

        # Sample points with PER or with uniform sampling
        training_indices = self.sample_global(L=self.total_sample_count,
                                              N=num_training_points,
                                              use_per=False)

        # Create training data for GPR if using a global model
        if not self.mc_hyper:
            Z_train = self.X[training_indices, :]  # Slice sample dimension
            Z_train.resize((1,) + Z_train.shape)  # Expand dims in-place
            Y_train = self.Y[training_indices, :]  # Slice sample dimension
            Y_train.resize((1,) + Y_train.shape)  # Expand dims in-place

            # Convert to torch tensor
            Z_train = torch.tensor(Z_train, device=self.device)  # Features
            Y_train = torch.tensor(Y_train, device=self.device)  # Targets

        # Compute global hyperparameters by averaging hyperparams from clusters
        elif self.mc_hyper:

            # Query neighbors using FAISS
            if self.use_faiss:
                global_neighbor_idx = self.faiss.query(
                    self.X_norm[training_indices, :])

            # Query neighbors using KD-Tree
            elif self.use_kd_tree:
                global_neighbor_idx = self.kd_tree.query(
                    self.X_norm[training_indices, :], k=self.kneighbors,
                    return_distance=False, dualtree=True, sort_results=True)

            # Query neighbors using KNN
            else:
                global_neighbor_idx = self.knn.kneighbors(
                    self.X_norm[training_indices, :], return_distance=False)

            # Create training/analytic_evaluation data
            Z_train = torch.tensor([self.X[k] for k in global_neighbor_idx],
                                   device=self.device)
            Y_train = torch.tensor([self.Y[k] for k in global_neighbor_idx],
                                   device=self.device)

        # Cast to desired precision (fp32 or fp64)
        if self.fp32:  # Single-precision, i.e. fp32
            Z_train = Z_train.float()
            Y_train = Y_train.float()
        else:  # Double-precision, i.e. fp64
            Z_train = Z_train.double()
            Y_train = Y_train.double()

        # Standardize y-dimension, if we perform standardization
        if self.normalize:

            # Map to standardized features and standardized targets
            if self.standardize_gpr_x:
                Z_train, _, _, Y_train, _, _ = \
                    gputils.standardize_features_standardize_targets(
                        Z_train, Y_train, keepdim_x=True, keepdim_y=True,
                        verbose=self.verbose, device=self.device)

            # Map to min-max normalized features and standardized targets
            else:
                Z_train, _, _, Y_train, _, _ = \
                    gputils.normalize_features_standardize_targets(
                        Z_train, Y_train, keepdim_x=True, keepdim_y=True,
                        verbose=self.verbose, device=self.device)

        # Train model via GPR hyperparameter optimization
        model, _ = gputils.gpr_train_batch(
            Z_train, Y_train, device=self.device,
            epochs=GP_EPOCHS[self.env],
            lr=GP_LR[self.env], use_ard=self.use_ard,
            composite_kernel=self.composite_kernel, kernel=self.kernel,
            mean_type=self.mean_type, matern_nu=self.matern_nu,
            ds=self.d_s, model_hyperparams=None,
            use_botorch=self.use_botorch, use_lbfgs=self.use_lbfgs,
            use_priors=self.use_priors,
            cluster_heuristic_lengthscales=None,
            lengthscale_constant=self.est_lengthscale_constant,
            lengthscale_prior_std=None,
            fp64=not self.fp32)

        # Extract and construct hyperparameters for neighborhood models
        self.extract_and_construct_hyperparams(
            model, mc_clustering=self.mc_hyper, batch_size=batch_size)

        # Log time for training models
        end_time_training = time.time()
        self.gpr_training_time += end_time_training - start_time_training

        # Run a garbage collection run to prevent memory leak
        gc.collect()

    def compute_delta_linear_gp(self, out_true, sample_indices,
                                interp_indices, gp_pred, b=None):
        """Method for comparing Gaussian Process Regression predictions at
        interpolated_replay points to counterfactual predictions produced by linear
        interpolation.

        Parameters:
            out_true (np.array): The next states/rewards from the replay buffer
                that are indexed into.
            sample_indices (np.array or list): The list or array of indices we
                index into - this is typically the list of samples produced by
                a uniform sampling or PER.
            interp_indices (np.array or list): The list or array of indices
                indexed into - this is typically the list of indices for points
                that are interpolated_replay with the sample points (the "neighbor"
                points).
            gp_pred (np.array): A 2D array of predicted values containing next
                states.
            idx (int): The dimension of next states we wish to index into.
                Note that the index is incremented for the GP predictions
                because the first dimension of these predictions is composed of
                the predicted reward.
            b (np.array): If using SMOTE or Mixup sampling, an array of values
                that define how far along the sample point to the neighboring
                point (on a scale of [0,1]). the interpolation is performed.
                Defaults to None, in which case the arithmetic mean of two
                points (b=0.5) is assumed.
        """
        # Extract data for plots using slicing and linear interpolation
        if b is None:  # Interpolate at arithmetic mean point
            linear_pred = np.add(out_true[sample_indices],
                            out_true[interp_indices]) / 2

        else:  # Interpolate with SMOTE or Mixup
            one_minus_b = np.subtract(np.ones(b.shape), b)
            linear_pred = np.add(np.multiply(b, out_true[sample_indices]),
                            np.multiply(one_minus_b, out_true[interp_indices]))

        # Now compare linear vs. GP
        delta_pred = linear_pred - gp_pred

        # Get average L2 and L1 norm difference
        mean_l1_delta = np.mean(np.linalg.norm(delta_pred, ord=1, axis=-1))
        mean_l2_delta = np.mean(np.linalg.norm(delta_pred, ord=2, axis=-1))

        # Get data and tags for tensorboard plotting
        scalars = [mean_l1_delta, mean_l2_delta]
        tags = ["Mean L1 Norm", "Mean L2 Norm"]

        # Loop jointly over scalars and tags, and log to Tensorboard
        for scalar, tag in zip(scalars, tags):
            if parameters.TB_WRITER is not None:  # Initialized as None
                parameters.TB_WRITER.add_scalar(
                    "Linear GP Compare/{}".format(tag), scalar,
                    global_step=parameters.GLOBAL_STEP_COUNT)
