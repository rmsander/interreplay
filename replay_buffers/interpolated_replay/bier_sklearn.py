"""Python class for Interpolated Experience Replay with Gaussian Process
Regression-based interpolation, implemented in the library Scikit-Learn
(sklearn).
"""

# External Python packages
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import torch

# Native Python packages
import time

# Ray/RLlib/CUDA
_ALL_POLICIES = "__all__"
USE_CUDA = torch.cuda.is_available()

# Custom Python packages/modules
from replay_buffers.interpolated_replay.ier_base import InterpolatedReplayBuffer


class BIERSk(InterpolatedReplayBuffer):
    """Performs Bayesian Interpolated Experience Replay (BIER) without
    pre-computing samples.

    Class for BIER without pre-computed samples. Inherits from
    InterpolatedReplayBuffer. Python class for
    Bayesian Interpolated Experience Replay with Gaussian Process
    Regression-based interpolation, implemented in the Sklearn.
    This class performs interpolation by fitting local environment
    models defined over the (s, a, r, s') transition space of the replay buffer.

    A few notes on this replay buffer class:

    1. The parameters for all types of interpolated_replay experience replay are given
        in ier_base.py.

    2. The BIER class (in bier.py) is recommended for use in
        high-dimensional environments, as it sees better performance and
        achieves substantially faster runtime through batched GPR training
        and inference backed by CUDA and GPyTorch.
    """
    def __init__(self, *args, **kwargs):

        # Call constructor of superclass (InterpolatedReplayBuffer)
        super().__init__(*args, **kwargs)

    def interpolate_samples(self):
        """Method for interpolating samples using selected sample indices, query
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

        # Time loop
        time_start_interpolation = time.time()

        # Placeholder for SMOTE values
        bs = []
        interp_indices = []

        # Create placeholder arrays to iteratively set - ensure data types match
        N = self.replay_batch_size
        dt = self.X.dtype
        Xn = np.zeros((N, self.d_s)).astype(dt)
        An = np.zeros((N, self.d_a)).astype(dt)
        Rn = np.squeeze(np.zeros((N, self.d_r))).astype(dt)  # (N,), not (N, 1)
        Xn1 = np.zeros((N, self.d_s)).astype(dt)

        # Iterate through N sampled points and interpolate N new points
        for i, (t, k) in enumerate(
                zip(self.sample_indices, self.nearest_neighbors_indices)):

            # Get components of query point
            (xt, at, rt, x_next_t) = self.X[t, :self.d_s], self.X[t, self.d_s:], \
                                     self.Y[t, :self.d_r], self.Y[t, self.d_r:]

            # Randomly determine if real sample appended, or synthetic sample
            if bool(np.random.binomial(1, self.prob_interpolation)):

                # Get batch of Z and Y
                Z_batch = self.X[k]
                Y_batch = self.Y[k]

                # Normalize Y to N(0, 1)
                norm = StandardScaler()
                Y_batch_norm = norm.fit_transform(Y_batch)

                # Determine if we use PCA with GP
                if self.gp_pca:  # Use PCA

                    # Transform with PCA
                    Z_pca, Y_pca = self.gp_pca(Z_batch, Y_batch)

                    # Fit a Gaussian Process
                    gpr = GaussianProcessRegressor(
                        kernel=None, random_state=self.seed).fit(
                        Z_pca, Y_pca)

                else:  # Do not use PCA
                    # Fit a Gaussian Process
                    gpr = GaussianProcessRegressor(
                        kernel=None, random_state=self.seed).fit(
                        Z_batch, Y_batch_norm)

                # Now sample a y value using the query point
                p_query = np.hstack((xt, at))
                idx_interp = np.random.choice(k[1:self.furthest_neighbor])
                p_interp = self.X[idx_interp]

                # Add to indices
                interp_indices.append(idx_interp)

                # Average points to form new query point
                if self.use_smote or self.use_mixup:  # Use SMOTE or Mixup
                    if self.use_smote:
                        b = np.random.random()  # Sample randomly between [0, 1]
                    elif self.use_mixup:
                        b = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    bs.append(b)
                    p_query_prime = np.add(b * p_query, (1-b) * p_interp)
                    p_query_prime = p_query_prime.reshape((1, -1))

                else:
                    bs.append(0.5)
                    p_query_prime = np.add(p_query, p_interp) / 2
                    p_query_prime = p_query_prime.reshape((1, -1))

                if self.gp_pca:  # Project queryed point to lower dimension
                    p_query_prime = self.pca_x.transform(p_query_prime)

                # Determine if we sample with mean and cov, or just mean
                if self.sample_gp_cov:  # Sample with mean and cov

                    # Predict mean and covariance from GP
                    y_mean, y_cov = gpr.predict(p_query_prime, return_cov=True)
                    sample = norm.inverse_transform(mean).flatten()

                    # Compute inverse determinant as likelihood weight
                    # TODO(rms): If we use sklearn, need to recompute likelihood
                    gp_weights[i] = 1/np.linalg.det(y_cov)

                else:  # Sample only mean (not likelihood-weighted)
                    mean = gpr.predict(
                        p_query_prime, return_cov=False).reshape((1, -1))
                    sample = norm.inverse_transform(mean).flatten()

                # Create interpolated_replay sample
                flattened_query = p_query_prime.flatten()
                xn = flattened_query[:self.d_s]
                an = flattened_query[self.d_s:]
                rn = np.squeeze(sample[:self.d_r])
                x_next_n = sample[self.d_r:]

                # Add synthetic sample to output arrays
                Xn[i] = xn
                An[i] = an
                Rn[i] = rn
                Xn1[i] = x_next_n

            else:  # Simply sample a true sample from the replay buffer
                # Add real sample to output arrays
                Xn[i] = xt
                An[i] = at
                Rn[i] = rt
                Xn1[i] = x_next_t

        # Get neighbor weights, if we perform an interpolated_replay update
        if self.interp_prio_update:
            neighbor_priorities = self.replay_buffers[self.policy_id].priorities[interp_indices]
            sample_priorities = self.replay_buffers[self.policy_id].priorities[self.sample_indices]

        # Now transform array into SampleBatch with SampleBatchWrapper
        X_interpolated = (Xn, An, Rn, Xn1)

        # Set variables to None if they don't exist
        if not (self.weighted_updates and self.gaussian_process):
            weights = None  # Likelihood weights, not IS weights

        if not self.interp_prio_update:
            self.neighbor_priorities = None
            self.sample_priorities = None

        # Stop timing interpolation
        time_end_interpolation = time.time()
        self.interpolation_time += time_end_interpolation - \
                                   time_start_interpolation
        self.average_interpolation_time = self.interpolation_time / self.step_count

        return X_interpolated, weights, interp_indices, bs, \
            sample_priorities, neighbor_priorities
