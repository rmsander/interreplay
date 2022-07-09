"""Python class for Interpolated Experience Replay with vectorized Linear
Interpolation. This class serves as a baseline test for comparing to
Gaussian Process regression.
"""
# External Python packages
import numpy as np

# Native Python packages
import time

# Ray/RLlib/CUDA
_ALL_POLICIES = "__all__"

# Custom Python packages/modules
from replay_buffers.interpolated_replay.ier_base import InterpolatedReplayBuffer


class VectorizedLIER(InterpolatedReplayBuffer):
    """Implements Linear Interpolated Experience Replay (LIER), which samples
    interpolated_replay batches using linear interpolation of existing points via
    Mixup sampling. This Linear IER class is vectorized.

    Class for LIER with vectorization implemented in NumPy. Inherits from
    InterpolatedReplayBuffer. Python class for Linear Interpolated Experience
    Replay with local Mixup sampling-based interpolation, implemented in NumPy.

    Some notes and recommendations on this replay buffer model:

    1. For predicting interpolated_replay points, this paper applies Mixup sampling
        (convex linear combinations of points), and takes this
        linearly-interpolated_replay point as the interpolated_replay batch sample.
    2. The parameters for all types of interpolated_replay experience replay are given
        in the InterpolatedReplayBuffer class.

    The parameters for all types of interpolated_replay experience replay are given
    in ier_base.py.
    """
    def __init__(self, *args, **kwargs):

        # Call constructor of superclass (InterpolatedReplayBuffer)
        super().__init__(*args, **kwargs)

        # Disable use of queuing
        self.use_queue = False

    def interpolate_samples(self):
        """Method for interpolating samples using nearest neighbors in tandem
        with sampling new interpolated_replay batches using convex linear combinations
        of existing points, where the convex coefficient is sampled using Mixup
        sampling.

        Returns:
            X_interp (np.array): A tuple composed of
                (obs, actions, rewards, next obs) of the training batch returned
                by the replay buffer.
            weights (None): N/A for this replay buffer. Set to None.
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

        # Time interpolation duration and log to Tensorboard
        time_start_interpolation = time.time()

        # Get interpolation indices - select neighbor uniformly at random
        interp_indices = [np.random.choice(k[1:])
                          for k in self.nearest_neighbors_indices]

        # Extract prio weights, if interpolated priority updates are performed
        if self.interp_prio_update:
            neighbor_priorities = self.replay_buffers[self.policy_id].priorities[interp_indices]
            sample_priorities = self.replay_buffers[self.policy_id].priorities[self.sample_indices]

        # Now get sample and neighbor datasets
        X_sample, X_neighbor = self.X[self.sample_indices, :], \
                               self.X[interp_indices, :]
        Y_sample, Y_neighbor = self.Y[self.sample_indices, :], \
                               self.Y[interp_indices, :]

        # Take dataset to interpolate
        if self.use_smote or self.use_mixup:  # Use SMOTE/Mixup Sampling

            # Select coefficient to determine convex linear interpolation
            if self.use_smote:
                b = np.random.random(size=X_sample.shape[0])
            elif self.use_mixup:
                b = np.random.beta(self.mixup_alpha, self.mixup_alpha,
                                   size=X_sample.shape[0])

            # Perform tiling of query interpolation coefficients
            nx, dx = X_sample.shape  # Get feature dimensions
            ny, dy = Y_sample.shape  # Get target dimensions

            # Tile convex combination coefficients across dimensions
            B_x = np.repeat(b, dx).reshape((nx, -1))
            B_y = np.repeat(b, dy).reshape((ny, -1))

            # Take 1-B tiled coefficients over dimensions
            one_minus_B_x = np.subtract(np.ones(B_x.shape), B_x)
            one_minus_B_y = np.subtract(np.ones(B_y.shape), B_y)

            # Interpolate linearly using tiled coefficients
            dtype = X_sample.dtype
            X_interp = np.add(np.multiply(B_x, X_sample),
                              np.multiply(one_minus_B_x, X_neighbor)).astype(dtype)
            Y_interp = np.add(np.multiply(B_y, Y_sample),
                              np.multiply(one_minus_B_y, Y_neighbor)).astype(dtype)

        else:  # Interpolate with arithmetic mean between two points
            X_interp = np.add(X_sample, X_neighbor) / 2
            Y_interp = np.add(Y_sample, Y_neighbor) / 2
            b = np.ones(X_sample.shape[0]) * 0.5

        # Extract outputs from interpolated_replay x and predicted y
        Xn = X_interp[:, :self.d_s]  # Observations - Shape is (N, self.d_s)
        An = X_interp[:, self.d_s:]  # Actions - Shape is (N, self.d_a)
        Rn = np.squeeze(Y_interp[:, :self.d_r])  # Rewards - Shape is (N,)

        # Interpolate new state based off of state vs. \delta state
        if self.use_delta:
            Xn1 = Xn + Y_interp[:, self.d_r:]  # new_obs=obs + interp_delta_obs
        else:
            Xn1 = Y_interp[:, self.d_r:]  # new obs = interp_new_obs

        # Check if prob_interpolation < 1
        if self.prob_interpolation < 1.0:  # TODO(rms): Make sure all priority updates are unaffected

            # Create mask to use vanilla replay over
            vanilla_mask = np.random.binomial(
                n=1, p=(1 - self.prob_interpolation), size=Xn.shape[0])

            # Set relevant interpolated_replay values to "vanilla" values
            vanilla_idx = self.sample_indices[vanilla_mask]
            Xn[vanilla_mask, :] = self.X[vanilla_idx, :self.d_s]
            An[vanilla_mask, :] = self.X[vanilla_idx, self.d_s:]
            Rn[vanilla_mask] = np.squeeze(self.Y[vanilla_idx, :self.d_r])
            Xn1[vanilla_mask, :] = self.Y[vanilla_idx, self.d_r:]

        # Now transform array into SampleBatch with SampleBatchWrapper
        X_interpolated = (Xn, An, Rn, Xn1)

        # Set variables to None if they don't exist (placeholder)
        weights = None  # Likelihood weights, not IS weights
        if not self.interp_prio_update:  # No interpolated priority updates
            neighbor_priorities = None
            sample_priorities = None

        # Increment counter
        self.transitions_sampled += 1 / self.replay_ratio

        # Stop timing interpolation
        time_end_interpolation = time.time()
        self.interpolation_time += time_end_interpolation - \
                                   time_start_interpolation
        self.average_interpolation_time = self.interpolation_time / self.step_count

        return X_interpolated, weights, interp_indices, b, \
            neighbor_priorities, sample_priorities
