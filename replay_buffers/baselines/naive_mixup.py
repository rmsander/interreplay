"""Python class for naive Mixup replay in reinforcement learning.

Paper: https://arxiv.org/pdf/2103.06326.pdf

Performs linear interpolation between different transitions in the replay
buffer, but locality between the interpolated samples is not taken into account, 
i.e. interpolation can be performed by any pairs of transitions in the replay
buffer.
"""
# External Python packages
import numpy as np

# Native Python packages
import time

# Ray/RLlib/CUDA
_ALL_POLICIES = "__all__"

# Custom Python packages/modules
from replay_buffers.interpolated_replay.ier_base import InterpolatedReplayBuffer
import parameters

class NaiveMixupReplay(InterpolatedReplayBuffer):
    """Samples linearly-interpolated transitions composed of any pair of
     transitions in the replay buffer.

    This method is a subclass of the generalized InterpolatedReplayBuffer base
    class, but does not leverage any of the neighbor computation and
    standardization methods implemented in that base class.

    This class cannot be used in tandem with Prioritized Experience Replay (PER)
    """
    def __init__(self, *args, **kwargs):

        # Call constructor of superclass (InterpolatedReplayBuffer)
        super().__init__(*args, **kwargs)

        # Disable some features from other interpolated replay buffers
        # GPR interpolation
        self.gaussian_process = False  # Use GPR for interpolating new samples
        self.weighted_updates = False  # Weight samples by likelihood weights

        # Right version of sampling is selected
        self.use_smote = False
        self.use_mixup = True

        # Calculates delta states
        self.use_delta = False

        # Uses interpolated priority update and IS weight calculation
        self.interp_prio_update = False

        # Ensure pre-processing is not needed for the base replay buffer
        self.preprocessing_required = False

    def interpolate_samples(self):
        """Method for interpolating samples using convex linear
        combinations of observation/next observation pairs as the current
        transition, where the convex coefficient is sampled using
        Mixup sampling.

        Returns:
            X_interp (np.array): A tuple composed of
                (obs, actions, rewards, next obs) of the training batch returned
                by the replay buffer.
            weights (None): N/A for this replay buffer.
            interp_indices (None): N/A for this replay buffer.
            bs (np.array/list): An array of interpolation coefficients.
            neighbor_priorities (None): N/A for this replay buffer.
            sample_priorities (None): N/A for this replay buffer.
            sample_priorities (None): N/A for this replay buffer.
        """
        # Time interpolation duration and log to Tensorboard
        time_start_interpolation = time.time()

        # Sample batch of transitions and their consecutive next transitions
        self.sample_training_batch()

        # Now get sample and neighbor datasets
        X_1, X_2 = self.X[self.sample_indices, :], \
                     self.X[self.neighbor_indices, :]
        Y_1, Y_2 = self.Y[self.sample_indices, :], \
                     self.Y[self.neighbor_indices, :]

        # Sample interpolation coefficient
        b = np.random.beta(self.mixup_alpha, self.mixup_alpha,
                           size=self.replay_batch_size)

        # Perform tiling of query interpolation coefficients
        nx, dx = X_1.shape  # Get feature dimensions
        ny, dy = Y_1.shape  # Get target dimensions
        B_x = np.repeat(b, dx).reshape((nx, -1))
        B_y = np.repeat(b, dy).reshape((ny, -1))
        one_minus_B_x = np.subtract(np.ones(B_x.shape), B_x)
        one_minus_B_y = np.subtract(np.ones(B_y.shape), B_y)

        # Interpolate linearly w/ tiled coefficients and consecutive transitions
        dtype = X_1.dtype
        X_interp = np.add(np.multiply(B_x, X_1),
                          np.multiply(one_minus_B_x, X_2)).astype(dtype)
        Y_interp = np.add(np.multiply(B_y, Y_1),
                          np.multiply(one_minus_B_y, Y_2)).astype(dtype)

        # Extract outputs from interpolated_replay x and predicted y
        Xn = X_interp[:, :self.d_s]  # Observations
        An = X_interp[:, self.d_s:]  # Actions
        Rn = np.squeeze(Y_interp[:, :self.d_r])  # Rewards
        Xn1 = Y_interp[:, self.d_r:]  # new obs = interp_new_obs

        # Stop timing interpolation
        time_end_interpolation = time.time()
        self.interpolation_time += time_end_interpolation - \
                                   time_start_interpolation
        self.average_interpolation_time = self.interpolation_time / (
                    self.step_count + 1)

        # Now transform array into SampleBatch with SampleBatchWrapper
        X_interpolated = (Xn, An, Rn, Xn1)

        # Set variables to None if they don't exist (placeholder)
        weights = None  # Likelihood weights, not IS weights

        # Increment counter and step count (not incremented in base class)
        self.transitions_sampled += 1 / self.replay_ratio
        self.step_count += 1
        parameters.GLOBAL_STEP_COUNT = self.step_count  # Sync for logging

        # Use interpolated priority update
        if self.interp_prio_update:
            neighbor_priorities = \
                self.replay_buffers[self.policy_id].priorities[self.neighbor_indices]
            sample_priorities = \
                self.replay_buffers[self.policy_id].priorities[self.sample_indices]
        else:
            neighbor_priorities = None
            sample_priorities = None

        return X_interpolated, weights, self.neighbor_indices, b, \
               neighbor_priorities, sample_priorities

    def sample_training_batch(self):
        """Method to sample a minibatch of transitions and their consecutive
        next transitions, i.e. the next transition along a trajectory.

        Sampling can be performed using either uniform sampling or Prioritized
        Experience Replay (PER).

        Modifies:
            self.sample_indices (list): Updates with the indices of the current
                sampled minibatch.
            self.neighbor_indices (list): Updates with the indices of the
                transitions (batched) following the transitions of those sampled
                in self.sample_indices.
        """
        # Uniformly sample training indices from replay buffer
        self.sample_indices = np.random.choice(
            self.total_sample_count, size=self.replay_batch_size, replace=True)
        self.neighbor_indices = np.random.choice(
            self.total_sample_count, size=self.replay_batch_size, replace=True)
