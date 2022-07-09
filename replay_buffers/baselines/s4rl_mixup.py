"""Python class for S4RL's Mixup implementation, to be used for comparing to
linear interpolated experience replay.

Paper: https://arxiv.org/pdf/2103.06326.pdf

Compared to Linear Interpolated Experience Replay (LIER), rather than
considering the closest points in transition space, S4RL State-Mixup
considers interpolating between a given transition's current state and its next
state.
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

class S4RLMixupBuffer(InterpolatedReplayBuffer):
    """Implements the S4RL Mixup Sampling baseline in which observations are
    replaced by a convex combination of the current observation and the next
    observation.

    This method is a subclass of the generalized InterpolatedReplayBuffer base
    class, but does not leverage any of the neighbor computation and
    standardization methods implemented in that base class.

    This class can be used in tandem with Prioritized Experience Replay (PER).
    Because the interpolated points apply to the same transition, no
    interpolated priority update is needed, and vanilla PER can be applied.
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

        # Sample transitions stochastically according to their priorities
        if self.prioritized_replay:
            self._sample_per()

        # Sample indices for transition batch uniformly from buffer
        else:
            self.sample_indices = np.random.choice(self.total_sample_count,
                size=self.replay_batch_size, replace=True)

        # Extract Observations at t and t+1
        S_t, S_tp1 = self.X[self.sample_indices, :self.d_s], \
                     self.Y[self.sample_indices, self.d_r:]

        # Extract Actions and rewards at time t
        A_t = self.X[self.sample_indices, self.d_s:]
        R_t = np.squeeze(self.Y[self.sample_indices, :self.d_r])

        # Sample interpolation coefficient
        b = np.random.beta(self.mixup_alpha, self.mixup_alpha,
                           size=self.replay_batch_size)

        # Perform tiling of query interpolation coefficients
        nx, dx = S_t.shape  # Get feature dimensions
        B = np.repeat(b, dx).reshape((nx, -1))  # Tile over obs dims
        one_minus_B = np.subtract(np.ones(B.shape), B)  # Tiled 1-b factor

        # Interpolate obs and next_obs linearly using tiled coefficients
        dtype = S_t.dtype
        S_interp = np.add(np.multiply(B, S_t),
                          np.multiply(one_minus_B, S_tp1)).astype(dtype)

        # Stop timing interpolation
        time_end_interpolation = time.time()
        self.interpolation_time += time_end_interpolation - \
                                   time_start_interpolation
        self.average_interpolation_time = self.interpolation_time / (self.step_count + 1)

        # Now transform array into SampleBatch with SampleBatchWrapper
        X_interpolated = (S_interp, A_t, R_t, S_tp1)

        # Set variables to None if they don't exist (placeholder)
        weights = None  # Likelihood weights, not IS weights
        neighbor_priorities = None
        sample_priorities = None
        interp_indices = None

        # Increment counter and step count (not incremented in base class)
        self.transitions_sampled += 1 / self.replay_ratio
        self.step_count += 1
        parameters.GLOBAL_STEP_COUNT = self.step_count  # Sync for logging

        return X_interpolated, weights, interp_indices, b, \
            neighbor_priorities, sample_priorities
