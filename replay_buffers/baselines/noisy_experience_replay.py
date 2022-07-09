"""Noisy experience replay buffer. Selects transitions uniformly at random, and
perturbs the transitions with zero-mean noise level specified in the
construction of the class.
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


class NoisyReplay(InterpolatedReplayBuffer):
    """Implements a noisy uniform replay buffer. Samples are drawn uniformly
    from the replay buffer, and are replaced in a FIFO fashion. Transitions are
    perturbed with i.i.d. noise in each dimension.

    Parameters:
        sigma (float): Value for the level of noise in the perturbed vector.

    """
    def __init__(self, sigma=0.1, *args, **kwargs):

        # Call constructor of superclass (InterpolatedReplayBuffer)
        super().__init__(*args, **kwargs)

        # Define noise level for perturbing samples
        self.sigma = sigma

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
        self.interp_prio_update = False  # Since using vanulla replay
        self.prioritized_replay = False  # Since using vanilla replay

        # Ensure pre-processing is not needed for the base replay buffer
        self.preprocessing_required = False

        print("USING NOISY ER")

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
        # Sample indices for transition batch uniformly from buffer
        self.sample_indices = np.random.choice(
            self.total_sample_count, size=self.replay_batch_size, replace=True)

        # Extract transition from stored datasets and perturb with noise
        # States/observations
        S_t = self.X[self.sample_indices, :self.d_s] * \
              (1 + np.random.normal(loc=0, scale=self.sigma, size=self.d_s))

        # Actions
        A_t = self.X[self.sample_indices, self.d_s:] * \
              (1 + np.random.normal(loc=0, scale=self.sigma, size=self.d_a))

        # Rewards
        R_t = np.squeeze(self.Y[self.sample_indices, :self.d_r]) * \
              (1 + np.random.normal(loc=0, scale=self.sigma, size=self.d_r))

        # Next states/observations
        S_tp1 = self.Y[self.sample_indices, self.d_r:] * \
              (1 + np.random.normal(loc=0, scale=self.sigma, size=self.d_s))

        # Sample interpolation coefficient
        b = np.ones(self.replay_batch_size)

        # Now transform array into SampleBatch with SampleBatchWrapper
        X_interpolated = (S_t, A_t, R_t, S_tp1)

        # Set variables to None if they don't exist (placeholder)
        weights = None  # Likelihood weights, not IS weights
        neighbor_priorities = None
        sample_priorities = None
        interp_indices = None

        # Increment counter and step count (not incremented in base class)
        self.transitions_sampled += 1 / self.replay_ratio
        self.step_count += 1

        # Sync/set global variables
        parameters.GLOBAL_STEP_COUNT = self.step_count  # Sync for logging
        parameters.REPLAY_RATIO = self.replay_ratio  # Set global parameter
        parameters.RL_ENV = self.env  # Set global parameters
        #parameters.SAC_L2_REG[self.env] = None

        return X_interpolated, weights, interp_indices, b, \
            neighbor_priorities, sample_priorities
