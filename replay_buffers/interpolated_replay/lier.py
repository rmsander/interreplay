"""Python class for Interpolated Experience Replay with linear interpolation
between sampled transitions and their nearest neighbors.
"""
# External Python packages
import numpy as np

# Native Python packages
import time

_ALL_POLICIES = "__all__"

# Custom Python packages/modules
from replay_buffers.interpolated_replay.ier_base import InterpolatedReplayBuffer


class LIER(InterpolatedReplayBuffer):
    """Implements Linear Interpolated Experience Replay (LIER), which samples
    interpolated_replay batches using linear interpolation of existing points via
    Mixup sampling.

    Class for LIER. Inherits from InterpolatedReplayBuffer.
    Python class for Linear Interpolated Experience Replay with local
    Mixup sampling-based interpolation, implemented in with NumPy.

    Some notes and recommendations on this model:

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

        # Time loop
        time_start_interpolation = time.time()

        # Placeholder for SMOTE values
        bs = []
        interp_indices = []

        # Create placeholder arrays to iteratively set - ensure data types match
        N = self.replay_batch_size  # Get training batch size
        dt = self.X.dtype  # Get dtype
        Xn = np.zeros((N, self.d_s)).astype(dt)  # Observations - (N, self.d_s)
        An = np.zeros((N, self.d_a)).astype(dt)  # Actions (N, self.d_s)
        Rn = np.squeeze(np.zeros((N, self.d_r))).astype(dt)  # (N,), not (N, 1)
        Xn1 = np.zeros((N, self.d_s)).astype(dt)  # New obs - (N, self.d_s)

        # Iterate through N samples and interpolate N new points for batch
        for i, (t, k) in enumerate(
                zip(self.sample_indices, self.nearest_neighbors_indices)):

            # Get components of query point
            (xt, at, rt, x_next_t) = self.X[t, :self.d_s], self.X[t, self.d_s:], \
                                     self.Y[t, :self.d_r], self.Y[t, self.d_r:]

            # Randomly determine if real sample appended, or synthetic sample
            if bool(np.random.binomial(1, self.prob_interpolation)):

                # Sample uniformly at random from nearest neighbor indices
                c = np.random.choice(k[1:self.furthest_neighbor])
                interp_indices.append(c)

                # Selected neighbor point transition
                (xc, ac, rc, x_next_c) = self.X[c, :self.d_s], self.X[c, self.d_s:], \
                                         self.Y[c, :self.d_r], self.Y[c, self.d_r:]

                # Interpolate current state, action, and reward
                if self.use_smote or self.use_mixup:  # Use SMOTE or Mixup
                    if self.use_smote:  # Use SMOTE combination coefficients
                        b = np.random.random()
                    elif self.use_mixup:  # Use Mixup sampling
                        b = np.random.beta(self.mixup_alpha, self.mixup_alpha)

                    # Keep track of interpolation coefficients
                    bs.append(b)  # Append Interpolation coefficient

                    # Linearly interpolate sample and neighbor points
                    xn = np.add(b * xt, (1-b) * xc)  # Interpolated obs
                    an = np.add(b * at, (1-b) * ac)  # Interpolated action
                    rn = np.add(b * rt, (1-b) * rc)  # Interpolated reward

                    # Use the provided delta obs
                    if self.use_delta:
                        delta_xt = b * x_next_t
                        delta_xc = (1-b) * x_next_c
                        delta_xn = np.add(delta_xt, delta_xc)

                    # Interpolate new obs by finding the change in new obs
                    else:
                        delta_xt = b * np.subtract(x_next_t, xt)
                        delta_xc = (1-b) * np.subtract(x_next_c, xc)
                        delta_xn = np.add(delta_xt, delta_xc)

                    # Finally, interpolate new delta
                    x_next_n = np.add(xn, delta_xn)  # Interpolated next obs

                else:  # Interpolate using arithmetic mean
                    bs.append(0.5)
                    xn = np.add(xt, xc) / 2
                    an = np.add(at, ac) / 2
                    rn = np.add(rt, rc) / 2

                    # Use the provided delta obs
                    if self.use_delta:
                        delta_xn = np.add(x_next_t, x_next_c) / 2

                    # Interpolate new obs by finding the change in new obs
                    else:
                        delta_xt = np.subtract(x_next_t, xt)
                        delta_xc = np.subtract(x_next_c, xc)
                        delta_xn = np.add(delta_xt, delta_xc) / 2

                    # Finally, interpolate new delta
                    x_next_n = np.add(xn, delta_xn)

                # Add synthetic sample to output arrays for storing interp batch
                Xn[i] = xn  # Interpolated obs
                An[i] = an  # Interpolated action
                Rn[i] = rn  # Interpolated reward
                Xn1[i] = x_next_n  # Interpolated next obs

            else:  # Simply sample a true sample from the replay buffer
                # Add real sample to output arrays
                Xn[i] = xt  # Sampled obs
                An[i] = at  # Sampled action
                Rn[i] = rt  # Sampled reward
                Xn1[i] = x_next_t  # Sampled next obs

        # Get neighbor weights, if we perform an interpolated_replay update
        if self.interp_prio_update:
            neighbor_priorities = self.replay_buffers[self.policy_id].priorities[interp_indices]
            sample_priorities = self.replay_buffers[self.policy_id].priorities[self.sample_indices]

        # Now transform array into SampleBatch with SampleBatchWrapper
        X_interpolated = (Xn, An, Rn, Xn1)

        # Set variables to None if they don't exist
        weights = None  # Likelihood weights, not IS weights

        if not self.interp_prio_update:
            neighbor_priorities = None
            sample_priorities = None

        # Stop timing interpolation
        time_end_interpolation = time.time()
        self.interpolation_time += time_end_interpolation - \
                                   time_start_interpolation
        self.average_interpolation_time = self.interpolation_time / self.step_count

        return X_interpolated, weights, interp_indices, bs, \
               sample_priorities, neighbor_priorities
