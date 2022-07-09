"""Script for computing the distance from a manifold to a given transition,
measured according to different distance metrics.

This analysis is defined to be done as a mechanism for evaluating how
"on-manifold" a given interpolation scheme is. As such, it is defined only
for evaluating experience replay methods that interpolate observed transitions.
"""

def compute_distance_to_manifold(X_interp, env_aux):
    """Given a batch of transitions X, computes the mean distances
    to the underlying manifold.

    Parameters:
        X (tuple): A tuple composed of np.arrays of:
            (observations, actions, rewards, next states).
            These transitions are likely interpolated.
        env (gym.Env): A Gym environment that can be used as an auxiliary
            environment to calculate the distance from the manifold.
    """
    # Separate transitions into component arrays
    (S, A, R_int, S_p_int) = X_interp

    # Compute the true reward and next state for interpolated transitions
    S_p_true, R_true, _, _ = env_aux.step()

    # Now compute different distance metrics
    next_state_diff = S_p_int - S_p_true  # Difference in next state
    reward_diff = R_int - R_true  # Difference in reward
