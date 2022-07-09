"""Utility functions for preprocessing and postprocessing interpolation
to produce minibatches used for training the DRL agents."""
# External Python packages
import numpy as np


def check_dims(A):
    """Function for ensuring components of experience trajectories have the
    appropriate dimensions.

    Parameters:
        A (np.array): Array corresponding to trajectory component.

    Returns:
        A' (np.array): If the number of dimensions of A is less than 2,
            returns an "expanded dimensions" version of A that has a
            dimension of 1 appended to its end, i.e. if A.shape is (N, ),
            then this returns A' such that A'.shape is (N, 1).
    """
    if len(A.shape) < 2:
        return np.expand_dims(A, axis=-1)
    else:
        return A


def add_experience_noise(dataset, noise_multiple=0.05):
    """Method for perturbing sampled experience with random Gaussian noise.

    Parameters:
        dataset (dict): Dictionary containing the transitions from the
            environment, stored in the replay buffer.
        noise_multiple (float): Value we multiply the covariance of each
            component of the transitions by when adding noise - i.e. how much
            we perturb the system by a zero-mean MV normal that has a multiple
            of the covariance of the transitions. Defaults to 0.05.

    Returns:
        noisy_states (np.array): Array corresponding to current states of experience
            that have been perturbed with zero-mean noise and covariance of the
            states scaled by noise_multiple.
        at (np.array): Array corresponding to actions of experience that have
            been perturbed with zero-mean noise and covariance of the
            actions scaled by noise_multiple.
        rt (np.array): Array corresponding to rewards of experience that have
            been perturbed with zero-mean noise and covariance of the
            rewards scaled by noise_multiple.
        st1 (np.array): Array corresponding to next states of experience that
            have been perturbed with zero-mean noise and covariance of the
            next states scaled by noise_multiple.
    """
    # Get observations, actions, rewards, and next states from samples
    states = dataset["obs"]
    controls = dataset["actions"]
    rewards = dataset["rewards"]
    next_states = dataset["new_obs"]

    # Add noise to experience:
    # States
    cov_states = np.cov(states) * noise_multiple
    noise_states = np.random.multivariate_normal(
        np.zeros(states.shape[1]), cov_states, size=states.shape[0])
    noisy_states = np.add(states, noise_states)

    # Actions
    cov_controls = np.cov(controls) * noise_multiple
    noise_controls = np.random.multivariate_normal(
        np.zeros(controls.shape[1]), cov_controls, size=controls.shape[0])
    noisy_controls = np.add(controls, noise_controls)

    # Rewards
    cov_rewards = np.cov(rewards) * noise_multiple
    noise_rewards = np.random.multivariate_normal(
        np.zeros(rewards.shape[1]), cov_rewards, size=rewards.shape[0])
    noisy_rewards = np.add(rewards, noise_rewards)

    # Next states
    cov_next_states = np.cov(next_states) * noise_multiple
    noise_next_states = np.random.multivariate_normal(
        np.zeros(next_states.shape[1]), cov_next_states,
        size=next_states.shape[0])
    noisy_next_states = np.add(next_states, noise_next_states)

    return noisy_states, noisy_controls, noisy_rewards, noisy_next_states


def str2bool(v):
    """Function for parsing a boolean from command-line parameters (argparse).

    Parameters:
        v (str): A string corresponding to a boolean.

    Returns:
        v_bool (bool): A boolean corresponding to True or False depending on
            the input string provided to the CLI parser.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
