"""Argument parser for custom_trainer. Provides a CLI for complete parameter
customization for running ablation studies on various replay buffers, RL agents,
and environments."""
# Native Python packages
import argparse

# Custom Python packages/modules
import utils.execution.preprocessing_utils as pre_util


def parse_arguments():
    """Creates a command-line based argument parser.

    Returns:
        args (Object): Object whose members/attributes are defined by the
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Argument Parser to create RL configuration "
                    "for training agents with BIER."
    )

    # Training and trainer args
    parser.add_argument("-local_dir", "--local_dir", type=str, default="~/ray_results",
                        help="File location to save training results.")
    parser.add_argument("-log_dir", "--log_dir", type=str, default="logging",
                        help="File location to save custom logging results.")
    parser.add_argument("-record_env", "--record_env", action="store_true",
                        help="Whether to record evaluation episodes. Defaults to False.")
    parser.add_argument("-checkpoint_freq", "--checkpoint_freq", type=int, default=500,
                        help="Number of episodes between saving training checkpoints.")
    parser.add_argument("-monitor", "--monitor", action="store_true",
                        help="Flag to generate gym renderings of rollouts.")
    parser.add_argument("-sleep_time", "--sleep_time", type=int, default=0,
                        help="Time (in seconds) to sleep before starting. Used "
                             "to ensure that different ray servers can be created.")
    parser.add_argument("-ts_it", "--timesteps_per_iteration", type=int, default=1000,
                        help="Timestep interval for which metrics are reported.")
    parser.add_argument("-trainer", "--trainer", type=str, default="SAC",
                        help="Name of trainer to use.")
    parser.add_argument("-num_workers", "--num_workers", type=int, default=0,
                        help="The number of parallel rollout workers.")
    parser.add_argument("-agent_name", "--agent_name", type=str, default="ISAC",
                        help="Name of agent. Experiment results will be placed in"
                             "the folder at path local_dir/agent_name.")
    parser.add_argument("-restore_path", "--restore_path", type=str, default="-1",
                        help="Path for loading desired rllib policy onto trainer.")
    parser.add_argument("-env", "--env", type=str, default="InvertedPendulum-v2",
                        help="Name of environment to use, including version number")

    # Environment and rollout args
    parser.add_argument("-horizon", "--horizon", default=None, type=int,
                        help="Horizon for training.")
    parser.add_argument("-n_step", "--n_step", default=1, type=int,
                        help="Number of steps for reward computation.")
    parser.add_argument("-evaluation_num_episodes", "--evaluation_num_episodes",
                        type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument("-evaluation_interval", "--evaluation_interval",
                        type=int, default=1, help="Number of iterations (100 steps)"
                                                  " between evaluation episodes.")

    # Training args
    parser.add_argument("-num_sgd_iter", "--num_sgd_iter", type=int, default=1,
                        help="Number of Stochastic Gradient Descent (SGD) iterations"
                             "to do on each step")
    parser.add_argument("-round_robin_weights", "--round_robin_weights", type=int,
                        default=1, help="Number of times to sample from replay buffer and update "
                                        "relative to sampling new data.")
    parser.add_argument("-target_by_steps_sampled", "--target_by_steps_sampled",
                        action="store_false", help="Calculate when to update "
                                                   "target network according to"
                                                   "steps sampled.")
    parser.add_argument("-framework", "--framework", type=str, default="torch",
                        help="Backend (torch, tf, or tf2) to use. Defaults to torch.")

    # Replay buffer structure
    parser.add_argument("-buffer_size", "--buffer_size", default=1000000,
                        type=int, help="Number of samples that can be stored in replay buffer")
    parser.add_argument("-batch", "--train_batch_size", type=int, default=256,
                        help="Number of samples in each training minibatch through "
                             "which to train the agent.")
    parser.add_argument("-replay_sequence_length", "--replay_sequence_length",
                        type=int, default=1,
                        help="Step duration to replay from buffer.")
    parser.add_argument("-rollout_fragment_length", "--rollout_fragment_length",
                        type=int, default=1,
                        help="Number of steps in a rollout.")
    parser.add_argument("-whole_episodes", "--whole_episodes", action="store_true",
                        help="Whether to store entire episodes in the replay buffer"
                             "at a time.")
    parser.add_argument("-use_queue", "--use_queue", action="store_true",
                        help="Whether to use a queue for storing predictions.")


    # Replay buffer interpolation
    parser.add_argument("-knn", "--kneighbors", type=int, default=50,
                        help="Number of neighbors to use for Nearest Neighbor Fitting.")
    parser.add_argument("-prioritized_replay", "--prioritized_replay",
                        action="store_true", help="Whether to use "
                                                  "Prioritized Experience Replay")
    parser.add_argument("-custom_replay_buffer", "--custom_replay_buffer",
                        type=pre_util.str2bool, default=False,
                        help="Flag to use an interpolated_replay replay buffer.")
    parser.add_argument("-prob_interpolation", "--prob_interpolation",
                        type=float, default=1.0,
                        help="If sampling from interpolated_replay replay buffer, this "
                             "is the probability of interpolating a given sample.")
    parser.add_argument("-smote", "--smote_interpolation", action="store_true",
                        help="Whether to use SMOTE-like interpolation along two points.")
    parser.add_argument("-mixup", "--mixup_interpolation", action="store_true",
                        help="Whether to use the interpolation mechanism used in "
                             "Mixup. If using, please look at the \alpha value"
                             "(another CLI parameter, defaults to 0.75).")
    parser.add_argument("-mixup_alpha", "--mixup_alpha", type=float, default=0.75,
                        help="Hyperparameter used to determine degree of interpolation."
                             "Defaults to 0.75.")
    parser.add_argument("-k_max", "--furthest_neighbor", type=int, default=-1,
                        help="Furthest neighbor to consider. Note that if this"
                             "exceeds k_neighbors, it will throw an Exception.")
    parser.add_argument("-use_delta", "--use_delta", action="store_true",
                        help="Whether to use delta of next states for the interpolation task.")
    parser.add_argument("-interp_prio_update", "--interp_prio_update", action="store_true",
                        help="Whether to update priorities of both points used in interpolation.")
    parser.add_argument("-is", "--use_importance_sampling", action="store_true",
                        help="Whether to use importance sampling with PER.")
    parser.add_argument("-only_noise", "--only_noise", action="store_true",
                        help="Flag to interpolate samples by only adding"
                             "Gaussian-distributed noise to states, actions, "
                             "and rewards.")
    parser.add_argument("-uniform_replay", "--uniform_replay", action="store_true",
                        help="Flag to sample uniformly from the replay buffer without"
                             "interpolation.")
    parser.add_argument("-noise_multiple", "--noise_multiple", type=float, default=0.0,
                        help="Level of noise to scale Cov for GP. Defaults to 0.0")
    parser.add_argument("-interp_decay_rate", "--interp_decay_rate", type=float,
                        default=0.0, help="Rate of decay for interpolation probability"
                                          "with respect to number of transitions added to"
                                          "the replay buffer.")
    # Hyperparameter setting
    parser.add_argument("-single_model", "--single_model", action="store_true",
                        help="Whether to use a single GPR model for the entire replay buffer.")
    parser.add_argument("-global_hyperparams", "--global_hyperparams", action="store_true",
                        help="Whether to use a global set of hyperparameters for a dataset.")
    parser.add_argument("-warmstart_global", "--warmstart_global", action="store_true",
                        help="Whether to retrain global hyperparameters from initialization"
                             "with the previous iteration's hyperparameters.")
    parser.add_argument("-multi_sample", "--multi_sample", action="store_true",
                        help="Whether to sample all neighbors from replay buffer.")

    # Hyperparameter optimization
    parser.add_argument("-use_botorch", "--use_botorch", action="store_true",
                        help="Whether to optimize with L-BFGS using Botorch.")
    parser.add_argument("-normalize", "--normalize", action="store_true",
                        help="Whether to normalize features and targets.")
    parser.add_argument("-std_x", "--standardize_gpr_x", action="store_true",
                        help="If using GPR for interpolation, whether to"
                             "standardize the features before interpolation.")
    parser.add_argument("-train_size", "--train_size", type=int, default=1000,
                        help="If training GP on one dataset, the number of points to "
                             "sample from the replay buffer.")
    parser.add_argument("-mc_hyper", "--mc_hyper", action="store_true",
                        help="Whether to train global hyperparameters on a Monte "
                             "Carlo estimate of clusters.")
    parser.add_argument("-retrain_interval", "--retrain_interval", type=int, default=1000,
                        help="Number of calls to replay before we retrain GPR.")
    parser.add_argument("-est_lengthscales", "--est_lengthscales", action="store_true",
                        help="Whether to estimate the lengthscales of "
                             "each neighborhood.")
    parser.add_argument("-est_lengthscale_constant",
                        "--est_lengthscale_constant", type=float, default=2.0,
                        help="Radius multiple to consider for local "
                             "neighborhood lengthscale estimates.")
    parser.add_argument("-update_hyperparams", "--update_hyperparams",
                        action="store_true", help="Whether to save and update"
                                                  "the hyperparams.")

    # Kernel Parameters
    parser.add_argument('-use_priors', "--use_priors", action="store_true",
                        help="Whether to use prior distributions for "
                             "kernel hyperparameters.")
    parser.add_argument("-use_ard", "--use_ard", action="store_true",
                        help="Whether to use ARD for GPR when fitting points.")
    parser.add_argument("-composite_kernel", "--composite_kernel",
                        action="store_true",
                        help="Whether to split kernel computations by states and actions.")
    parser.add_argument("-kernel", "--kernel", type=str, default="rq",
                        help="Type of kernel to use for optimization. Defaults to "
                             "Rational Quadratic kernel ('rq'). Other options include RBF "
                             "(Radial Basis Function)/SE (Squared Exponential) ('rbf'), "
                             "and Matern ('matern'). If using 'matern', please set the"
                             "'matern_nu' variable below.")
    parser.add_argument("-matern_nu", "--matern_nu", type=float, default=2.5,
                        help="Nu hyperparameter value to use for the matern kernel.")

    # Mean Parameters
    parser.add_argument("-mean_type", "--mean_type", type=str, default="zero",
                        help="Type of mean function to use for Gaussian Process."
                             "Defaults to constant mean ('constant'). Other options:"
                             "linear ('linear'), and zero ('zero').")

    # Gaussian Process
    parser.add_argument("-gp", "--gaussian_process", action="store_true",
                        help="Whether to use a Gaussian Process for interpolation.")

    parser.add_argument("-gp_cov", "--sample_gp_cov", action="store_true",
                        help="Whether to sample from GP using mean and covariance,"
                             "or just mean (in the case where this flag is false.")
    parser.add_argument("-weighted_updates", "--weighted_updates", action="store_true",
                        help="Whether to weight gradient updates probabilistically according"
                             "to the variance of a sample from the GP.")
    parser.add_argument("-gp_pca", "--gp_pca", action="store_true",
                        help="Whether to use PCA to reduce dimensionality of data"
                             "we fit for the Gaussian Process.")
    parser.add_argument("-gp_pca_components", "--gp_pca_components", type=int,
                        default=2, help="Number of PCA components to use for"
                                        "reducing dimensionality of GP")

    # Debugging
    parser.add_argument("-local_mode", "--local_mode", action="store_true",
                        help="Flag to run on a single process (e.g. for debugging")
    parser.add_argument("-debug_plot", "--debug_plot", action="store_true",
                        help="Flag to generate plots used to access interpolation.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Whether to print training progress and information"
                             "as frequently to the console.")

    # Reproducibility:
    parser.add_argument("-seed", "--seed", type=int, default=42,
                        help="Random seed to produce results with.")

    # Metrics customization
    parser.add_argument("-episodes_smoothing", "--metrics_smoothing_episodes",
                        type=int, default=10, help="Number of episodes to smooth over when"
                                                  "reporting mean reward.")

    # Performance
    parser.add_argument("-gpytorch", "--gpytorch", action="store_true",
                        help="Whether to use GPyTorch for GP computation.")
    parser.add_argument("-cpu_only", "--cpu_only", action="store_true",
                        help="Whether to only use CPU, even if GPU is available.")
    parser.add_argument("-fp32", "--fp32", action="store_true",
                        help="Whether to run GPR code in single-precision (float32)."
                             "Else, defaults 5o double-precision (float64).")

    # Interpolation Evaluation
    parser.add_argument("-perform_holdout_eval", "--perform_holdout_eval",
                        action="store_true", help="Whether to perform holdout evaluation.")
    parser.add_argument("-holdout_split", "--holdout_split", type=float,
                        default=0.2, help="If holdout splitting enabled, the "
                                          "fraction of cluster data to reserve "
                                          "for analytic_evaluation points.")
    # Baselines
    parser.add_argument("-ct", "--ct_baseline", action="store_true",
                        help="Whether to use the Continuous Transition baseline.")
    parser.add_argument("-s4rl", "--s4rl_baseline", action="store_true",
                        help="Whether to use the S4RL Mixup baseline.")
    parser.add_argument("-knn_baseline", "--knn_baseline", action="store_true",
                        help="Whether to use a K-Nearest Neighbor baseline "
                             "(an extreme case of Gaussian Process Regression).")
    parser.add_argument("-naive_mixup", "--naive_mixup_baseline",
                        action="store_true", help="Use a Naive Mixup sampling"
                                                  "baseline.")
    parser.add_argument("-noisy_replay", "--noisy_replay_baseline",
                        action="store_true", help="Use a noisy experience replay"
                                                  "baseline.")
    parser.add_argument("-man_testing", "--on_manifold_testing",
                        action="store_true",
                        help="Whether we are performing on-manifold testing.")

    return parser.parse_args()
