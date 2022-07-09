"""Python base class for Interpolated Experience Replay (IER). This class contains
methods and members for interpolating minibatches of data from the replay buffer
intelligently in order to train the agent, and is fully integrated with the RLlib
API (see custom_training.py).

This base class inherits from the general ReplayBuffer object (base_replay_buffer.py)
and defines additional methods that help for the computation of neighbors,
appending to arrays, etc.
"""

# External Python packages
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.tensorboard import SummaryWriter
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, \
    DEFAULT_POLICY_ID

# Native Python packages
import time
import os
import pickle
import copy

# Ray/RLlib/CUDA
_ALL_POLICIES = "__all__"
USE_CUDA = torch.cuda.is_available()

# Custom Python packages/modules
from utils.execution.performance_utils import determine_device
from utils.execution.preprocessing_utils import add_experience_noise
from utils.visualization.visualization import pca_heatmap, pca_1D
from utils.neighbors.faiss import FaissKNeighbors
from utils.neighbors.composite_norm import composite_product_norm
from utils.execution.execution_utils import set_seeds
from replay_buffers.base_replay_buffer import LocalReplayBuffer
from parameters import LEARNING_STARTS
import parameters
#from evaluation.distance_to_manifold import distance_to_manifold


class InterpolatedReplayBuffer(LocalReplayBuffer):
    """Class for a modified replay buffer. Inherits from LocalReplayBuffer,
    and uses a modified sampling function for sampling interpolated/augmented
    samples from the replay buffer.

    This is the parent class for the custom replay buffers defined for this
    module. This function contains all command-line parameter configurations
    for all the base classes, as well as utility functions to help with the
    storage of arrays, computation and querying of nearest neighbors,
    post-processing of interpolated batches, and visualization and logging.

    The subclasses of this function all crucially implement the
    interpolate_samples method, which is intentionally left blank in this class.
    Defining this method, which defines how interpolated samples are produced,
    constitutes the creation of a new replay buffer.

    Parameters:
        prob_interpolation (float): The probability (in range (0, 1)) of
            interpolating a sample.
        gaussian_process (bool): Whether or not to use a Gaussian Process
            Regressor for interpolation. Defaults to 1.0.
        only_noise (bool): Whether to not interpolate, and simply add noise to
            the sampled transitions from the replay buffer when sampling a
            minibatch for training. Defaults to False.
        uniform_replay (bool): Whether to sample from the replay buffer
            uniformly, rather than use a scheme such as prioritized experience
            replay.
        prioritized_replay (bool): Whether to select samples from the replay
            buffer using probabilistic weights determined with PER.
        kneighbors (int): The number of neighbors to consider when computing
            queries for interpolation. Defaults to 2.
        gpu_devices (list): A list of strings to GPUs to use for computing
            interpolated samples.
        noise_multiple (float): ...
        sample_gp_cov (bool): Whether or not to sample from the GPR when
            producing samples for the training minibatch. Defaults to False.
        gp_pca_components (int): If using PCA with GP, this defines the number
            of PCA components to use. Only applicable if gp_pca is True.
        gp_pca (bool): Whether to use PCA with GPR. Defaults to False.
        framework (str): The Deep Learning backend to use with RLlib. Defaults
            to 'torch' for PyTorch. Other option is 'tf2' for TensorFlow-2.
            PyTorch is recommended if using GPR, since GPRs are created and trained
            using the GPyTorch library.
        seed (int): The random seed used for training. Defaults to 42.
        gpytorch (bool): Whether to use GPyTorch for training.
        use_cuda (bool): Whether to use CUDA with PyTorch for hardware
            acceleration.
        use_kd_tree (bool): Whether to use a KDTree for querying nearest neighbors.
            If False and use_faiss is False., uses naive K-Nearest Neighbors object.
            Defaults to False.
        use_faiss (bool): Whether to use Faiss for querying nearest neighbors.
            Defaults to True.
        min_size (int): The minimum size of the replay buffer before minibatches are
            sampled (i.e. the number of samples that are taken before learning begins.)
        weighted_updates (bool): Whether likelihood-weighted updates are used
            when training the GPR. Only applicable if gpytorch is True.
        smote_interpolation (bool): Whether to perform query point selection and
            interpolation in a SMOTE-like fashion. Defaults to False.
        mixup_interpolation (bool): Whether to perform query point selection and
            interpolation in a Mixup-like fashion. Defaults to False.
        mixup_alpha (float): Hyperparameter used for Mixup. Only applicable
            if Mixup is used for interpolation. Defaults to 0.5.
        log_dir (str): The directory where results will be logged. Defaults to
            '~/ray_results/`.
        multi_sample (bool): Whether to sample an interpolated point from every
            sample point-neighbor combination. This means that we will have a
            replay ratio of replay_ratio <-- replay_ratio * kneighbors. However,
            because this hardly impacts inference
        interp_prio_update (bool): Whether to update the priorities of both points
                used in interpolating a new sample (the sampled and interpolated point).
                Defaults to False.
        single_model (bool): Whether to use a single GPR model for sampling new
            points. Defaults to False.
        retrain_interval (int): The number of new transitions added to the replay
            buffer before a global model (either for hyperparameters-only or for
            performing inference) before the global model is retrained. Defaults
            to 1000, the episode length for most environments we consider.
        train_size (int): The number of samples that a global model is fit to.
            Defaults to 1000.
        replay_ratio (int): The ratio of replay operations/samples added to the
            replay buffer. Defaults to 1.
        use_ard (bool): Whether to use Automatic Relevance Determination (ARD)
            for GPR when fitting points. Defaults to False.
        composite_kernel (bool): Whether to split kernel computations by
            states and actions, as is with Contextual Bandits. Defaults to False.
        kernel (str): Type of kernel to use for optimization. Defaults to "
            "Matern kernel ('matern'). Other options include RBF
            (Radial Basis Function)/SE (Squared Exponential) ('rbf'), and RQ
            (Rational Quadratic) ('rq').
        mean_type (str): Type of mean function to use for Gaussian Process.
            Defaults to zero mean ('zero'). Other options: linear
            ('linear'), and constant ('constant').")
        global_hyperparams (bool): Whether to use a single set of hyperparameters
            over an entire model. Defaults to False.
        warmstart_global (bool): If using global hyperparameters, whether to retrain
            the model's parameters from scratch each time. Defaults to False.
        use_botorch (bool): Whether to optimize with L-BFGS using Botorch.
        use_lbfgs (bool): Whether to use L-BFGS with PyTorch optimizers.
        normalize (bool): Whether to perform normalization/standardization of
            the features and targets. Only need to specify if GPR is used for
            interpolation. Defaults to False. Defaults to False.
        standardize_gpr_x (bool): Whether to standardize the input features
            for performing GPR interpolation. Only has an effect if normalize is
            also set to True. Defaults to False, in which case the features are
            min-max normalized.
        use_priors (bool): Whether to use prior distribution over lengthscale and
            outputscale hyperparameters. Defaults to False.
        knn_baseline (bool): Whether to use a K-Nearest Neighbor baseline
            (an extreme case of Gaussian Process Regression).
        cpu_only (bool): Whether to run everything on the CPU, regardless of
            whether a GPU is available. Defaults to False.
        environment (str): Name of the RL environment in which the agent is
            being trained.
        est_lengthscales (bool): Whether to estimate the lengthscales of each
            cluster of neighborhood by finding the farthest point in each dimension.
            Defaults to False.
        debug_plot (bool): Whether to create plots for debugging. Defaults to False.
        interp_decay_rate (float): The decay rate of the interpolation w.r.t.
            number of transitions sampled. Defaults to 0.0.
        matern_nu (float): Value in set if {1/2, 3/2, 5/2} that denotes the power
            to raise the matern kernel evaluation to. Smaller values allow for
            greater discontinuity. Only relevant if kernel is matern. Defaults to
            2.5.
        est_lengthscale_constant (float): Value which we multiply estimated lengthscales
            by. Defaults to 0.5.
        restore_path (str): If restoring from an old replay buffer, the path to load the
            replay buffer from. Defaults to None.
        perform_holdout_eval (bool): Whether to perform holdout evaluation to
            measure interpolation error.
        holdout_freq (int): The frequency at which we should perform holdout
            evaluation.
        fp32 (bool): Whether to run GPR code in single-precision (float32).
            Else, defaults 5o double-precision (float64). Defaults to False.
        log_tb (bool): Whether to log to Tensorboard directory. Defaults to True.
        verbose (bool): Whether to explicitly print statements to console.
            Defaults to False.
        furthest_neighbor (int): The furthest neighbor index, as an integer, to
            consider for selecting a query point for interpolation. Defaults to
            -1, in which case all neighbors are considered.
    """
    def __init__(self, prob_interpolation=1.0, gaussian_process=False, only_noise=False,
                 uniform_replay=False, prioritized_replay=False, kneighbors=2,
                 gpu_devices=[], noise_multiple=0.0, sample_gp_cov=False,
                 gp_pca_components=2, gp_pca=False, framework="torch", seed=42,
                 gpytorch=False, use_cuda=USE_CUDA, use_kd_tree=False,
                 use_faiss=True, min_size=1500, weighted_updates=False,
                 smote_interpolation=False, mixup_interpolation=False,
                 mixup_alpha=1.0, log_dir="logging", tb_dir=".",
                 multi_sample=False, interp_prio_update=False,
                 use_importance_sampling=False, single_model=False,
                 retrain_interval=1000, train_size=1000, replay_ratio=1,
                 use_ard=False, composite_kernel=False, kernel='matern',
                 mean_type='constant', global_hyperparams=False, warmstart_global=False,
                 use_botorch=False, use_lbfgs=False, normalize=False,
                 standardize_gpr_x=False, use_delta=False,
                 use_priors=False, knn_baseline=False, cpu_only=False,
                 env="HalfCheetah-v2", est_lengthscales=False,
                 debug_plot=False, interp_decay_rate=0.0, matern_nu=2.5,
                 est_lengthscale_constant=0.5, checkpoint_freq=500,
                 timesteps_per_iteration=100, restore_path=None, mc_hyper=False,
                 perform_holdout_eval=True, holdout_freq=1000, use_queue=False,
                 fp32=False, log_tb=True, verbose=False, wll_max=2.0,
                 furthest_neighbor=-1, on_manifold_testing=False,
                 on_manifold_intervals=None, *args, **kwargs):

        # Call constructor of superclass (ReplayBuffer)
        super().__init__(*args, **kwargs)

        # Interpolation type
        self.prob_interpolation = prob_interpolation  # P(interpolation)
        self.only_noise = only_noise  # Return ground truth with normally-distributed noise
        self.use_delta = use_delta  # Use delta state for the interpolation task
        self.noise_multiple = noise_multiple  # Only applicable for only_noise
        self.uniform_replay = uniform_replay  # Select uniformly at random
        self.prioritized_replay = prioritized_replay  # Use PER
        self.interp_prio_update = interp_prio_update  # Prio update with interp point
        self.use_importance_sampling = use_importance_sampling  # IS with PER
        if not self.prioritized_replay:  # Using Vanilla Replay
            self.interp_prio_update = False  # Interp prio update not needed
            self.use_importance_sampling = False  # Importance sampling not needed
        self.use_smote = smote_interpolation  # Interpolate points with SMOTE
        self.use_mixup = mixup_interpolation  # Interpolate points with Mixup Sampling
        self.interp_decay_rate = interp_decay_rate  # Rate interp decay wrt env interacts

        # Prioritized Experience Replay
        self.prioritized_replay_alpha = kwargs["prioritized_replay_alpha"]  # Alpha coef
        self.prioritized_replay_beta = kwargs["prioritized_replay_beta"]  # Beta coef
        self.prioritized_replay_eps = kwargs["prioritized_replay_eps"]  # Epsilon coef

        # Query point selection
        self.kneighbors = kneighbors + 1  # Add 1 to include query point
        self.furthest_neighbor = furthest_neighbor
        if self.furthest_neighbor > self.kneighbors:
            raise Exception("Error: Furthest Neighbor > Neighbors Selected")
        if self.furthest_neighbor == -1:
            self.furthest_neighbor = None
        self.mixup_alpha = mixup_alpha  # Distribution param for Mixup sampling
        self.use_kd_tree = use_kd_tree  # Use KD-Tree for queries
        self.use_faiss = use_faiss  # Use FAISS for queries

        # Training characteristics
        self.env = env  # RL environment agent is being trained in
        self.current_replay_count = 0  # Number of replays in a step
        self.multi_sample = multi_sample  # Sample from > 1 sample-neighbor pairs
        self.replay_ratio = replay_ratio  # Ratio training steps: env interacts
        if self.multi_sample:  # Sample all neighbor-sample point queries
            self.sample_replays = kneighbors  # Ratio training: sampling new transitions
        else:
            self.sample_replays = 1  # Only train once on sampled points
        self.min_size = LEARNING_STARTS[self.env]  # Min buffer size to replay

        # GPR Hyperparameter Settings
        self.single_model = single_model  # Single, global model for hyperparams/inference
        self.global_hyperparams = global_hyperparams  # Single set of global hyperparams
        self.use_priors = use_priors  # Place prior distributions on kernel hyperparams
        self.warmstart_global = warmstart_global  # Use previous hyperparams as next stage hyperparams
        self.train_size = train_size  # Size training set for global model
        self.retrain_counter = 0  # Init counter for retraining global model
        self.retrain_interval = retrain_interval  # How often to retrain global model
        self.steps_between_retrain = self.retrain_interval * self.replay_ratio
        self.mc_hyper = mc_hyper  # Use MC hyperparameter optimization

        # GPR Hyperparameter Optimization
        self.use_botorch = use_botorch  # Use L-BFGS with optimization_test_functions
        self.use_lbfgs = use_lbfgs  # Use L-BFGS with PyTorch optimizers
        self.normalize = normalize  # Whether to normalize features
        self.standardize_gpr_x = standardize_gpr_x  # Whether to standardize GPR
        self.gp_pca = gp_pca  # Whether to use PCA with Gaussian
        if not self.gp_pca:  # Reduce dimensionality of PCA
            self.gp_pca_components = None
        else:
            self.gp_pca_components = gp_pca_components  # Number of GP PCA components
        self.est_lengthscales = est_lengthscales  # Estimate lengthscales manually
        self.est_lengthscale_constant = est_lengthscale_constant  # Neighborhood of lengthscale

        # GPR Kernel
        self.use_ard = use_ard  # Use ARD for GPR kernel (GPR-only)
        self.composite_kernel = composite_kernel  # Composite state-action GPR kernel
        self.kernel = kernel  # GPR kernel (wrapped with a Scale kernel)
        self.matern_nu = matern_nu  # Nu hyperparameter for Matern discontinuity

        # GPR Mean
        self.mean_type = mean_type  # Type of GPR mean function

        # GPR
        self.gaussian_process = gaussian_process  # Use GPR interpolation
        self.use_gpytorch = gpytorch  # Use gpytorch (else sklearn)
        self.sample_gp_cov = sample_gp_cov  # Samples mean/covariance from GPR
        self.weighted_updates = weighted_updates  # Perform weighted updates
        self.wll_max = wll_max  # Maximum clipped likelihood weight

        # Baselines
        self.knn_baseline = knn_baseline

        # On-manifold testing
        self.on_manifold_testing = on_manifold_testing
        print("ON-MANIFOLD TESTING? {}".format(self.on_manifold_testing))

        # Create storage for qpos and qvel
        if self.on_manifold_testing:

            # Set the "mix-in" samples to be 100K
            self.min_size = int(1e5)

            # Adjust the batch size for better statistical accuracy, init
            self.replay_batch_size = 1000
            print("ON-MANIFOLD")
            self.Qpos = []
            self.Qvel = []

        if on_manifold_intervals is None:
            self.on_manifold_intervals = [100, 1000, 10000, 100000]
        else:
            self.on_manifold_intervals = on_manifold_intervals

        # Device management
        self.gpu_devices = gpu_devices  # List of GPU devices
        self.framework = framework  # Either "torch", "tf2", or "tf"
        self.device = determine_device(self.gpu_devices, self.framework,
                                       cpu_only=cpu_only)  # GPU or CPU
        self.cpu_only = cpu_only  # Only use CPU, regardless of GPU
        self.use_cuda = (use_cuda) and (len(gpu_devices) > 0) and (not self.cpu_only)
        if self.use_cuda:  # Use for FAISS
            self.gpu_id = 0  # Note: Only applies to single-GPU setup
        else:
            self.gpu_id = None  # CPU-only
        self.fp32 = fp32  # Whether to execute GPR code in fp32 (single-precision)
        self.debug_plot = debug_plot  # Plug parameters and compared predictions
        self.log_tb = True  # Whether to log to Tensorboard directory

        # Reproducibility (seeding)
        self.seed = seed  # Seed for RNG
        set_seeds(self.seed)  # Sets seeds for RNGs
        self.checkpoint_freq = checkpoint_freq  # How often to checkpoint
        self.timesteps_per_iteration = timesteps_per_iteration  # Sampling steps in each training iter
        self.restore_path = restore_path  # Path of replay buffer to restore

        # Tensorboard writer and logging
        self.log_dir = log_dir
        self.replay_buffer_checkpoint_dir = os.path.join(
            log_dir, "replay_buffer_checkpoints")
        if not os.path.exists(self.replay_buffer_checkpoint_dir):
            os.makedirs(self.replay_buffer_checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir, max_queue=1e6)  # tb logger
        parameters.TB_WRITER = self.writer  # Global logger for package
        self.verbose = verbose  # Whether to display print messages

        # Log total time in each interpolation routine/sub-routine
        self.time_started = time.time()
        self.interpolation_time = 0  # Total time spent on interpolating
        self.preprocessing_time = 0  # Total time spent on preprocessing
        self.buffer_preprocessing_time = 0  # Total time spent on tree preprocessing
        self.batch_construction_time = 0  # Total time spent on batch construction
        self.per_sample_time = 0  # Total time spent on sampling with PER
        self.prio_update_time = 0  # Total time spent on priority updates
        self.step_count = 0  # Total number of training steps taken
        self.transitions_sampled = 0  # Increment by 1 / RR

        # Log average time in each interpolation routine/sub-routine
        self.average_preprocessing_time = 0
        self.average_interpolation_time = 0
        self.average_buffer_preprocessing_time = 0
        self.average_batch_construction_time = 0
        self.average_query_time = 0
        self.average_per_sample_time = 0
        self.average_prio_update_time = 0
        self.total_query_time = 0
        self.batch_add_count = 0

        # Keep track of GPR operations
        self.gpr_tensor_creation_time = 0
        self.gpr_normalization_time = 0
        self.gpr_model_formation_time = 0
        self.gpr_inference_preprocessing_time = 0
        self.gpr_inference_time = 0
        self.gpr_likelihood_weights_time = 0
        self.gpr_training_time = 0

        # Interpolation quality evaluation
        self.perform_holdout_eval = perform_holdout_eval
        self.holdout_freq = holdout_freq
        self.steps_between_holdout = self.holdout_freq * self.replay_ratio

        # Speeding up interpolation with pre-computed samples (recommended)
        self.use_queue = use_queue

        # Require preprocessing (normalization, neighbor data structures)
        self.preprocessing_required = True

        # Initialize variable for keeping track of dones
        self.episode_step = 0

        # Display messages to console upon initialization
        print("Device: {}".format(self.device))
        print("Using Cuda For Replay Buffer? {}".format(self.use_cuda))
        print("Probability of interpolating: {} \n"
              "Using Only Ground Truth + Noise: {} \n"
              "Uniform Replay Only: {} \n"
              "KNeighbors: {} \n"
              "Noise Multiple for GP: {} \n"
              "GP with PCA: {} \n"
              "GP PCA components: {} \n"
              "USING SMOTE INTERPOLATION: {} \n"
              "USING MIXUP INTERPOLATION: {} \n"
              "USING MEAN INTERPOLATION: {} \n"
              "USING PER: {} \n"
              "FURTHEST NEIGHBOR (None --> Max): {}".format(
            self.prob_interpolation, self.only_noise, self.uniform_replay,
            self.kneighbors, self.noise_multiple, self.gp_pca,
            self.gp_pca_components, self.use_smote, self.use_mixup,
            (not self.use_smote) and (not self.use_mixup),
            self.prioritized_replay, self.furthest_neighbor))

        if self.prioritized_replay:
            print("USING PRIORITIZED EXPERIENCE REPLAY IN INTERPOLATED REPLAY BUFFER")
            print("Prioritized Replay Beta: {} \n"
                  "Prioritized Replay Eps: {} \n".format(self.prioritized_replay_beta,
                                                         self.prioritized_replay_eps))
        else:
            print("NOT USING PRIORITIZED REPLAY IN INTERPOLATED REPLAY BUFFER")
        print("CREATED INTERPOLATED REPLAY BUFFER...")
        print("DEVICE IS: {}".format(self.device))

    def add_batch(self, batch):
        """Modified function that adds a batch of experience from the RL agent.
        This experience is added to the arrays constructed by this class and
        its child classes for efficient computation.

        Once learning has begun, the replay buffer is "preprocessed" every time
        a new batch of experience is added.

        Parameters:
            batch (MultiAgentBatch): A batch of training experience
                collected by Ray workers that will be used for sampling
                minibatches to train the agent with experience replay.
        """
        # Make a copy so the replay buffer doesn't pin plasma memory.
        batch = batch.copy()

        # If we need auxiliary environment, check if we have an ENV attribute
        if self.on_manifold_testing:
            if not hasattr(self, "mujoco_env"):
                self.mujoco_env = copy.deepcopy(parameters.ENV)

        # Handle everything as if multiagent
        if isinstance(batch, SampleBatch):
            batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
        with self.add_batch_timer:
            if self.replay_mode == "lockstep":
                # Note that prioritization is not supported in this mode.
                for s in batch.timeslices(self.replay_sequence_length):

                    # Check and replace dones
                    s = self.check_dones(s)

                    self.replay_buffers[_ALL_POLICIES].add(s, weight=None)
            else:
                for policy_id, b in batch.policy_batches.items():
                    for s in b.timeslices(self.replay_sequence_length):

                        # Check and replace dones
                        s = self.check_dones(s)

                        if "weights" in s:
                            weight = np.mean(s["weights"])
                        else:
                            weight = None
                        self.replay_buffers[policy_id].add(s, weight=weight)

                        # Check number of samples, and display if at an interval
                        self.total_sample_count = len(
                            self.replay_buffers[policy_id]._storage)
                        self.newest_idx = self.total_sample_count - 1

                        if self.total_sample_count == 1:
                            # Check if restoring replay buffer
                            if self.restore_path is not None:
                                self.load_replay_buffer(self.restore_path)

                            # Initialize arrays
                            self.d_s = s["obs"].size
                            self.d_a = s["actions"].size
                            self.d_r = s["rewards"].size

                            # Display replay buffer and space sizes
                            print("OBSERVATION SIZE: {}, "
                                  "ACTION SIZE: {}, "
                                  "REWARD SIZE: {}".format(
                                self.d_s, self.d_a, self.d_r))
                            print("REPLAY BUFFER SIZE: {}".format(
                                self.buffer_size))

                            # Declare the replay buffer
                            self.policy_id = policy_id

                            # Initialize arrays to store transitions
                            # Store observations and actions
                            self.X = np.zeros(
                                (self.buffer_size, self.d_s + self.d_a)).astype(np.float32)
                            # Store rewards and next observations
                            self.Y = np.zeros(
                                (self.buffer_size, self.d_r + self.d_s)).astype(np.float32)
                            self.episode_ids = np.zeros(self.buffer_size)

                            # Create the dataset and loop over contents
                            ds = {}
                            items = list(SampleBatch.concat_samples(
                                self.replay_buffers[policy_id]._storage).items())
                            for (key, val) in items:
                                ds[key] = np.squeeze(
                                    np.zeros((self.buffer_size, val.size)))
                                ds[key][0] = val
                            self.dataset = SampleBatch(ds)

                        else:  # Update dataset with most recent observation

                            for key in self.dataset.keys():
                                self.dataset[key][self.total_sample_count-1] = s[key]

                        if self.total_sample_count % parameters.LOG_INTERVAL == 0:
                            print("NUMBER OF SAMPLES IN REPLAY BUFFER: {}".format(
                                self.total_sample_count))

                        # Update (obs, action) inputs
                        self.X[self.total_sample_count-1, :] = np.hstack(
                            (np.squeeze(s["obs"]), np.squeeze(s["actions"])))

                        # Update qpos, qvel
                        if self.on_manifold_testing:
                            self.Qpos.append(parameters.ENV.sim.get_state().qpos)
                            self.Qvel.append(parameters.ENV.sim.get_state().qvel)

                        # Update (reward, delta_obs/new_obs) outputs
                        if self.use_delta:  # Use delta_obs (recommended)
                            self.Y[self.total_sample_count - 1, :] = np.hstack(
                                (np.squeeze(s["rewards"]),
                                 np.squeeze(s["new_obs"]-s["obs"])))
                        else:  # Use new_obs
                            self.Y[self.total_sample_count - 1, :] = np.hstack(
                                (np.squeeze(s["rewards"]),
                                 np.squeeze(s["new_obs"])))

                        # Update episode IDs
                        self.episode_ids[self.total_sample_count - 1] = s["eps_id"]

                    # Process replay buffer if we have started sampling from it
                    if self.total_sample_count >= self.min_size \
                        and self.preprocessing_required:
                        self._preprocess_replay_buffer()

                    # If running on-manifold testing, check if it's needed
                    if self.on_manifold_testing:
                        if self.total_sample_count in self.on_manifold_intervals:  # Time to compute

                            # Prep the replay buffer
                            self._preprocess_replay_buffer()

                            # Generate an interpolated batch
                            X_interpolated, sample_weights, neighbor_indices, b, \
                            _, _ = self.interpolate_samples()

                            # Get needed indices
                            sample_idx = self.sample_indices

                            # Now compute metrics for it
                            reward_dist, next_state_dist = self.compute_distance_to_manifold(
                                X_interpolated, sample_idx, neighbor_indices, b)
                            print("SAMPLES: {}, REWARD DIST: {}, NEXT STATE DIST: {}".format(
                                self.total_sample_count, reward_dist, next_state_dist))


        # Increment total number of batches added
        self.num_added += batch.count

    def replay(self):
        """Function for modifying the experiences sampled from the replay buffer
        via interpolation.

        This function calls the interpolate_samples method defined in the
        subclasses of this class to generate new samples. These samples are
        then used to construct new MultiAgentBatch objects in this function that
        are then passed back to the Ray actors for training the RL agents.

        Returns:
            batch (MultiAgentBatch): Returns a batch for the number of agents
                in the environment (note this is just one for non-MARL tasks.
        """
        if self._fake_batch:
            fake_batch = SampleBatch(self._fake_batch)
            return MultiAgentBatch({
                DEFAULT_POLICY_ID: fake_batch
            }, fake_batch.count)

        if self.num_added < self.replay_starts:
            return None

        with self.replay_timer:

            # Use to set batches for all policies
            policy_batches = {}

            # Iterate through policies if MARL
            for policy_id, replay_buffer in self.replay_buffers.items():

                # Only add noise to ground truth
                if self.only_noise:
                    # Add noise to experience
                    X_noise, A_noise, R_noise, X_noise1 = add_experience_noise(
                        dataset, self.noise_multiple)

                    # Create a SampleBatch object for noisy data
                    # TODO(rms): Change if this is used
                    noisy_batch = self.sample_batch_wrapper(
                        (X_noise, A_noise, R_noise, X_noise1),
                        self.prioritized_replay)
                    policy_batches[policy_id] = noisy_batch

                # Same as vanilla replay without PER
                elif self.uniform_replay:  # Sample uniformly from replay buffer
                    policy_batches[policy_id] = self._uniformly_sample(dataset)

                # Interpolate samples using ChildClass.interpolate_samples()
                else:
                    # Now interpolate self.replay_batch_size samples
                    X_interpolated, sample_weights, neighbor_indices, b, \
                    sample_priorities, neighbor_priorities = self.interpolate_samples()

                    # Create interpolated batch from interpolated samples
                    interpolated_batch = self.sample_batch_wrapper(
                        X_interpolated, self.sample_indices, self.prioritized_replay,
                        sample_weights=sample_weights,
                        interp_prio_update=self.interp_prio_update,
                        neighbor_indices=neighbor_indices, b=b,
                        sample_priorities=sample_priorities,
                        neighbor_priorities=neighbor_priorities)

                    # Log interpolated batch metrics to tensorboard
                    if self.log_tb and \
                            self.step_count % (parameters.LOG_INTERVAL * self.replay_ratio) == 0:
                        self.compute_batch_metrics(interpolated_batch)

                    # Remove to prevent memory leak
                    del sample_weights

                    # Now add interpolated_batch to policy_batches
                    policy_batches[policy_id] = interpolated_batch

                    # Visualize interpolated batch
                    if self.debug_plot and self.step_count % parameters.LOG_INTERVAL == 0:
                        pca_heatmap(
                            interpolated_batch, self.dataset,
                            self.sample_indices, neighbor_indices, env=self.env)
                        pca_1D(
                            interpolated_batch, self.dataset,
                            self.sample_indices, neighbor_indices, env=self.env)

            # Log metrics
            self.log_and_update_metrics()

            # Determine if we need to checkpoint
            train_steps_per_iter = self.timesteps_per_iteration * self.replay_ratio
            checkpoint_interval = self.checkpoint_freq * train_steps_per_iter
            if (int(self.step_count % checkpoint_interval) == 0) or \
                (self.step_count == parameters.TOTAL_ENV_INTERACTS[self.env]):  # Make sure to get replay buffer on last rollout
                check_pt_count = str(self.step_count // train_steps_per_iter)
                out_path = os.path.join(
                    self.replay_buffer_checkpoint_dir, check_pt_count,
                    "replay_buffer.pickle")
                if not os.path.exists(os.path.dirname(out_path)):
                    os.mkdir(os.path.dirname(out_path))
                print("STEP COUNT --> {} ... SAVING REPLAY BUFFER TO: {}".format(
                    self.step_count, out_path))
                self.save_replay_buffer(out_path)

            # Lastly, make sure global parameters are set
            parameters.REPLAY_RATIO = self.replay_ratio  # Set global parameter
            parameters.RL_ENV = self.env  # Set global parameters

            # Finally, return the batch
            return MultiAgentBatch(policy_batches, self.replay_batch_size)

    def interpolate_samples(self):
        """Base class method for performing interpolation. In this base replay
        buffer, this operation should not be called, and should be overwritten."""
        raise Exception("Base class does not perform interpolation. Please call"
                        "one of the defined interpolated replay buffers.")

    def update_priorities(self, prio_dict):
        """Used for updating priorities used with Prioritized Experience Replay.
        Supports two variants: (i) standard updates, (ii) interpolated updates.

        Method to update the priorities for sampling points using Importance
        Sampling and Prioritized Experience Replay. These updates are made
        using the computed TD-error at each sampled point. Note that
        a small factor of eps has been added to ensure that priorities are always
        greater than zero. This eps is given by self.prioritized_replay_ops.

        Furthermore, two variants of this function are supported:

        (i) Standard update: This performs the standard PER update by completely
            overwriting the priority for the sampled value, and not adjusting
            the priority for the neighbor value.

        (ii) Interpolated update: This performs the PER update by performing an
            update of the current priority proportionally to a {sample, neighbor}
            point's contribution to the formation of the new point. i.e:

            If the new point is given by:

                x_interp = (b * x_sample) + ((1-b) * x_neighbor)

            Then the priorities of x_sample and x_neighbor are updated as follows:

                p_new(x_sample) = b * p(x_interp) + (1 - b) * p_old(x_sample)
                p_new(x_neighbor) = (1-b) * p(x_interp) + b * p_old(x_neighbor)

            This interpolated update ensures that the priorities of both the
            sampled and neighboring points are updated according to their
            contribution to the formation of the interpolated point.

        Parameters:
            prio_dict (dict): Dictionary containing information on batch indices
                and the TD errors associated with each element of the replay buffer.
        """
        with self.update_priorities_timer:

            # Start logging time for priority update
            prio_update_start_time = time.time()

            # (i) Perform "Interpolated" Prioritized Experience Replay update
            if self.interp_prio_update:
                for policy_id, (batch_indexes, neighbor_indexes, b, prev_priorities_samples,
                                prev_priorities_neighbors, td_errors) in prio_dict.items():

                    # Compute TD error of interpolated points
                    interp_priorities = np.abs(td_errors)

                    # Construct new priorities for sample points
                    new_priorities_samples = np.add(
                        np.multiply(b, interp_priorities),
                        np.multiply(1-b, prev_priorities_samples)) + \
                                             self.prioritized_replay_eps

                    # Construct new priorities for neighbor points
                    new_priorities_neighbors = np.add(
                        np.multiply(1-b, interp_priorities),
                        np.multiply(b, prev_priorities_neighbors)) + \
                                               self.prioritized_replay_eps

                    # Stack new priorities to update samples and neighbors
                    concat_priorities = np.concatenate(
                        (new_priorities_samples, new_priorities_neighbors))

                    # Stack indices of samples and neighbors for prio update
                    concat_indices = np.concatenate(
                        (batch_indexes, neighbor_indexes))

                    # Priority update using concatenated priorities and indices
                    self.replay_buffers[policy_id].update_priorities(
                        concat_indices, concat_priorities)

            # (ii) Perform standard Prioritized Experience Replay update
            else:
                for policy_id, (batch_indexes, td_errors) in prio_dict.items():
                    new_priorities = (
                            np.abs(td_errors) + self.prioritized_replay_eps)
                    self.replay_buffers[policy_id].update_priorities(
                        batch_indexes, new_priorities)

            # Start logging time for priority update
            prio_update_end_time = time.time()
            self.prio_update_time += prio_update_end_time - \
                                     prio_update_start_time
            if parameters.GLOBAL_STEP_COUNT % parameters.LOG_INTERVAL == 0:
                self.average_prio_update_time = \
                    self.prio_update_time / (self.step_count + 1)

    def _uniformly_sample(self, dataset):
        """Method for sampling uniformly from the replay buffer. Samples
        uniformly from the given dataset and returns a Sample Batch object.

        Parameters:
            dataset (SampleBatch): A Sample Batch consisting of observations,
                actions, rewards, next states, dones, episode ids, rollout ids,
                agent index, and importance sampling weights.

        Returns:
            sampled_batch (SampleBatch): Object corresponding to the
                uniformly-sampled batch.
        """
        # Get number of samples
        N = dataset["obs"].shape[0]

        # Sample uniformly from replay buffer
        uniform_batch_indices = np.random.choice(
            N, size=self.replay_batch_size, replace=True,)

        # Construct dataset from a uniform sub-sample
        sampled_batch = {}
        for key in list(dataset.keys()):
            sampled_batch[key] = dataset[key][uniform_batch_indices]

        if self.prioritized_replay:  # Only need to return indices if using PER
            sampled_batch["batch_indexes"] = uniform_batch_indices

        return sampled_batch

    def generate_knn(self):
        """Method to generate a K-Nearest Neighbors object
        using the contents of a replay buffer.

        Allows for querying nearest neighbors to generate neighborhoods for
        sampling interpolated batches.
        """
        # Create KNN
        # If using a composite kernel
        if self.composite_kernel and self.gaussian_process:
            self.knn = NearestNeighbors(
                n_neighbors=self.kneighbors, metric=composite_product_norm,
                metric_params={"ds": self.d_s}).fit(
                self.X_norm[:self.total_sample_count])

        else:
            self.knn = NearestNeighbors(n_neighbors=self.kneighbors).fit(
                self.X_norm[:self.total_sample_count])

    def generate_kdtree(self):
        """Method to generate a kdtree using the contents of a replay buffer.

        Allows for querying nearest neighbors to generate neighborhoods for
        sampling interpolated batches. Z_norm is a normalized concatenation of
        (state, action) and has dimensions (N_samples, dim_state + dim_action).
        """
        # Create KDTree
        self.kd_tree = KDTree(self.X_norm[:self.total_sample_count])

    def generate_or_update_faiss(self):
        """Method to generate a faiss-tree using the contents of a replay buffer.

        Allows for querying nearest neighbors to generate neighborhoods for
        sampling interpolated batches (recommended). Z_norm is a normalized
        concatenation of (state, action) and has dimensions
        (N_samples, dim_state + dim_action)."""
        # Create or update FAISS
        if hasattr(self, "faiss"):  # Only need to update with new index
            self.faiss.index.add(self.X_norm[-1:])
        else:  # Need to create FAISS
            self.faiss = FaissKNeighbors(k=self.kneighbors, gpu_id=self.gpu_id)
            self.faiss.fit(self.X_norm)

    def _sample_per(self):
        """Helper function to sample points using Prioritized Experience Replay.
        Automatically updates the weights parameter in the sampled batch in
        order to leverage these Importance Sampling (IS) weights in the gradient
        update step."""
        # Log PER sample time
        per_sample_start_time = time.time()

        sample_batch = self.replay_buffers[self.policy_id].sample(
            self.replay_batch_size, beta=self.prioritized_replay_beta)

        # Add normalized Importance Sampling weights to sample batch
        self.sample_indices = sample_batch["batch_indexes"]
        if self.use_importance_sampling:
            self.dataset["weights"][self.sample_indices] = sample_batch["weights"]

        # Log PER sample time
        per_sample_end_time = time.time()

        # Report PER time
        self.per_sample_time += per_sample_end_time - per_sample_start_time
        if parameters.GLOBAL_STEP_COUNT % parameters.LOG_INTERVAL == 0:
            self.average_per_sample_time = \
                self.per_sample_time / (self.step_count + 1)

    def _query_neighbors(self, precompute=False):
        """Helper method to query neighbors using an updated self.sample_indices
        array. NOTE: This requires that a tree (e.g. KNN, KDTree, or FAISS) is
        already fit with the dataset.

        Parameters:
            precompute (bool): Whether to precompute nearest neighbors, i.e.
                for uniform replay with replay_ratio > 1.
        """
        # Start logging query time
        query_time_start = time.time()

        if precompute:  # Precompute, e.g. if sampling uniformly from buffer

            # Normalized sampled features for computing nearest neighbors
            self.block_sample_features = self.X_norm[self.block_sample_idx, :]

            # Compute nearest neighbor indices - FAISS
            if self.use_faiss:
                self.block_nn_idx = self.faiss.query(
                    self.block_sample_features)
                query_time_end = time.time()

            # Compute nearest neighbor indices - KDTree
            elif self.use_kd_tree:
                self.block_nn_idx = self.kd_tree.query(
                    self.block_sample_features, k=self.kneighbors,
                    return_distance=False, dualtree=True,
                    sort_results=True)
                query_time_end = time.time()

            # Compute neighbors with KNN
            else:
                self.block_nn_idx = self.knn.kneighbors(
                    self.block_sample_features, return_distance=False)
                query_time_end = time.time()

        else:  # Do not precompute, e.g. if using Prioritized Experience

            # Normalized sampled features for computing nearest neighbors
            self.sample_features = self.X_norm[self.sample_indices, :]

            # Compute nearest neighbor indices - FAISS
            if self.use_faiss:
                self.nearest_neighbors_indices = self.faiss.query(
                    self.sample_features)
                query_time_end = time.time()

            # Compute nearest neighbor indices - KDTree
            elif self.use_kd_tree:
                self.nearest_neighbors_indices = self.kd_tree.query(
                    self.sample_features, k=self.kneighbors,
                    return_distance=False, dualtree=True,
                    sort_results=True)
                query_time_end = time.time()

            # Compute neighbors with KNN
            else:
                self.nearest_neighbors_indices = self.knn.kneighbors(
                    self.sample_features, return_distance=False)
                query_time_end = time.time()

        # Timing for computing neighbors
        self.total_query_time += query_time_end - query_time_start
        if parameters.GLOBAL_STEP_COUNT % parameters.LOG_INTERVAL == 0:
            self.average_query_time = self.total_query_time / (self.step_count + 1)

    def _preprocess_replay_buffer(self):
        """Method for pre-processing the dataset for use with our
        sampling function.

        This method preprocesses the replay buffer whenever a new environment
        interaction is added. These steps include:

        1. Normalizing the state and action spaces of the replay buffer using
            Z-score standardization.
        2. Fitting Neighbor/Similarity objects for later neighbor querying.
        3. If using a precomputed BIER buffer, computes the neighbor change
            mask to find which points require retraining on new neighborhoods.
        """
        # Log time needed to preprocess replay buffer
        time_start_buffer_preprocessing = time.time()

        # Normalize data to Standard Normal Range
        # TODO(rms): Determine if robust scaler leads to better performance
        self.X_norm = StandardScaler(with_mean=False).fit_transform(
            self.X[:self.total_sample_count])
        #self.X_norm = RobustScaler(with_centering=False).fit_transform(
        #    self.X[:self.total_sample_count])

        if self.use_faiss:  # FAISS
            self.generate_or_update_faiss()  # Updates self.faiss w/ self.X_norm
        elif self.use_kd_tree:  # KDTree
            self.generate_kdtree()  # Auto-updates self.kdtree with self.X_norm
        else:  # KNN
            self.generate_knn()  # Auto-updates self.knn with self.X_norm

        # If using a queue, compute a mask of points that have changed
        if self.use_queue:
            self.compute_neighbor_change_mask()

        # Log time needed to preprocess replay buffer (below is pre-computation)
        time_end_buffer_preprocessing = time.time()

        # If not using PER, can sample all points for next replay_ratio batches
        if not self.prioritized_replay:

            # Since sampling/querying, log to that timer
            time_start_preprocessing = time.time()

            # Sample next replay_ratio number of batches with replacement
            # Samples until a new environment interaction is added
            self.block_sample_idx = np.random.choice(
                self.total_sample_count, replace=True,
                size=self.replay_batch_size * self.replay_ratio)

            # Now query nearest neighbor queries in batched form
            self._query_neighbors(precompute=True)
            self.precomp_neighbor_counter = 0  # Counter for correct indexing

            # End sample/querying timer
            time_end_preprocessing = time.time()
            self.preprocessing_time += time_end_preprocessing - \
                                       time_start_preprocessing

        # Log timing metrics
        self.buffer_preprocessing_time += time_end_buffer_preprocessing - \
                                          time_start_buffer_preprocessing
        if parameters.GLOBAL_STEP_COUNT % parameters.LOG_INTERVAL == 0:
            self.average_buffer_preprocessing_time = \
                self.buffer_preprocessing_time / (self.step_count + 1)

    def gp_pca(self, X, Y):
        """PCA subroutine that can be used for reducing the dimensionality of
        the data we fit using the Gaussian Process. Takes as input an array X
        and an array Y.

        Parameters:
            X (np.array): Array corresponding to the features used in Gaussian
                Process Regression (GPR).
            Y (np.array): Array corresponding to the targets used in Gaussian
                Process Regression (GPR)

        Returns:
            X_transformed (np.array): Array corresponding to the features after
                PCA has been applied for GPR.
            Y_transformed (np.array): Array corresponding to the targets after
                PCA has been applied for GPR.
        """
        # Create PCA object for X and fit data; store for inverse transform.
        self.pca_x = PCA(n_components=self.gp_pca_components)
        X_transformed = self.pca_x.fit_transform(X)

        # Create PCA object for Y and fit data; store for inverse transform.
        self.pca_y = PCA(n_components=self.gp_pca_components)
        Y_transformed = self.pca_x.fit_transform(Y)

        return X_transformed, Y_transformed

    def sample_and_query(self, sample_points_only=False):
        """Helper function to sample transitions from the replay buffer and query
        the nearest neighbors of these points.

        Parameters:
            sample_points_only (bool): Whether to sample points only, and not
                recompute neighbors. Defaults to False.

        Modifies:
            self.sample_indices (overwrites)
            self.sample_features (overwrites)
            self.nearest_neighbor_indices (overwrites if not sample_points_only)
        """
        # Print the step count
        if self.step_count % parameters.LOG_INTERVAL == 0 and self.verbose:
            print("Steps: {}".format(self.step_count))

        # Start preprocessing timer
        time_start_preprocessing = time.time()

        # Sample from dataset using prioritized experience replay
        if self.prioritized_replay:

            # Sample points with PER
            self._sample_per()

            # Now sample points
            if not sample_points_only:
                self._query_neighbors()

        # Sample uniformly from replay buffer
        else:

            # Get start/end batch indices from precomputed query
            idx_start = self.precomp_neighbor_counter * \
                        self.replay_batch_size
            idx_end = (self.precomp_neighbor_counter + 1) * \
                       self.replay_batch_size

            # Get sample and nearest neighbor indices
            self.sample_indices = self.block_sample_idx[idx_start:idx_end]
            self.nearest_neighbors_indices = self.block_nn_idx[idx_start:idx_end]

            # Increment counter
            self.precomp_neighbor_counter += 1

        # End preprocessing timer
        time_end_preprocessing = time.time()
        self.preprocessing_time += time_end_preprocessing - \
                                   time_start_preprocessing
        if parameters.GLOBAL_STEP_COUNT % parameters.LOG_INTERVAL == 0:
            self.average_preprocessing_time = self.preprocessing_time / (self.step_count + 1)

        # Increment the global step count, and synchronize
        self.step_count += 1
        parameters.GLOBAL_STEP_COUNT = self.step_count  # Sync global step value

        # Check if we need to change probability of interpolation
        if self.step_count > 0 and self.step_count % self.replay_ratio == 0:
            # Decrement interpolation probability
            self.prob_interpolation -= self.interp_decay_rate  # Typically, 0

    def sample_global(self, L, N=None, use_per=False):
        """Method to sample from the replay buffer using either uniform sampling
        (default) or per.

        Parameters:
            L (int): The length of the replay buffer - i.e. the number of
                samples in the buffer.
            N (int): Number of samples to take from the experience replay buffer.
                Defaults to None, in which case the batch size is selected.
            use_per (bool): Whether to sample new points using Prioritized
                Experience Replay (PER).

        Returns:
            training_indices (np.array): An array of indices corresponding to
                the sampled points.
        """
        if N is None:  # If None, just set to batch size
            N = self.replay_batch_size

        # Sample with PER
        if use_per:
            training_indices = self.replay_buffers[self.policy_id].sample(
                N, beta=self.prioritized_replay_beta)["batch_indexes"]

        # Sample points uniformly
        else:
            training_indices = np.random.choice(L, size=N, replace=True)
        return training_indices

    def priority_sample_neighbors(self):
        """Method to sample neighbors of a selected sample using PER.

        Returns:
            interp_indices (np.array): Array of indices corresponding to the
                neighbors we sample.
        """
        # Take all priority weights of neighbors
        all_weights = [
            np.float_power(self.replay_buffers[self.policy_id].priorities[k[1:]],
                           self.prioritized_replay_alpha) for k
            in self.nearest_neighbors_indices]

        # Compute sums of weights and normalize
        summed_weights = [np.sum(aw) for aw in all_weights]
        normalized_weights = [w / s for w, s in
                              zip(all_weights, summed_weights)]

        # Sample neighbor indices according to normalized priority weights
        interp_indices = \
            [np.random.choice(k[1:self.furthest_neighbor], p=n) for k, n in zip(
                self.nearest_neighbors_indices, normalized_weights)]

        return interp_indices

    def sample_batch_wrapper(self, X_interpolated, sample_indices,
                             prioritized_replay, sample_weights=None,
                             interp_prio_update=False, neighbor_indices=None,
                             b=None, sample_priorities=None,
                             neighbor_priorities=None):
        """Transform same-length observation and rewards NumPy arrays into a
        sample batch of trajectory experience.

        Parameters:
            X_interpolated (tuple): Tuple composed of:
                st (np.array): Array corresponding to current states of experience.
                at (np.array): Array corresponding to actions of experience.
                rt (np.array): Array corresponding to rewards of experience.
                st1 (np.array): Array corresponding to next states of experience.
            sample_indices (np.array): Indices of points sampled.
            prioritized_replay (bool): Whether we use PER.
            sample_weights (np.array): Array of weights to use for sampling.
            interp_prio_update (bool): Whether to update the priorities of the
                interpolated points using both the sample and neighbor point.
                Defaults to False.
            neighbor_indices (np.array): Indices of points corresponding to
                neighbors of sampled points.
            b (np.array): Linear combination coefficient. Only relevant if
                using SMOTE or Mixup sampling.
            sample_priorities (np.array): Array of old priority weights for
                sample points.
            neighbor_priorities (np.array): Array of old priority weights for
            neighbor points.

        Returns:
            interpolated_batch (SampleBatch): Returns an updated SampleBatch
                object.
        """
        # Gather timing metrics
        time_start_batch_construction = time.time()

        # Initialize SampleBatch
        interpolated_batch = {}

        # Specific, interpolated elements
        experience_keys = ["obs", "actions", "rewards", "new_obs"]
        mapping = {"obs": X_interpolated[0], "actions": X_interpolated[1],
                   "rewards": X_interpolated[2], "new_obs": X_interpolated[3]}

        # Check if interpolating with a terminal state
        check_terminal_interpolation = (neighbor_indices is not None) and \
                                       (not parameters.NO_DONE_AT_END[self.env])
        if check_terminal_interpolation:

            # Get "done" features for the sample and neighbor indices
            sample_dones = self.dataset["dones"][sample_indices]
            neighbor_dones = self.dataset["dones"][neighbor_indices]
            # Take element-wise or to determine which interp samples to replace
            # Indexed [0, ..., self.replay_batch_size]
            terminal_interpolation_idx = np.logical_or(
                sample_dones, neighbor_dones)
            num_terminal_interpolated_samples = terminal_interpolation_idx.size

        # Iterate over all keys in batch dictionary
        for key in list(self.dataset.keys()):
            if key in experience_keys:  # Set interpolated values directly
                interpolated_batch[key] = mapping[key]

                # Overwrite interpolated samples containing terminal states
                if check_terminal_interpolation:
                    if num_terminal_interpolated_samples > 0:

                        # Indices with terminal state to change back
                        # Indexed [0, ..., self.total_sample_count]
                        idx_to_change = sample_indices[terminal_interpolation_idx]

                        # Get target shape for replacing samples
                        target_shape = interpolated_batch[key][terminal_interpolation_idx].shape

                        # Set interpolated samples to be original samples
                        interpolated_batch[key][terminal_interpolation_idx] = \
                            self.dataset[key][idx_to_change].reshape(target_shape)

            else:  # Slice along the sampled indices
                interpolated_batch[key] = self.dataset[key][sample_indices]

        # Add "batch_indexes" key to interpolated batch
        interpolated_batch["batch_indexes"] = sample_indices

        # Add/update batch_indexes
        if prioritized_replay:  # Only need to return batch_indexes if using PER

            # If using interpolated priority updates
            if interp_prio_update:  # Use sample and neighbor point

                # Add indices for samples and neighbors
                interpolated_batch["neighbor_indexes"] = neighbor_indices
                interpolated_batch["b"] = b

                # Overwrite interpolation coefficient when terminal states exist
                if check_terminal_interpolation:
                    if num_terminal_interpolated_samples > 0:
                        interpolated_batch["b"][terminal_interpolation_idx] = \
                            np.ones(num_terminal_interpolated_samples)

                # Add previous priorities
                interpolated_batch["sample_priorities"] = sample_priorities
                interpolated_batch["neighbor_priorities"] = neighbor_priorities

                # Need to interpolate IS weights (o.w. left as standard)
                if self.use_importance_sampling:
                    w_S = interpolated_batch["weights"]  # Get IS weights of samples
                    w_N = self.replay_buffers[self.policy_id].compute_is_weights(
                        neighbor_indices, beta=self.prioritized_replay_beta)  # IS_N

                    # Construct IS weights of interpolated points as sum of two
                    interpolated_weights = (w_S * b) + (w_N * (1-b))

                    # Normalize weights by dividing by max
                    interpolated_batch["weights"] = interpolated_weights

            # Log IS weights to tensorboard
            if parameters.GLOBAL_STEP_COUNT % parameters.LOG_INTERVAL == 0 \
                and self.use_importance_sampling:
                base_tag = "Importance Sampling Weights/ {} Weights"

                # Get IS weights from batch
                is_weights = interpolated_batch["weights"]

                # Compute min, max, mean, and var of IS weights
                is_min, is_max = np.min(is_weights), np.max(is_weights)
                is_mean, is_var = np.mean(is_weights), np.var(is_weights)

                # Log to tensorboard
                tag_types = ["Min", "Max", "Mean", "Variance"]
                values = [is_min, is_max, is_mean, is_var]
                for t, v in zip(tag_types, values):
                    self.writer.add_scalar(base_tag.format(t), v,
                                           global_step=self.step_count)

        # Add sample weights if using likelihood weighting
        if sample_weights is not None:
            interpolated_batch["sample_weights"] = sample_weights

        # Finish timing batch construction
        time_end_batch_construction = time.time()
        self.batch_construction_time += time_end_batch_construction - \
                                        time_start_batch_construction
        if parameters.GLOBAL_STEP_COUNT % parameters.LOG_INTERVAL == 0:
            self.average_batch_construction_time = \
                self.batch_construction_time / (self.step_count + 1)

        return SampleBatch(interpolated_batch)

    def sample_batch_wrapper_queue(self, obs, actions, rewards, new_obs,
                                   sample_indices, prioritized_replay,
                                   sample_weights=None, interp_prio_update=False,
                                   neighbor_indices=None, b=None,
                                   sample_priorities=None,
                                   neighbor_priorities=None):
        """Transform same-length observation and rewards NumPy arrays into a
        sample batch of trajectory experience. This variant is used for when
        we leverage precomputed samples with BIER.

        Parameters:
            obs (np.array): Array corresponding to current states of experience.
            actions (np.array): Array corresponding to actions of experience.
            rewards (np.array): Array corresponding to rewards of experience.
            new_obs (np.array): Array corresponding to next states of experience.
            sample_indices (np.array): Indices of points sampled.
            prioritized_replay (bool): Whether we use PER.
            sample_weights (np.array): Array of weights to use for sampling.
            interp_prio_update (bool): Whether to update the priorities of the
                interpolated points using both the sample and neighbor point.
                Defaults to False.
            neighbor_indices (np.array): Indices of points corresponding to
                neighbors of sampled points.
            b (np.array): Linear combination coefficient. Only relevant if using
                SMOTE or Mixup sampling.
            sample_priorities (np.array): Array of old priority weights for
                sample points.
            neighbor_priorities (np.array): Array of old priority weights for
                neighbor points.

        Returns:
            interpolated_batch (SampleBatch): Returns an updated SampleBatch
                object.
        """
        # Gather timing metrics
        time_start_batch_construction = time.time()

        # Initialize SampleBatch
        interpolated_batch = {}

        # Specific, interpolated elements
        experience_keys = ["obs", "actions", "rewards", "new_obs"]
        mapping = {"obs": obs, "actions": actions,
                   "rewards": rewards, "new_obs": new_obs}

        # Check if interpolating with a terminal state
        check_terminal_interpolation = (neighbor_indices is not None) and \
                                       (not parameters.NO_DONE_AT_END[self.env])
        if check_terminal_interpolation:

            # Get "done" features for the sample and neighbor indices
            sample_dones = self.dataset["dones"][sample_indices]
            neighbor_dones = self.dataset["dones"][neighbor_indices]
            # Take element-wise or to determine which interp samples to replace
            # Indexed [0, ..., self.replay_batch_size]
            terminal_interpolation_idx = np.logical_or(
                sample_dones, neighbor_dones)
            num_terminal_interpolated_samples = terminal_interpolation_idx.size

        # Iterate over all keys in batch dictionary
        for key in list(self.dataset.keys()):
            if key in experience_keys:  # Set interpolated values directly
                interpolated_batch[key] = mapping[key]

                # Overwrite interpolated samples containing terminal states
                if check_terminal_interpolation:
                    if num_terminal_interpolated_samples > 0:

                        # Indices with terminal state to change back
                        # Indexed [0, ..., self.total_sample_count]
                        idx_to_change = sample_indices[terminal_interpolation_idx]

                        # Get target shape for replacing samples
                        target_shape = interpolated_batch[key][terminal_interpolation_idx].shape

                        # Set interpolated samples to be original samples
                        interpolated_batch[key][terminal_interpolation_idx] = \
                            self.dataset[key][idx_to_change].reshape(target_shape)

            else:  # Slice along the sampled indices
                interpolated_batch[key] = self.dataset[key][sample_indices]

        # Add "batch_indexes" key to interpolated batch
        interpolated_batch["batch_indexes"] = sample_indices

        # Add/update batch_indexes
        if prioritized_replay:  # Only need to return batch_indexes if using PER

            # If using interpolated priority updates
            if interp_prio_update:  # Use sample and neighbor point

                # Add indices for samples and neighbors
                interpolated_batch["neighbor_indexes"] = neighbor_indices
                interpolated_batch["b"] = b

                # Overwrite interpolation coefficient when terminal states exist
                if check_terminal_interpolation:
                    if num_terminal_interpolated_samples > 0:
                        interpolated_batch["b"][terminal_interpolation_idx] = \
                            np.ones(num_terminal_interpolated_samples)

                # Add previous priorities
                interpolated_batch["sample_priorities"] = sample_priorities
                interpolated_batch["neighbor_priorities"] = neighbor_priorities

                # Need to interpolate IS weights (o.w. left as standard)
                if self.use_importance_sampling:
                    w_S = interpolated_batch["weights"]  # Get IS weights of samples
                    w_N = self.replay_buffers[self.policy_id].compute_is_weights(
                        neighbor_indices, beta=self.prioritized_replay_beta)  # IS_N

                    # Construct IS weights of interpolated points as sum of two
                    interpolated_weights = (w_S * b) + (w_N * (1-b))

                    # Normalize weights by dividing by max
                    interpolated_batch["weights"] = interpolated_weights

            # Log IS weights to tensorboard
            if parameters.GLOBAL_STEP_COUNT % parameters.LOG_INTERVAL == 0 \
                and self.use_importance_sampling:
                base_tag = "Importance Sampling Weights/ {} Weights"

                # Get IS weights from batch
                is_weights = interpolated_batch["weights"]

                # Compute min, max, mean, and var of IS weights
                is_min, is_max = np.min(is_weights), np.max(is_weights)
                is_mean, is_var = np.mean(is_weights), np.var(is_weights)

                # Log to tensorboard
                tag_types = ["Min", "Max", "Mean", "Variance"]
                values = [is_min, is_max, is_mean, is_var]
                for t, v in zip(tag_types, values):
                    self.writer.add_scalar(base_tag.format(t), v,
                                           global_step=self.step_count)

        # Add sample weights if using likelihood weighting
        if sample_weights is not None:
            interpolated_batch["sample_weights"] = sample_weights

        # Finish timing batch construction
        time_end_batch_construction = time.time()
        self.batch_construction_time += time_end_batch_construction - \
                                        time_start_batch_construction
        if parameters.GLOBAL_STEP_COUNT % parameters.LOG_INTERVAL == 0:
            self.average_batch_construction_time = \
                self.batch_construction_time / (self.step_count + 1)

        return SampleBatch(interpolated_batch)

    def perform_holdout_evaluation(self):
        """Method to perform holdout evaluation to assess
        the quality of interpolation.

        This function is defined as a placeholder, and should be defined in the
        child classes of this class.
        """
        raise Exception(
            "Base class does not perform holdout analytic_evaluation. Please call"
            "one of the defined interpolated replay buffers.")

    def compute_neighbor_change_mask(self):
        """Method to compute the changed neighbors when a new environment
        interaction is added.

        Specifically, after a new environment interaction is added, this
        method checks the k-nearest neighbors of all points after the nearest
        neighbor objects have been refit, and creates a mask of transition
        indices where the most-recently-added environment interaction is part of
        the set of a given point's neighbors.
        """
        # FAISS
        if self.use_faiss:
            neighbor_idx = self.faiss.query(self.X_norm)

        # KD Tree
        elif self.use_kd_tree:
            neighbor_idx = self.kd_tree.query(self.X_norm, k=self.kneighbors,
                return_distance=False, dualtree=True, sort_results=True)
        # KNN
        else:
            neighbor_idx = self.knn.kneighbors(
                self.X_norm, return_distance=False)

        # Compute a binary mask of where neighbors have changed
        idx_last_added = self.total_sample_count - 1
        self.replace_mask = np.any(neighbor_idx == idx_last_added, axis=1)

        # Activates condition to re-train models for neighbors with new samples
        self.check_update_neighbors = True

    def log_and_update_metrics(self):
        """Method for logging metrics, when appropriate, to tensorboard.

        This method first determines if metrics should be updated. If so,
        metrics are updated and logged to tensorboard."""
        # Print timing
        if self.step_count % parameters.LOG_INTERVAL == 0 and self.verbose:
            print("AVERAGE SAMPLE/QUERY PREPROCESSING TIME: {} \n"
                  "AVERAGE INTERPOLATION TIME: {} \n"
                  "AVERAGE BUFFER PROCESSING TIME: {} \n"
                  "AVERAGE NEIGHBOR QUERY TIME: {} \n"
                  "AVERAGE BATCH CONSTRUCTION TIME: {} \n"
                  "ENV INTERACTION COUNT: {} \n"
                  "STEP COUNT: {}".format(
                round(self.average_preprocessing_time, 8),
                round(self.average_interpolation_time, 8),
                round(self.average_buffer_preprocessing_time, 8),
                round(self.average_query_time, 8),
                round(self.average_batch_construction_time, 8),
                self.batch_add_count, self.step_count))

            # If using PER, log PER sampling and prio update times
            if self.prioritized_replay:
                print("AVERAGE PER SAMPLING TIME: {} \n"
                      "AVERAGE PRIORITY UPDATE TIME: {} \n".format(
                    round(self.average_per_sample_time, 8),
                    round(self.average_prio_update_time, 8)))

            # Log scalars to tensorboard
            names = ["Average Preprocessing Time", "Average Interpolation Time",
                     "Average Buffer Processing Time", "Average Query Time",
                     "Average Batch Construction Time"]
            vals = [self.average_preprocessing_time,
                    self.average_interpolation_time,
                    self.average_buffer_preprocessing_time,
                    self.average_query_time,
                    self.average_batch_construction_time]

            # If using PER, log PER sampling and prio update times to Tb
            if self.prioritized_replay:
                names.extend(["Average PER Sampling Time",
                              "Average Priority Update Time"])
                vals.extend([self.average_per_sample_time,
                             self.average_prio_update_time])

            for name, val in zip(names, vals):
                self.writer.add_scalar("Runtime/{}".format(name), val,
                                       self.step_count)

            # Track GPR runtime of operations for benchmarking
            if self.gaussian_process and self.use_gpytorch:
                avg_create_time = round(
                    self.gpr_tensor_creation_time / (self.step_count + 1), 8)
                avg_normal_time = round(
                    self.gpr_normalization_time / (self.step_count + 1), 8)
                avg_model_time = round(
                    self.gpr_model_formation_time / (self.step_count + 1), 8)
                avg_preprop_inf_time = round(
                    self.gpr_inference_preprocessing_time / (self.step_count + 1), 8)
                avg_inf_time = round(
                    self.gpr_inference_time / (self.step_count + 1), 8)
                avg_ll_weights_time = round(
                    self.gpr_likelihood_weights_time / (self.step_count + 1), 8)
                avg_training_time = round(
                    self.gpr_training_time / (self.step_count + 1), 8)

                # Display GPR runtime metrics
                print("GPR STATS: \n"
                      "AVERAGE TENSOR CREATION_TIME: {} \n"
                      "AVERAGE NORMALIZATION_TIME: {} \n"
                      "AVERAGE MODEL CREATION_TIME: {} \n"
                      "AVERAGE INFERENCE PREPROCESSING TIME: {} \n"
                      "AVERAGE INFERENCE TIME: {} \n"
                      "AVERAGE LIKELIHOOD WEIGHTS TIME: {} \n"
                      "AVERAGE HYPERPARAMETER OPT. TIME: {}".format(
                    avg_create_time, avg_normal_time, avg_model_time,
                    avg_preprop_inf_time, avg_inf_time, avg_ll_weights_time,
                    avg_training_time)
                )

                # Log scalars to tensorboard
                names = ["Average Tensor Creation Time",
                         "Average Normalization Time",
                         "Average Model Creation Time",
                         "Average Inference Preprocessing Time",
                         "Average Inference Time",
                         "Average Likelihood Weights Time",
                         "Average Hyperparameter Opt. Time"]
                vals = [avg_create_time, avg_normal_time, avg_model_time,
                        avg_preprop_inf_time, avg_inf_time, avg_ll_weights_time,
                        avg_training_time]

                # Loop through values and names to log to tb
                for name, val in zip(names, vals):
                    self.writer.add_scalar(
                        "Runtime/{}".format(name), val, self.step_count)

    def compute_batch_metrics(self, interpolated_batch):
        """Computes metrics over interpolated batches to determine metrics for
        the interpolated batch.

        Parameters:
            interpolated_batch (MultiAgentBatch): Batch corresponding to the
                interpolated batch produced by the InterpolatedReplayBuffer
                and one of its child classes.
        """
        # Get transition space from interpolated batch
        obs = interpolated_batch["obs"]
        actions = interpolated_batch["actions"]
        rewards = interpolated_batch["rewards"]
        if rewards.ndim < 2:
            rewards = rewards.reshape((-1, 1))
        new_obs = interpolated_batch["new_obs"]

        # Stack features in transition space
        D = np.hstack((obs, actions, rewards, new_obs))

        # Use list comprehension to compute pairwise distance metrics
        l2_dist = np.unique(
            [np.linalg.norm(p1 - p2) for i, p1 in enumerate(D) for j, p2 in enumerate(D) if i != j])

        # Compute min/max + distance moments
        l2_min = np.min(l2_dist)  # Minimum L2 pairwise distance
        l2_max = np.max(l2_dist)  # Maximum L2 pairwise distance
        l2_mean = np.mean(l2_dist)  # Mean L2 pairwise distance
        l2_variance = np.var(l2_dist)  # Variance L2 pairwise distance

        # Log to tensorboard
        vals = [l2_min, l2_max, l2_mean, l2_variance]
        names = ["Minimum", "Maximum", "Mean", "Variance"]
        for v, n in zip(vals, names):
            tag = "Interpolated Batch Metrics/{} L2 Distance".format(n)
            self.writer.add_scalar(tag, v, global_step=self.step_count)

    def save_replay_buffer(self, out_path):
        """Method to save the contents of the replay buffer to file.
        Can be used to provide a restore from backup by first saving.

        Parameters:
            out_path (str): String corresponding either to relative or absolute
                (recommended) path to save the current replay buffer in the
                current training run.
        """
        # Get policy ID to index into buffers
        policy_id = self.policy_id
        buffer = self.replay_buffers[policy_id]

        # Get attributes from base buffer
        base_storage = buffer._storage
        base_priorities = buffer.priorities
        base_it_sum = buffer._it_sum
        base_it_min = buffer._it_min
        base_max_priority = buffer._max_priority
        base_prio_change_states = buffer._prio_change_stats
        base_num_sampled = buffer._num_timesteps_sampled
        base_num_added = buffer._num_timesteps_added
        base_num_added_wrap = buffer._num_timesteps_added_wrap
        base_est_size = buffer._est_size_bytes
        base_next_idx = buffer._next_idx
        base_hit_count = buffer._hit_count
        base_eviction_started = buffer._eviction_started
        base_eviction_hit_stats = buffer._evicted_hit_stats

        # Compile attributes into dictionary
        buffer_data = {"step_count": self.step_count,
                       "transitions_sampled": self.transitions_sampled,
                       "prob_interpolation": self.prob_interpolation,
                       "base_storage": base_storage,
                       "base_priorities": base_priorities,
                       "base_it_sum": base_it_sum,
                       "base_it_min": base_it_min,
                       "base_max_priority": base_max_priority,
                       "base_prio_change_states": base_prio_change_states,
                       "base_num_sampled": base_num_sampled,
                       "base_num_added": base_num_added,
                       "base_num_added_wrap": base_num_added_wrap,
                       "base_est_size": base_est_size,
                       "base_next_idx": base_next_idx,
                       "base_hit_count": base_hit_count,
                       "base_eviction_started": base_eviction_started,
                       "base_eviction_hit_stats": base_eviction_hit_stats,
                       "num_added": self.num_added,
                       "fake_batch": self._fake_batch,
                       "retrain_counter": self.retrain_counter,
                       "current_replay_count": self.current_replay_count}

        # Write to pickle file
        with open(out_path, "wb") as file_writer:
            pickle.dump(buffer_data, file_writer)
            file_writer.close()

    def load_replay_buffer(self, in_path):
        """Method to load the contents of the replay buffer to file.
        Can be used to provide a restore from backup by first saving.

        Parameters:
            in_path (str): String corresponding either to relative or absolute
                (recommended) path to replay buffer that will be loaded for this
                given training run.
        """
        # Check if in_path exists
        if os.path.exists(in_path):
            print("LOADING SAVED REPLAY BUFFER FROM: {}".format(in_path))
            with open(in_path, "rb") as file_writer:
                buffer_data = pickle.load(file_writer)
                file_writer.close()

            # Get policy ID and local buffer
            print("REPLAY BUFFERS: {}".format(self.replay_buffers))
            policy_id = list(self.replay_buffers.keys())[0]
            buffer = self.replay_buffers[policy_id]

            # Get attributes from base buffer
            buffer._storage = buffer_data["base_storage"]
            buffer.priorities = buffer_data["base_priorities"]
            buffer._it_sum = buffer_data["base_it_sum"]
            buffer._it_min = buffer_data["base_it_min"]
            buffer._max_priority = buffer_data["base_max_priority"]
            buffer._prio_change_stats = buffer_data["base_prio_change_states"]
            buffer._num_timesteps_sampled = buffer_data["base_num_sampled"]
            buffer._num_timesteps_added = buffer_data["base_num_added"]
            buffer._num_timesteps_added_wrap = buffer_data["base_num_added_wrap"]
            buffer._est_size_bytes = buffer_data["base_est_size"]
            buffer._next_idx = buffer_data["base_next_idx"]
            buffer._hit_count = buffer_data["base_hit_count"]
            buffer._eviction_started = buffer_data["base_eviction_started"]
            buffer._evicted_hit_stats = buffer_data["base_eviction_hit_stats"]


            # Get attributes from LocalReplayBuffer
            self.step_count = buffer_data["step_count"]
            self.num_added = buffer_data["num_added"]
            self._fake_batch = buffer_data["fake_batch"]
            self.retrain_counter = buffer_data["retrain_counter"]
            self.current_replay_count = buffer_data["current_replay_count"]
            self.dataset = SampleBatch.concat_samples(buffer._storage)
            self.step_count = buffer_data["step_count"]
            self.transitions_sampled = buffer_data["transitions_sampled"]
            self.prob_interpolation = buffer_data["prob_interpolation"]

            # Initialize arrays - TODO(rms): Generalize to fragments > 1
            self.d_s = self.dataset["obs"].size[-1]
            self.d_a = self.dataset["actions"].size[-1]
            self.d_r = self.dataset["rewards"].size[-1]
            print(
                "OBSERVATION SIZE: {}, ACTION SIZE: {}, REWARD SIZE: {}".format(
                    self.d_s, self.d_a, self.d_r))
            samples_added = len(buffer._storage)
            self.X = np.zeros(
                (self.buffer_size, self.d_s + self.d_a)).astype(np.float32)
            self.Y = np.zeros(
                (self.buffer_size, self.d_r + self.d_s)).astype(np.float32)
            self.X[:samples_added] = np.hstack((
                self.dataset["obs"], self.dataset["actions"]))
            if self.use_delta:
                self.Y[samples_added - 1, :] = np.hstack(
                    (np.squeeze(self.dataset["rewards"]),
                     np.squeeze(self.dataset["new_obs"] - s["obs"])))
            else:
                self.Y[samples_added - 1, :] = np.hstack(
                    (np.squeeze(self.dataset["rewards"]),
                     np.squeeze(self.dataset["new_obs"])))

            # Finally, preprocess the replay buffer
            self._preprocess_replay_buffer()

        # Path does not exist
        else:
            raise Exception("Tried to load replay buffer from path {}, "
                            "but path does not exist".format(in_path))

    def compute_distance_to_manifold(
            self, X_interp, idx_samples, idx_neighbors, B):
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
        (_, A, R_int, S_p_int) = X_interp

        # Euclidean distance
        reward_dist = []
        next_state_dist = []

        if B is None:
            B = np.ones(len(idx_samples))

        if idx_neighbors is None:
            qposes = [self.Qpos[idx_samples[i]] for i in range(len(idx_samples))]
            qvels = [self.Qvel[idx_samples[i]] for i in range(len(idx_samples))]

        else:
            # Now jointly loop over elements of batch and interpolate qpos/qvel
            qposes = [(B[i] * self.Qpos[idx_samples[i]]) +
                      (1 - B[i]) * self.Qpos[idx_neighbors[i]] for i in range(len(B))]
            qvels = [(B[i] * self.Qvel[idx_samples[i]]) +
                      (1 - B[i]) * self.Qvel[idx_neighbors[i]] for i in range(len(B))]

        # Now loop jointly over all quantities
        for qpos, qvel, a, r_int, s_p_int in zip(qposes, qvels, A, R_int, S_p_int):

            # Set the state for the auxiliary environment
            self.mujoco_env.set_state(qpos, qvel)

            # Next, compute the reward
            s_p_true, r_true, _, _ = self.mujoco_env.step(a)

            # Now compute different distance metrics
            next_state_diff = np.linalg.norm(s_p_int - s_p_true) ** 2  # Difference in next state
            reward_diff = np.linalg.norm(r_int - r_true) ** 2  # Difference in reward

            # Append to lists
            next_state_dist.append(next_state_diff)
            reward_dist.append(reward_diff)

        # Lastly, compute the mean
        return np.mean(reward_dist), np.mean(next_state_dist)
