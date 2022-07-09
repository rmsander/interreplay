"""Variable definitions or parameter values. These are passed in to files such
as interpolated_replay.py."""
# Native Python packages
import math
import torch

# Custom Python packages
import utils.execution.performance_utils as pu

# Gaussian Process variables
# Number of epochs to train GP
GP_EPOCHS = {
    "HalfCheetah-v2": 10,  # TODO(rms): 10 for local models
    "InvertedPendulum-v2": 10,  # TODO(rms): 10 for local models
    "InvertedDoublePendulum-v2": 10,
    "Ant-v2": 10,
    "Hopper-v2": 10,
    "Pendulum-v0": 10,
    "Walker2d-v2": 10,
    "Humanoid-v2": 10,
    "HumanoidStandup-v2": 10,
    "Reacher-v2": 10,
    "Swimmer-v2": 10
}

# Learning rate for Gaussian processes
GP_LR = {
    "HalfCheetah-v2": 0.5,  # TODO(rms): Check before ablations
    "InvertedPendulum-v2": 0.5,
    "InvertedDoublePendulum-v2": 0.5,
    "Ant-v2": 0.5,
    "Hopper-v2": 0.5,
    "Pendulum-v0": 0.5,
    "Walker2d-v2": 0.5,
    "Humanoid-v2": 0.5,
    "HumanoidStandup-v2": 0.5,
    "Reacher-v2": 0.5,
    "Swimmer-v2": 0.5
}

# Threshold for GP_THR hyperparameter optimization
GP_THR = -math.inf

############### ENVIRONMENT_SPECIFIC PARAMETERS ########################
# Set parameters for learning starts
LEARNING_STARTS = {
    "HalfCheetah-v2": 10000,
    "InvertedPendulum-v2": 1500,
    "InvertedDoublePendulum-v2": 1500,
    "Pendulum-v0": 1500,
    "Reacher-v2": 1500,
    "MountainCarContinuous-v0": 1500,
    "LunarLanderContinuous-v2": 1500,
    "Ant-v2": 10000,
    "Hopper-v2": 10000,
    "Walker2d-v2": 10000,
    "Humanoid-v2": 10000,
    "HumanoidStandup-v2": 10000,
    "Swimmer-v2": 10000,
    "dmc.dm_control_configs.walker_walk": 10000,
    "dmc.dm_control_configs.walker_run": 10000,
    "dmc.dm_control_configs.acrobot_swingup": 10000,
    "dmc.dm_control_configs.hopper_hop": 10000,
    "dmc.dm_control_configs.hopper_stand": 10000,
    "dmc.dm_control_configs.cheetah_run": 10000,
    "dmc.dm_control_configs.pendulum_swingup": 10000,
    "dmc.dm_control_configs.cartpole_swingup": 10000,
    "dmc.dm_control_configs.humanoid_walk": 10000,
    "dmc.dm_control_configs.quadruped_run": 10000,
    "dmc.dm_control_configs.quadruped_walk": 10000,
    "dmc.dm_control_configs.finger_spin": 10000,
    "dmc.dm_control_configs.cup_catch": 10000,
    "dmc.dm_control_configs.reacher_easy": 10000,
    "dmc.dm_control_configs.reacher_hard": 10000
}

# Set total environment interacts by environment
TOTAL_ENV_INTERACTS = {
    "HalfCheetah-v2": int(1e6),
    "InvertedPendulum-v2": int(5e4),
    "InvertedDoublePendulum-v2": int(5e4),
    "MountainCarContinuous-v0": int(5e4),
    "LunarLanderContinuous-v2": int(5e4),
    "Reacher-v2": int(1e6),
    "Hopper-v2": int(1e6),
    "Walker2d-v2": int(1e6),
    "Humanoid-v2": int(1e6),
    "HumanoidStandup-v2": int(1e6),
    "Swimmer-v2": int(1e6),
    "Ant-v2": int(1e6),
    "dmc.dm_control_configs.walker_walk": int(1e6),
    "dmc.dm_control_configs.walker_run": int(1e6),
    "dmc.dm_control_configs.acrobot_swingup": int(1e6),
    "dmc.dm_control_configs.hopper_hop": int(1e6),
    "dmc.dm_control_configs.hopper_stand": int(1e6),
    "dmc.dm_control_configs.cheetah_run": int(1e6),
    "dmc.dm_control_configs.pendulum_swingup": int(1e6),
    "dmc.dm_control_configs.cartpole_swingup": int(1e6),
    "dmc.dm_control_configs.humanoid_walk": int(1e6),
    "dmc.dm_control_configs.quadruped_run": int(1e6),
    "dmc.dm_control_configs.quadruped_walk": int(1e6),
    "dmc.dm_control_configs.finger_spin": int(1e6),
    "dmc.dm_control_configs.cup_catch": int(1e6),
    "dmc.dm_control_configs.reacher_easy": int(1e6),
    "dmc.dm_control_configs.reacher_hard": int(1e6)

}

# Set limit of environment interacts
#TOTAL_ENV_INTERACTS = int(2e5)
#print("Total Environment Interactions for episode: {}".format(
#    TOTAL_ENV_INTERACTS))

# Set horizons - "none" horizons are considered infinite-horizon tasks
HORIZON = {
    "HalfCheetah-v2": 1000,
    "InvertedPendulum-v2": 1000,
    "InvertedDoublePendulum-v2": 1000,
    "MountainCarContinuous-v0": 999,
    "LunarLanderContinuous-v2": None,
    "Ant-v2": None,
    "Hopper-v2": None,
    "Pendulum-v0": 200,
    "Walker2d-v2": None,
    "Humanoid-v2": None,  # Check
    "HumanoidStandup-v2": None,  # Check
    "Reacher-v2": 50,
    "Swimmer-v2": 1000,
    "dmc.dm_control_configs.walker_walk": 1000,
    "dmc.dm_control_configs.walker_run": 1000,
    "dmc.dm_control_configs.acrobot_swingup": 1000,
    "dmc.dm_control_configs.hopper_hop": 1000,
    "dmc.dm_control_configs.hopper_stand": 1000,
    "dmc.dm_control_configs.cheetah_run": 1000,
    "dmc.dm_control_configs.pendulum_swingup": 1000,
    "dmc.dm_control_configs.cartpole_swingup": 1000,
    "dmc.dm_control_configs.humanoid_walk": 1000,
    "dmc.dm_control_configs.quadruped_run": 1000,
    "dmc.dm_control_configs.quadruped_walk": 1000,
    "dmc.dm_control_configs.finger_spin": 1000,
    "dmc.dm_control_configs.cup_catch": 1000,
    "dmc.dm_control_configs.reacher_easy": 1000,
    "dmc.dm_control_configs.reacher_hard": 1000

}

# Create gradient clipping parameters
GRAD_CLIP = {
    "HalfCheetah-v2": None,  # Previously 5
    "InvertedPendulum-v2": None,  # Previously 2
    "InvertedDoublePendulum-v2": None,  # Previously 2
    "MountainCarContinuous-v0": None,
    "LunarLanderContinuous-v2": None,
    "Ant-v2": None,
    "Swimmer-v2": None,  # TODO(rms): Tune
    "Hopper-v2": None,  # Previously 40
    "Pendulum-v0": None,  # TODO(rms): Tune
    "Walker2d-v2": None,  # Previously 10
    "Humanoid-v2": None,  # TODO(rms): Tune
    "HumanoidStandup-v2": None,  # TODO(rms): Tune
    "Reacher-v2": None,  # TODO(rms): Tune
    "dmc.dm_control_configs.walker_walk": None,
    "dmc.dm_control_configs.walker_run": None,
    "dmc.dm_control_configs.acrobot_swingup": None,
    "dmc.dm_control_configs.hopper_hop": None,
    "dmc.dm_control_configs.hopper_stand": None,
    "dmc.dm_control_configs.cheetah_run": None,
    "dmc.dm_control_configs.pendulum_swingup": None,
    "dmc.dm_control_configs.cartpole_swingup": None,
    "dmc.dm_control_configs.humanoid_walk": None,
    "dmc.dm_control_configs.quadruped_run": None,
    "dmc.dm_control_configs.quadruped_walk": None,
    "dmc.dm_control_configs.finger_spin": None,
    "dmc.dm_control_configs.cup_catch": None,
    "dmc.dm_control_configs.reacher_easy": None,
    "dmc.dm_control_configs.reacher_hard": None

}
############### ALGORITHM-SPECIFIC PARAMETERS ########################

# Set parameters for target update frequency
TARGET_UPDATE_FREQ = {
    "SAC": 0,
    "DDPG": 0,
    "TD3": 0
}

# Create dictionary for buffer size
BUFFER_SIZE = {
    "SAC": 1000000,
    "DDPG": 50000,
    "TD3": 1000000
}

# Check for NaNs in gradient updates
CHECK_FOR_NANS = False

# Setting maximum amount of allowed memory (relative to computer - i.e. [0, 1])
BYTES_THRESHOLD = 0.9

# Setting directory for Ray (randomly-generated to avoid conflicting servers)
HASH_LENGTH = 8

# Set True for environments that don't conditionally terminate based on state
# FALSE - Indicates the agent's state can trigger early episode termination
# NOTE: If an agent reaches 1000 time steps in an episode, done is set to False
# regardless, since this captures the infinite horizon aspects of the MDP.
NO_DONE_AT_END = {
    "HalfCheetah-v2": True,
    "Pendulum-v0": True,
    "Reacher-v2": True,
    "Swimmer-v2": True,
    "MountainCarContinuous-v0": True,
    "HumanoidStandup-v2": True,
    "LunarLanderContinuous-v2": False,
    "InvertedPendulum-v2": False,
    "InvertedDoublePendulum-v2": False,
    "Ant-v2": False,
    "Hopper-v2": False,
    "Walker2d-v2": False,
    "Humanoid-v2": False,
    "dmc.dm_control_configs.walker_walk": True,
    "dmc.dm_control_configs.walker_run": True,
    "dmc.dm_control_configs.acrobot_swingup": True,
    "dmc.dm_control_configs.hopper_hop": True,
    "dmc.dm_control_configs.hopper_stand": True,
    "dmc.dm_control_configs.cheetah_run": True,
    "dmc.dm_control_configs.pendulum_swingup": True,
    "dmc.dm_control_configs.cartpole_swingup": True,
    "dmc.dm_control_configs.humanoid_walk": True,
    "dmc.dm_control_configs.quadruped_run": True,
    "dmc.dm_control_configs.quadruped_walk": True,
    "dmc.dm_control_configs.finger_spin": True,
    "dmc.dm_control_configs.cup_catch": True,
    "dmc.dm_control_configs.reacher_easy": True,
    "dmc.dm_control_configs.reacher_hard": True

}

# Add L2 regularization to SAC actor net for stability, if needed

SAC_L2_REG = {
    "HalfCheetah-v2": 1e-3,
    "InvertedPendulum-v2": 1e-3,
    "InvertedDoublePendulum-v2": 1e-3,
    "Ant-v2": 2e-3,
    "Swimmer-v2": 2e-4,
    "Hopper-v2": 2e-2,
    "Pendulum-v0": 1e-3,
    "Walker2d-v2": 1e-2,
    "dmc.dm_control_configs.walker_walk": 1e-2,
    "Humanoid-v2": 1e-3,
    "HumanoidStandup-v2": 1e-3,
    "Reacher-v2": 1e-3,
    "dmc.dm_control_configs.walker_walk": 1e-2,
    "dmc.dm_control_configs.walker_run": 1e-2,
    "dmc.dm_control_configs.acrobot_swingup": 1e-2,
    "dmc.dm_control_configs.hopper_hop": 1e-2,
    "dmc.dm_control_configs.hopper_stand": 1e-2,
    "dmc.dm_control_configs.cheetah_run": 1e-2,
    "dmc.dm_control_configs.pendulum_swingup": 1e-2,
    "dmc.dm_control_configs.cartpole_swingup": 1e-2,
    "dmc.dm_control_configs.humanoid_walk": 1e-2,
    "dmc.dm_control_configs.quadruped_run": 1e-2,
    "dmc.dm_control_configs.quadruped_walk": 1e-2,
    "dmc.dm_control_configs.finger_spin": 1e-2,
    "dmc.dm_control_configs.cup_catch": 1e-2,
    "dmc.dm_control_configs.reacher_easy": 1e-2,
    "dmc.dm_control_configs.reacher_hard": 1e-2

}

# Specify from pixels for DM Control Environments
FROM_PIXELS = False
FRAME_SKIP = 1

# SET NUMBER OF CPUs for execution robustness
NUM_CPU = 4

# GP hyperparameter optimization
GP_PATIENCE = 1
GP_LAG = 1

# Set minimum inferred noise level for likelihood
MIN_INFERRED_NOISE_LEVEL = 1e-3

# Use for keeping track of step count across function calls in different files
GLOBAL_STEP_COUNT = 0  # Incremented in replay buffer classes
LOG_INTERVAL = 1000  # How many training steps before quantities logged
REPLAY_RATIO = 1  # Set later when class initialized
RL_ENV = None   # Set later when class initialized

# Placeholder for TB Writer
TB_WRITER = None

# Get device to use globally
DEVICE = pu.determine_device_torch()
CUDA_AVAILABLE = torch.cuda.is_available()

# ENV
ENV = None
