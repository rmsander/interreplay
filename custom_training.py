"""Script to run custom training using any replay buffer, algorithm, or RL
environment. Designed to be run to perform ablation studies on different replay
buffers.

The main script in this file relies on the tune API to run training. The
reinforcement learning algorithm, environment, and replay buffer are determined
using the execution utilities found in this repository.
"""
# Native Python packages
import os
import time
import psutil

# Ray/rllib
import ray
from ray import tune
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from utils.execution.execution_utils import get_random_string, set_seeds
import utils.execution.custom_training_helper_functions as helper_utils

# Used to check for GPUs and seeding
from tensorflow.python.client import device_lib
import torch

# Argument parser
from utils.execution.argument_parser import parse_arguments

# Parameters
from parameters import BYTES_THRESHOLD, HASH_LENGTH, LEARNING_STARTS, \
    TARGET_UPDATE_FREQ, HORIZON, BUFFER_SIZE, NO_DONE_AT_END, \
    NUM_CPU, CHECK_FOR_NANS, ENV
import parameters

def custom_execution_plan(workers, config):
    """Function called by ray.tune that executes a customized training routine.

    Parameters:
        workers (WorkerSet): A set of workers in charge of executing
            agent training and evaluation in a distributed fashion.

        config (TrainerConfigDict): Dictionary containing parameter
            configurations for running training and hyperparameter tuning.

    Returns:
        output_op (Object): Object for setting up the output operation.
    """
    # Set global configuration parameters
    replay_args = REPLAY_ARGS
    training_args = TRAINING_ARGS
    prio_args, local_replay_buffer = helper_utils.get_prio_and_buffer(
        replay_args, training_args, config)

    # SET VARIABLE FOR ENV
    parameters.ENV = workers.local_worker().env

    # Collect rollouts
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # We execute the following steps concurrently:
    # (1) Generate rollouts and store them in our local replay buffer.
    # Calling next() on store_op drives this.
    store_op = rollouts.for_each(
        StoreToReplayBuffer(local_buffer=local_replay_buffer))

    def update_prio(item):
        """Helper function for updating the priorities of the samples in the
        replay buffer.

        Parameters:
            item (tuple): A tuple containing batch information and metrics for
                use in updating the priorities of the samples used for training.

        Returns:
            info_dict (dict): A dictionary containing metrics and priority
                information for a given training batch sampled from the replay
                buffer.
        """
        # Extract training batch samples and get info for updating priorities
        samples, info_dict = item

        # Check if using PER
        if replay_args["prioritized_replay"]:
            prio_dict = {}

            for policy_id, info in info_dict.items():

                # Update priorities with "Interpolated PER"
                if replay_args["interp_prio_update"] and training_args["custom_replay_buffer"]:
                    td_error = info.get(
                        "td_error", info[LEARNER_STATS_KEY].get("td_error"))

                    # Update the priority dictionary with other pertinent info
                    sample_data = samples.policy_batches[policy_id].data
                    prio_dict[policy_id] = (
                        sample_data.get("batch_indexes"),
                        sample_data.get("neighbor_indexes"),
                        sample_data.get("b"),
                        sample_data.get("sample_priorities"),
                        sample_data.get("neighbor_priorities"),
                        td_error)

                else:  # Only update priorities of sampled point
                    td_error = info.get("td_error",
                                        info[LEARNER_STATS_KEY].get("td_error"))
                    prio_dict[policy_id] = (samples.policy_batches[policy_id]
                                            .data.get("batch_indexes"), td_error)

            # Update priorities after looping over info
            local_replay_buffer.update_priorities(prio_dict)

        return info_dict

    # Select method of updating
    training_op = TrainOneStep(workers, num_sgd_iter=training_args["num_sgd_iter"])

    # (2) Read and train on experiences from the replay buffer. Every batch
    # returned from the LocalReplay() iterator is passed to TrainOneStep to
    # take a SGD step, and then we decide whether to update the target network.
    post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)

    # Replay operation - can also update target network by "by_steps_sampled"
    replay_op = Replay(local_buffer=local_replay_buffer) \
        .for_each(lambda x: post_fn(x, workers, config)) \
        .for_each(training_op) \
        .for_each(update_prio) \
        .for_each(UpdateTargetNetwork(
        workers, config["target_network_update_freq"],
        by_steps_trained=training_args["by_steps_trained"]))
    print("BY STEPS TRAINED: {}".format(training_args["by_steps_trained"]))

    # Alternate deterministically between (1) and (2). Only return the output
    # of (2) since training metrics are not available until (2) runs.
    train_op = Concurrently(
        [store_op, replay_op], mode="round_robin", output_indexes=[1],
        round_robin_weights=[1, training_args["round_robin_weights"]])

    return StandardMetricsReporting(train_op, workers, config)


def main(args):
    """Function for running execution plan with Ray, Tune, and Torch on the
    backend.

    Parameters:
        args: A list of command line arguments.
    """
    # Check if the rollout fragment length needs to be trained
    if args.whole_episodes:
        args.rollout_fragment_length = 1000  # TOOD(rms): Set episode?
        args.round_robin_weights *= 1000

    # Display current configuration
    print("Configuration: \n {}".format(args))

    # Devicestarget network
    local_device_protos = device_lib.list_local_devices()
    gpu_devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
    num_gpus = min(len(gpu_devices), 1)  # Limit each experiment to 1 GPU
    print("NUM GPUs AVAILABLE: {}".format(num_gpus))

    # Set seeds
    set_seeds(args.seed)

    # Flags for Replay Buffer
    global TRAINING_ARGS, REPLAY_ARGS
    TRAINING_ARGS, REPLAY_ARGS = helper_utils.create_config_dicts(args, gpu_devices)

    # Get trainer
    os.environ["RL_ENV"] = args.env  # Set RL env global variable
    trainer = helper_utils.get_trainer(args, custom_execution_plan)
    # If you want to call trainer from path, can do so here
    if os.path.exists(args.restore_path):
        trainer.restore(args.restore_path)

    # Compute maximum allowable memory
    total_bytes = psutil.virtual_memory().total
    bytes_allowed = int(total_bytes * BYTES_THRESHOLD)  # Max 90% capacity
    print("MAXIMUM BYTES ALLOWED: {}".format(bytes_allowed))

    # Determine stopping condition(s) - i.e. total episodes
    stop = {"timesteps_total": parameters.TOTAL_ENV_INTERACTS[args.env]}
    print("Trial termination completion conditions: \n{}".format(stop))

    # Set environment variables and ray server specs.
    os.environ["DISPLAY"] = "localhost:10.0"
    cuda_visible_devices = ""
    for i in range(num_gpus):
        cuda_visible_devices += "{},".format(i)
    cuda_visible_devices = cuda_visible_devices[:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    print("CUDA VISIBLE DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("CUDA AVAILABLE?: {}".format(torch.cuda.is_available()))
    print("CHECKING FOR NAN IN POLICY UPDATE? {}".format(CHECK_FOR_NANS))
    hash = get_random_string(HASH_LENGTH)

    # Before we start, sleep for a given amount of time
    print("SLEEPING FOR {} SECONDS".format(args.sleep_time))
    time.sleep(args.sleep_time)

    # Initialize global variable for replay ratio
    parameters.REPLAY_RATIO = args.round_robin_weights
    print("GLOBAL PARAM REPLAY RATIO: {}".format(parameters.REPLAY_RATIO))

    # Now initialize server
    ray.init(num_gpus=num_gpus, num_cpus=NUM_CPU,
             _temp_dir='/tmp/{}_{}'.format(args.agent_name, hash),
             local_mode=args.local_mode)

    # Specify configuration for training
    config = {
        "env": args.env, "log_level": "INFO", "num_gpus": num_gpus,
        "framework": args.framework, "monitor": args.monitor,
        "seed": args.seed, "buffer_size": BUFFER_SIZE[args.trainer],
        "metrics_smoothing_episodes": args.metrics_smoothing_episodes,
        "train_batch_size": args.train_batch_size,
        "timesteps_per_iteration": args.timesteps_per_iteration,
        "replay_sequence_length": args.replay_sequence_length,
        "num_workers": args.num_workers,
        "rollout_fragment_length": args.rollout_fragment_length,
        "evaluation_interval": args.evaluation_interval,
        "evaluation_num_episodes": args.evaluation_num_episodes,
        "n_step": args.n_step, "horizon": HORIZON[args.env],
        "learning_starts": LEARNING_STARTS[args.env],
        "target_network_update_freq": TARGET_UPDATE_FREQ[args.trainer],
        "no_done_at_end": NO_DONE_AT_END[args.env],
        "prioritized_replay": False
     }

    # Specify the local directory
    local_dir = os.path.join(args.local_dir, args.agent_name)

    # Finally, run training for (agent, environment, replay buffer) combination
    tune.run(trainer, local_dir=local_dir, checkpoint_freq=args.checkpoint_freq,
             config=config, stop=stop)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
