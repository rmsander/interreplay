"""Helper functions for executing and deploying experiments in
custom_training.py. This is largely boilerplate code, but is used for providing
an explicit way to pipe appropriate parameters and replay buffer configurations
for running ablation studies.
"""
# Custom packages/modules - replay buffers
import replay_buffers.interpolated_replay.lier
import replay_buffers.interpolated_replay.lier_vectorized
import replay_buffers.interpolated_replay.bier
import replay_buffers.interpolated_replay.bier_precomputed
import replay_buffers.interpolated_replay.bier_local_precomputed
import replay_buffers.interpolated_replay.bier_sklearn
import replay_buffers.baselines.knn_ier
import replay_buffers.baselines.continuous_transition
import replay_buffers.baselines.s4rl_mixup
import replay_buffers.baselines.naive_mixup
import replay_buffers.baselines.vanilla_replay_buffer
import replay_buffers.base_replay_buffer


def get_trainer(args, custom_execution_plan):
    """Utility function to create RLlib trainer. Used in custom_training script.

    Parameters:
        args (argparse.ArgumentParser): Parsed argumentParser object with access
            to command-line parameters for running custom_training.
        custom_execution_plan (function): Function for executing the trainer.
            Created and specified in custom_training.

    Returns:
        trainer (GenericOffPolicyTrainer): A trainer object augmented with a
            configuration dictionary. If using a likelihood-weighted trainer,
            these corresponding likelihood-weighted objects can be found in
            the directory agents.
    """
    from agents.likelihood_weighted.sac.custom_sac_policy import \
        SACTrainerLikelihoodWeighting, SACTrainer
    from agents.likelihood_weighted.ddpg.custom_ddpg_policy import \
        DDPGTrainerLikelihoodWeighting, DDPGTrainer
    from agents.likelihood_weighted.td3.custom_td3_policy import \
        TD3TrainerLikelihoodWeighting, TD3Trainer

    # Create SAC trainer
    if args.trainer == "SAC":  # Likelihood weighting with SAC
        if args.weighted_updates and args.custom_replay_buffer and \
                args.gaussian_process and args.gpytorch:
            print("CUSTOM TRAINER: LIKELIHOOD-WEIGHTED SAC")
            trainer = SACTrainerLikelihoodWeighting.with_updates(
                name=args.agent_name, execution_plan=custom_execution_plan)

        else:  # SAC without likelihood weighting
            print("CUSTOM TRAINER: STANDARD WEIGHTED SAC")
            trainer = SACTrainer.with_updates(
                name=args.agent_name, execution_plan=custom_execution_plan)

    # Create DDPG trainer
    elif args.trainer == "DDPG":  # Likelihood weighting with DDPG
        if args.weighted_updates and args.custom_replay_buffer and \
                args.gaussian_process and args.gpytorch:
            print("CUSTOM TRAINER: LIKELIHOOD-WEIGHTED DDPG")
            trainer = DDPGTrainerLikelihoodWeighting.with_updates(
                name=args.agent_name, execution_plan=custom_execution_plan)

        else:  # DDPG without likelihood weighting
            print("CUSTOM TRAINER: STANDARD WEIGHTED DDPG")
            trainer = DDPGTrainer.with_updates(
                name=args.agent_name, execution_plan=custom_execution_plan)

    # Create TD3 trainer
    elif args.trainer == "TD3":  # Likelihood weighting with TD3
        if args.weighted_updates and args.custom_replay_buffer and \
                args.gaussian_process and args.gpytorch:
            print("CUSTOM TRAINER: LIKELIHOOD-WEIGHTED TD3")
            trainer = TD3TrainerLikelihoodWeighting.with_updates(
                name=args.agent_name, execution_plan=custom_execution_plan)

        else:  # TD3 Without likelihood weighting
            print("CUSTOM TRAINER: STANDARD WEIGHTED TD3")
            trainer = TD3Trainer.with_updates(
                name=args.agent_name, execution_plan=custom_execution_plan)

    return trainer


def create_config_dicts(args, gpu_devices):
    """Utility function to create configuration dictionaries for replay
    and training.

    Parameters:
        args (argparse.ArgumentParser): Parsed argumentParser object with access
            to command-line parameters for running custom_training.
        gpu_devices (list): A list of CUDA devices for hardware-accelerated
            computing.

    Returns:
        TRAINING_ARGS (dict): Dictionary containing configuration parameters
            for training in RLlib.
        REPLAY_ARGS (dict): Dictionary containing configuration parameters
            for sampling from the replay buffer in RLlib.
    """
    TRAINING_ARGS = {"custom_replay_buffer": args.custom_replay_buffer,
                     "round_robin_weights": args.round_robin_weights,
                     "num_sgd_iter": args.num_sgd_iter,
                     "use_queue": args.use_queue,
                     "by_steps_trained": args.target_by_steps_sampled,
                     "ct_baseline": args.ct_baseline,
                     "s4rl_baseline": args.s4rl_baseline,
                     "naive_mixup_baseline": args.naive_mixup_baseline,
                     "noisy_replay_baseline": args.noisy_replay_baseline,
                     "update_hyperparams": args.update_hyperparams
                     }
    REPLAY_ARGS = {"prob_interpolation": args.prob_interpolation,
                   "only_noise": args.only_noise,
                   "uniform_replay": args.uniform_replay,
                   "prioritized_replay": args.prioritized_replay,
                   "gaussian_process": args.gaussian_process,
                   "kneighbors": args.kneighbors,
                   "gpu_devices": gpu_devices,
                   "noise_multiple": args.noise_multiple,
                   "sample_gp_cov": args.sample_gp_cov,
                   "weighted_updates": args.weighted_updates,
                   "gp_pca": args.gp_pca,
                   "gp_pca_components": args.gp_pca_components,
                   "framework": args.framework,
                   "seed": args.seed,
                   "gpytorch": args.gpytorch,
                   "smote_interpolation": args.smote_interpolation,
                   "mixup_interpolation": args.mixup_interpolation,
                   "mixup_alpha": args.mixup_alpha,
                   "furthest_neighbor": args.furthest_neighbor,
                   "multi_sample": args.multi_sample,
                   "log_dir": args.log_dir,
                   "interp_prio_update": args.interp_prio_update,
                   "use_importance_sampling": args.use_importance_sampling,
                   "single_model": args.single_model,
                   "retrain_interval": args.retrain_interval,
                   "train_size": args.train_size,
                   "replay_ratio": args.round_robin_weights,
                   "use_ard": args.use_ard,
                   "composite_kernel": args.composite_kernel,
                   "global_hyperparams": args.global_hyperparams,
                   "warmstart_global": args.warmstart_global,
                   "use_botorch": args.use_botorch,
                   "normalize": args.normalize,
                   "standardize_gpr_x": args.standardize_gpr_x,
                   "use_delta": args.use_delta,
                   "use_priors": args.use_priors,
                   "knn_baseline": args.knn_baseline,
                   "mean_type": args.mean_type,
                   "kernel": args.kernel,
                   "matern_nu": args.matern_nu,
                   "cpu_only": args.cpu_only,
                   "env": args.env,
                   "est_lengthscales": args.est_lengthscales,
                   "est_lengthscale_constant": args.est_lengthscale_constant,
                   "interp_decay_rate": args.interp_decay_rate,
                   "debug_plot": args.debug_plot,
                   "checkpoint_freq": args.checkpoint_freq,
                   "timesteps_per_iteration": args.timesteps_per_iteration,
                   "mc_hyper": args.mc_hyper,
                   "use_queue": args.use_queue,
                   "fp32": args.fp32,
                   "verbose": args.verbose,
                   "on_manifold_testing": args.on_manifold_testing
                   }

    return TRAINING_ARGS, REPLAY_ARGS


def get_prio_and_buffer(replay_args, training_args, config):
    """Utility function to create prioritized experience replay
    arguments and replay buffer.

    Parameters:
        replay_args (dict): Dictionary containing configuration parameters
            for sampling from the replay buffer in RLlib.
        training_args (dict): Dictionary containing configuration parameters
            for training in RLlib.
        config (dict): Dictionary containing configuration parameters for RLlib.

    Returns:
        prio_args (dict): Dictionary containing arguments for prioritized
            experience replay coefficients.
        local_replay_buffer (ReplayBuffer): Replay buffer object used for
            sampling.
    """
    # PER
    if replay_args["prioritized_replay"]:
        print("USING PRIORITIZED EXPERIENCE REPLAY")
        prio_args = {
            "prioritized_replay_alpha": config["prioritized_replay_alpha"],
            "prioritized_replay_beta": config["prioritized_replay_beta"],
            "prioritized_replay_eps": config["prioritized_replay_eps"],
        }

    # Vanilla Replay
    else:
        print("NOT USING PRIORITIZED EXPERIENCE REPLAY")
        # Set as a placeholder - not used
        prio_args = {"prioritized_replay_alpha": 1e-12,
            "prioritized_replay_beta": 0.0,
            "prioritized_replay_eps":0.0}

    # Determine if we use standard or custom replay buffer
    if training_args["custom_replay_buffer"]:

        # Select replay buffer
        if replay_args["gaussian_process"]:

            if replay_args["gpytorch"]:  # Use GPR with gpytorch

                if training_args["use_queue"]:
                    print("CREATING QUEUE-BASED CUSTOM "
                          "REPLAY BUFFER WITH GPYTORCH GPR")

                    if training_args["update_hyperparams"]:
                        print("APPLYING LOCAL, UPDATED HYPERPARAMS")
                        buffer = replay_buffers.interpolated_replay.bier_local_precomputed.PreCompDynamicBIER

                    else:
                        buffer = replay_buffers.interpolated_replay.bier_precomputed.PreCompBIER

                else:
                    print("CREATING CUSTOM REPLAY BUFFER WITH GPYTORCH GPR")
                    buffer = replay_buffers.interpolated_replay.bier.BIER

            else:  # Use GPR with sklearn
                print("CREATING CUSTOM REPLAY BUFFER WITH SKLEARN GPR")
                buffer = replay_buffers.interpolated_replay.bier_sklearn.BIERSk

        elif replay_args["knn_baseline"]:  # Use Nearest Neighbor
            print("CREATING CUSTOM REPLAY BUFFER WITH KNN Baseline")
            buffer = replay_buffers.baselines.knn_ier.KnnIER

        elif training_args["ct_baseline"]:  # Use Continuous Transition Baseline
            buffer = replay_buffers.baselines.continuous_transition.ContinuousTransitionBuffer

        elif training_args["s4rl_baseline"]:  # Use S4RL Mixup Baseline
            buffer = replay_buffers.baselines.s4rl_mixup.S4RLMixupBuffer

        elif training_args["naive_mixup_baseline"]:  # Use Naive mixup
            buffer = replay_buffers.baselines.naive_mixup.NaiveMixupReplay

        elif training_args["noisy_replay_baseline"]:  # Use noisy ER baseline
            buffer = replay_buffers.baselines.vanilla_replay_buffer.NoisyReplay

        else:  # Linear replay buffer
            print("CREATING CUSTOM REPLAY BUFFER WITH VECTORIZED LINEAR INTERP")
            buffer = replay_buffers.interpolated_replay.lier_vectorized.VectorizedLIER

        # Create kwargs to pass to base IER class
        kwargs = {**replay_args, **prio_args}
        config_replay_args = {
            "learning_starts": config["learning_starts"],
            "buffer_size": config["buffer_size"],
            "replay_batch_size": config["train_batch_size"],
            "replay_mode": config["multiagent"]["replay_mode"],
            "replay_sequence_length": config["replay_sequence_length"]}
        kwargs.update(config_replay_args)

        # Create IER replay buffer with kwargs
        local_replay_buffer = buffer(num_shards=1, **kwargs)

    # Use Vanilla Replay Buffer, with or without PER
    else:
        print("USING VANILLA REPLAY BUFFER...")
        if replay_args["prioritized_replay"]:  # Use PER
            buffer = replay_buffers.base_replay_buffer.LocalReplayBuffer
        else:
            buffer = replay_buffers.baselines.vanilla_replay_buffer.UniformReplay

        # Create non-interpolated replay buffer
        local_replay_buffer = buffer(
            num_shards=1, learning_starts=config["learning_starts"],
            buffer_size=config["buffer_size"],
            replay_batch_size=config["train_batch_size"],
            replay_mode=config["multiagent"]["replay_mode"],
            replay_sequence_length=config["replay_sequence_length"],
            replay_ratio=replay_args["replay_ratio"],
            env=replay_args["env"],
            **prio_args)

    print("Learning Starts: {} \n"
          "Buffer Size: {} \n"
          "Replay Batch Size: {} \n"
          "Replay mode: {} \n"
          "Replay Sequence Length: {} \n"
          "Fake batch: {} \n"
          "Target Network Update Frequency: {} \n"
          "N-Step: {} \n"
          "Rollout Fragment Length: {} \n"
          "No Done At End: {} \n"
          "Num Envs Per Worker: {} \n"
          "Evaluation Interval: {} \n"
          "Evaluation Episodes: {} \n"
          "Horizon: {} \n"
          "Clip Actions: {} \n"
          "Normalize Actions: {}".format(
        config["learning_starts"], config["buffer_size"],
        config["train_batch_size"], config["multiagent"]["replay_mode"],
        config["replay_sequence_length"], local_replay_buffer._fake_batch,
        config["target_network_update_freq"], config["n_step"],
        config["rollout_fragment_length"], config["no_done_at_end"],
        config["num_envs_per_worker"], config["evaluation_interval"],
        config["evaluation_num_episodes"], config["horizon"],
        config["clip_actions"], config["normalize_actions"]))

    return prio_args, local_replay_buffer
