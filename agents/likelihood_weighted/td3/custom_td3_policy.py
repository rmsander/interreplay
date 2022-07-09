"""A more stable successor to TD3.

By default, this uses a near-identical configuration to that reported in the
TD3 paper.
"""
# Import from RLlib
from ray.rllib.agents.ddpg.ddpg import DEFAULT_CONFIG as DDPG_CONFIG

# Import from this repo
from agents.likelihood_weighted.ddpg.custom_ddpg_policy import \
    DDPGTrainerLikelihoodWeighting, DDPGTrainer

# Import gradient clipping parameters
from parameters import GRAD_CLIP, TARGET_UPDATE_FREQ, NO_DONE_AT_END
import os

TD3_DEFAULT_CONFIG = DDPGTrainerLikelihoodWeighting.merge_trainer_configs(
    DDPG_CONFIG,
    {
        # largest changes: twin Q functions, delayed policy updates, and target
        # smoothing
        "twin_q": True,
        "policy_delay": 2,
        "smooth_target_policy": True,
        "target_noise": 0.2,
        "target_noise_clip": 0.5,
        "exploration_config": {
            # TD3 uses simple Gaussian noise on top of deterministic NN-output
            # actions (after a possible pure random phase of n timesteps).
            "type": "GaussianNoise",
            # For how many timesteps should we return completely random
            # actions, before we start adding (scaled) noise?
            "random_timesteps": 10000,
            # Gaussian stddev of action noise for exploration.
            "stddev": 0.1,
            # Scaling settings by which the Gaussian noise is scaled before
            # being added to the actions. NOTE: The scale timesteps start only
            # after(!) any random steps have been finished.
            # By default, do not anneal over time (fixed 1.0).
            "initial_scale": 1.0,
            "final_scale": 1.0,
            "scale_timesteps": 1
        },

        "grad_clip": GRAD_CLIP[os.environ["RL_ENV"]],
        "no_done_at_end": NO_DONE_AT_END[os.environ["RL_ENV"]],

        # other changes & things we want to keep fixed:
        # larger actor learning rate, no l2 regularisation, no Huber loss, etc.
        "learning_starts": 10000,
        "actor_hiddens": [400, 300],
        "critic_hiddens": [400, 300],
        "n_step": 1,
        "gamma": 0.99,
        ############################## BIER ####################################
        # NOTE: Changed from 1e-3 to 5e-4
        "actor_lr": 5e-4,
        "critic_lr": 5e-4,
        ############################## BIER ####################################
        "l2_reg": 0,
        "tau": 5e-3,
        "train_batch_size": 100,
        "use_huber": False,
        "target_network_update_freq": TARGET_UPDATE_FREQ["TD3"],
        "num_workers": 0,
        "num_gpus_per_worker": 0,
        "worker_side_prioritization": False,
        "buffer_size": 1000000,
        "prioritized_replay": False,
        "clip_rewards": False,
        "use_state_preprocessor": False,
    })

############################## BIER ############################################
# Likelihood weighted
TD3TrainerLikelihoodWeighting = DDPGTrainerLikelihoodWeighting.with_updates(
    name="TD3",
    default_config=TD3_DEFAULT_CONFIG,
)

# Non-likelihood weighted
TD3Trainer = DDPGTrainer.with_updates(
    name="TD3",
    default_config=TD3_DEFAULT_CONFIG,
)
############################## BIER ############################################
