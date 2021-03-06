import logging
import numpy as np

import ray
import utils.execution.grad_clipping
from ray.rllib.agents.ddpg.ddpg_tf_policy import build_ddpg_models, \
    get_distribution_inputs_and_class, validate_spaces
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio, \
    PRIO_WEIGHTS
from ray.rllib.models.torch.torch_action_dist import TorchDeterministic
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import huber_loss, l2_loss
import parameters

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


def build_ddpg_models_and_action_dist(policy, obs_space, action_space, config):
    model = build_ddpg_models(policy, obs_space, action_space, config)
    # TODO(sven): Unify this once we generically support creating more than
    #  one Model per policy. Note: Device placement is done automatically
    #  already for `policy.model` (but not for the target model).
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    policy.target_model = policy.target_model.to(device)
    return model, TorchDeterministic


def ddpg_actor_critic_loss_likelihood_weighted(policy, model, _, train_batch):
    twin_q = policy.config["twin_q"]
    gamma = policy.config["gamma"]
    n_step = policy.config["n_step"]
    use_huber = policy.config["use_huber"]
    huber_threshold = policy.config["huber_threshold"]
    l2_reg = policy.config["l2_reg"]

    input_dict = {
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": True,
    }
    input_dict_next = {
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": True,
    }

    model_out_t, _ = model(input_dict, [], None)
    model_out_tp1, _ = model(input_dict_next, [], None)
    target_model_out_tp1, _ = policy.target_model(input_dict_next, [], None)

    ############################## BIER ########################################
    # Check current and next observations for NaNs
    if parameters.CHECK_FOR_NANS:

        # NaNs current obs
        nans_current_obs = torch.any(
            torch.isnan(train_batch[SampleBatch.CUR_OBS]), axis=-1)
        # NaNs actions
        nans_actions = torch.any(
            torch.isnan(train_batch[SampleBatch.ACTIONS]), axis=-1)
        # NaNs rewards
        nans_rewards = torch.isnan(train_batch[SampleBatch.REWARDS])  # Already 1D
        # NaNs next obs
        nans_next_obs = torch.any(
            torch.isnan(train_batch[SampleBatch.NEXT_OBS]), axis=-1)

        # Group together NaNs, labels, and keys from transitions
        nan_idxes = [nans_current_obs, nans_actions, nans_rewards, nans_next_obs]
        nan_labels = ["CURRENT OBS", "ACTIONS", "REWARDS", "NEXT OBS"]
        keys = [SampleBatch.CUR_OBS, SampleBatch.ACTIONS,
                SampleBatch.REWARDS, SampleBatch.NEXT_OBS]

        # Jointly loop over NaNs in transitions
        for nan_idx, nan_label, key in zip(nan_idxes, nan_labels, keys):

            # Now check for NaNs in observations
            if torch.any(nan_idx):
                print("NaNs detected in {}: \n{}".format(
                    nan_label, train_batch[key][nan_idx]))
                print("Number of {} NaNs: {}".format(nan_label, torch.sum(nan_idx)))
                print("Setting {} NaNs to 0".format(nan_label))

                # Set values of NaNs to 0
                train_batch[key][nan_idx] = 0

                # Display cur obs after fixing NaNs
                print("{} after removing NaNs: \n{}".format(
                    nan_label, train_batch[key][nan_idx]))

                # Write to tensorboard, if writer initialized
                if parameters.TB_WRITER is not None:
                    label = "NaN Warning/{}".format(nan_label)
                    parameters.TB_WRITER.add_scalar(
                        label, torch.sum(nan_idx),
                        global_step=parameters.GLOBAL_STEP_COUNT)

        # Check policy and target network outputs for NaNs
        # NaNs t
        nans_model_out_t = torch.any(
            torch.isnan(model_out_t), axis=-1)
        # NaNs t+1
        nans_model_out_tp1 = torch.any(
            torch.isnan(model_out_tp1), axis=-1)
        # NaNs target t+1
        nans_target_model_out_tp1 = torch.any(
            torch.isnan(target_model_out_tp1), axis=-1)

        # Group together NaNs from model outputs
        nan_idxes = [nans_model_out_t, nans_model_out_tp1,
                     nans_target_model_out_tp1]
        nan_labels = ["MODEL OUT t", "MODEL OUT tp1", "TARGET MODEL OUT tp1"]
        vals = [model_out_t, model_out_tp1, target_model_out_tp1]

        # Jointly loop over NaNs in transitions
        for nan_idx, nan_label, val in zip(nan_idxes, nan_labels, vals):

            # Now check for NaNs in observations
            if torch.any(nan_idx):
                print("NaNs detected in {}: \n{}".format(
                    nan_label, val[nan_idx]))
                print("Number of {} NaNs: {}".format(nan_label, torch.sum(nan_idx)))
                print("Setting {} NaNs to 0".format(nan_label))

                # Set values of NaNs to 0
                val[nan_idx] = 0

                # Display cur obs after fixing NaNs
                print("{} after removing NaNs: \n{}".format(
                    nan_label, val[nan_idx]))

                # Write to tensorboard, if writer initialized
                if parameters.TB_WRITER is not None:
                    label = "NaN Warning/{}".format(nan_label)
                    parameters.TB_WRITER.add_scalar(
                        label, torch.sum(nan_idx),
                        global_step=parameters.GLOBAL_STEP_COUNT)

        # For any NaNs, set the likelihood weights to zero in the batch dimension
        all_nans = torch.stack(
            (nans_current_obs, nans_actions, nans_rewards, nans_next_obs,
             nans_model_out_t, nans_model_out_tp1, nans_target_model_out_tp1),
            axis=-1)

        # Create binary mask for NaNs for zeroing out NaN samples
        nans_mask = (~torch.any(all_nans, axis=-1)).type(torch.int32)

    # If not checking for NaNs, just create single array for nans mask
    else:
        nans_mask = torch.ones(train_batch[PRIO_WEIGHTS].shape,
                               device=parameters.DEVICE)

    # SET LIKELIHOOD WEIGHTS
    if "sample_weights" in train_batch:
        likelihood_weights = train_batch["sample_weights"]

    else:
        print("WARNING: USING UNIT LIKELIHOOD WEIGHTS")
        likelihood_weights = torch.ones(train_batch[PRIO_WEIGHTS].shape,
                                        device=parameters.DEVICE)
        if parameters.CUDA_AVAILABLE:
            likelihood_weights = likelihood_weights.cuda()

    # Multiply likelihood weights by mask
    likelihood_weights = likelihood_weights * nans_mask
    ############################## BIER ########################################

    # Policy network evaluation.
    # prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
    policy_t = model.get_policy_output(model_out_t)
    # policy_batchnorm_update_ops = list(
    #    set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

    policy_tp1 = \
        policy.target_model.get_policy_output(target_model_out_tp1)

    # Action outputs.
    if policy.config["smooth_target_policy"]:
        target_noise_clip = policy.config["target_noise_clip"]
        clipped_normal_sample = torch.clamp(
            torch.normal(
                mean=torch.zeros(policy_tp1.size()),
                std=policy.config["target_noise"]).to(policy_tp1.device),
            -target_noise_clip, target_noise_clip)

        policy_tp1_smoothed = torch.min(
            torch.max(
                policy_tp1 + clipped_normal_sample,
                torch.tensor(
                    policy.action_space.low,
                    dtype=torch.float32,
                    device=policy_tp1.device)),
            torch.tensor(
                policy.action_space.high,
                dtype=torch.float32,
                device=policy_tp1.device))
    else:
        # No smoothing, just use deterministic actions.
        policy_tp1_smoothed = policy_tp1

    # Q-net(s) evaluation.
    # prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
    # Q-values for given actions & observations in given current
    q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])

    # Q-values for current policy (no noise) in given current state
    q_t_det_policy = model.get_q_values(model_out_t, policy_t)

    ############################## BIER ########################################
    # ACTOR LOSS
    ############################################################################
    actor_loss = -torch.mean(likelihood_weights * q_t_det_policy)
    ############################## BIER ########################################

    if twin_q:
        twin_q_t = model.get_twin_q_values(model_out_t,
                                           train_batch[SampleBatch.ACTIONS])
    # q_batchnorm_update_ops = list(
    #     set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

    # Target q-net(s) evaluation.
    q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
                                             policy_tp1_smoothed)

    if twin_q:
        twin_q_tp1 = policy.target_model.get_twin_q_values(
            target_model_out_tp1, policy_tp1_smoothed)

    q_t_selected = torch.squeeze(q_t, axis=len(q_t.shape) - 1)
    if twin_q:
        twin_q_t_selected = torch.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
        q_tp1 = torch.min(q_tp1, twin_q_tp1)

    q_tp1_best = torch.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
    q_tp1_best_masked = \
        (1.0 - train_batch[SampleBatch.DONES].float()) * \
        q_tp1_best

    # Compute RHS of bellman equation.
    q_t_selected_target = (train_batch[SampleBatch.REWARDS] +
                           gamma**n_step * q_tp1_best_masked).detach()

    # Compute the error (potentially clipped).
    if twin_q:
        td_error = q_t_selected - q_t_selected_target
        twin_td_error = twin_q_t_selected - q_t_selected_target
        if use_huber:
            errors = huber_loss(td_error, huber_threshold) \
                + huber_loss(twin_td_error, huber_threshold)
        else:
            errors = 0.5 * \
                (torch.pow(td_error, 2.0) + torch.pow(twin_td_error, 2.0))
    else:
        td_error = q_t_selected - q_t_selected_target
        if use_huber:
            errors = huber_loss(td_error, huber_threshold)
        else:
            errors = 0.5 * torch.pow(td_error, 2.0)

    ############################## BIER ########################################
    # CRITIC LOSS
    critic_loss = torch.mean(
        train_batch[PRIO_WEIGHTS] * likelihood_weights * errors)
    ############################## BIER ########################################

    # Add l2-regularization if required.
    if l2_reg is not None:
        for name, var in policy.model.policy_variables(as_dict=True).items():
            if "bias" not in name:
                actor_loss += (l2_reg * l2_loss(var))
        for name, var in policy.model.q_variables(as_dict=True).items():
            if "bias" not in name:
                critic_loss += (l2_reg * l2_loss(var))

    # Model self-supervised losses.
    if policy.config["use_state_preprocessor"]:
        # Expand input_dict in case custom_loss' need them.
        input_dict[SampleBatch.ACTIONS] = train_batch[SampleBatch.ACTIONS]
        input_dict[SampleBatch.REWARDS] = train_batch[SampleBatch.REWARDS]
        input_dict[SampleBatch.DONES] = train_batch[SampleBatch.DONES]
        input_dict[SampleBatch.NEXT_OBS] = train_batch[SampleBatch.NEXT_OBS]
        [actor_loss, critic_loss] = model.custom_loss(
            [actor_loss, critic_loss], input_dict)

    # Store values for stats function.
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.td_error = td_error
    policy.q_t = q_t

    ############################## BIER ########################################
    # Log every LOG_INTERVAL steps or if there is a NaN
    if parameters.GLOBAL_STEP_COUNT % (parameters.LOG_INTERVAL * parameters.REPLAY_RATIO) == 0:

        # Get quantities and names to be logged
        vals = [train_batch[PRIO_WEIGHTS], likelihood_weights, errors,
                actor_loss, td_error, q_t]
        names = ["Priority Weights", "Likelihood Weights", "Errors",
                 "Actor Loss", "TD Error", "Q-vals"]

        # Add critic loss
        if isinstance(critic_loss, list):
            vals.extend(critic_loss)
            names.extend(
                ["Critic Loss {}".format(i) for i in range(len(critic_loss))])
        else:
            vals.append(critic_loss)
            names.append("Critic Loss")

        # Loop over values and metrics
        for val, name in zip(vals, names):

            # Compute moments and log to Tensorboard
            if torch.numel(val) > 1:
                mean_val = torch.mean(val)
                var_val = torch.var(val)

                # Compute min and max
                max_val = torch.max(val)
                min_val = torch.min(val)

                values = [mean_val, var_val, max_val, min_val]
                types = ["Mean", "Variance", "Maximum", "Minimum"]

            else:
                # Variables and types
                values = [val.flatten()]
                types = [""]

            # Log to tensorboard
            for v, t in zip(values, types):
                if parameters.TB_WRITER is not None:  # Initialized as None
                    parameters.TB_WRITER.add_scalar(
                        "DDPG Metrics/{}/{}".format(name, t), v,
                        global_step=parameters.GLOBAL_STEP_COUNT)
    ############################## BIER ########################################

    # Return two loss terms (corresponding to the two optimizers, we create).
    return policy.actor_loss, policy.critic_loss


def ddpg_actor_critic_loss(policy, model, _, train_batch):
    twin_q = policy.config["twin_q"]
    gamma = policy.config["gamma"]
    n_step = policy.config["n_step"]
    use_huber = policy.config["use_huber"]
    huber_threshold = policy.config["huber_threshold"]
    l2_reg = policy.config["l2_reg"]

    input_dict = {
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": True,
    }
    input_dict_next = {
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": True,
    }

    model_out_t, _ = model(input_dict, [], None)
    model_out_tp1, _ = model(input_dict_next, [], None)
    target_model_out_tp1, _ = policy.target_model(input_dict_next, [], None)

    ############################## BIER ########################################
    # Check current and next observations for NaNs
    if parameters.CHECK_FOR_NANS:

        # NaNs current obs
        nans_current_obs = torch.any(
            torch.isnan(train_batch[SampleBatch.CUR_OBS]), axis=-1)
        # NaNs actions
        nans_actions = torch.any(
            torch.isnan(train_batch[SampleBatch.ACTIONS]), axis=-1)
        # NaNs rewards
        nans_rewards = torch.isnan(train_batch[SampleBatch.REWARDS])  # Already 1D
        # NaNs next obs
        nans_next_obs = torch.any(
            torch.isnan(train_batch[SampleBatch.NEXT_OBS]), axis=-1)

        # Group together NaNs, labels, and keys from transitions
        nan_idxes = [nans_current_obs, nans_actions, nans_rewards, nans_next_obs]
        nan_labels = ["CURRENT OBS", "ACTIONS", "REWARDS", "NEXT OBS"]
        keys = [SampleBatch.CUR_OBS, SampleBatch.ACTIONS,
                SampleBatch.REWARDS, SampleBatch.NEXT_OBS]

        # Jointly loop over NaNs in transitions
        for nan_idx, nan_label, key in zip(nan_idxes, nan_labels, keys):

            # Now check for NaNs in observations
            if torch.any(nan_idx):
                print("NaNs detected in {}: \n{}".format(
                    nan_label, train_batch[key][nan_idx]))
                print("Number of {} NaNs: {}".format(nan_label, torch.sum(nan_idx)))
                print("Setting {} NaNs to 0".format(nan_label))

                # Set values of NaNs to 0
                train_batch[key][nan_idx] = 0

                # Display cur obs after fixing NaNs
                print("{} after removing NaNs: \n{}".format(
                    nan_label, train_batch[key][nan_idx]))

                # Write to tensorboard, if writer initialized
                if parameters.TB_WRITER is not None:
                    label = "NaN Warning/{}".format(nan_label)
                    parameters.TB_WRITER.add_scalar(
                        label, torch.sum(nan_idx),
                        global_step=parameters.GLOBAL_STEP_COUNT)

        # Check policy and target network outputs for NaNs
        # NaNs t
        nans_model_out_t = torch.any(
            torch.isnan(model_out_t), axis=-1)
        # NaNs t+1
        nans_model_out_tp1 = torch.any(
            torch.isnan(model_out_tp1), axis=-1)
        # NaNs target t+1
        nans_target_model_out_tp1 = torch.any(
            torch.isnan(target_model_out_tp1), axis=-1)

        # Group together NaNs from model outputs
        nan_idxes = [nans_model_out_t, nans_model_out_tp1,
                     nans_target_model_out_tp1]
        nan_labels = ["MODEL OUT t", "MODEL OUT tp1", "TARGET MODEL OUT tp1"]
        vals = [model_out_t, model_out_tp1, target_model_out_tp1]

        # Jointly loop over NaNs in transitions
        for nan_idx, nan_label, val in zip(nan_idxes, nan_labels, vals):

            # Now check for NaNs in observations
            if torch.any(nan_idx):
                print("NaNs detected in {}: \n{}".format(
                    nan_label, val[nan_idx]))
                print("Number of {} NaNs: {}".format(nan_label, torch.sum(nan_idx)))
                print("Setting {} NaNs to 0".format(nan_label))

                # Set values of NaNs to 0
                val[nan_idx] = 0

                # Display cur obs after fixing NaNs
                print("{} after removing NaNs: \n{}".format(
                    nan_label, val[nan_idx]))

                # Write to tensorboard, if writer initialized
                if parameters.TB_WRITER is not None:
                    label = "NaN Warning/{}".format(nan_label)
                    parameters.TB_WRITER.add_scalar(
                        label, torch.sum(nan_idx),
                        global_step=parameters.GLOBAL_STEP_COUNT)

        # For any NaNs, set the likelihood weights to zero in the batch dimension
        all_nans = torch.stack(
            (nans_current_obs, nans_actions, nans_rewards, nans_next_obs,
             nans_model_out_t, nans_model_out_tp1, nans_target_model_out_tp1),
            axis=-1)

        # Create binary mask for NaNs for zeroing out NaN samples
        nans_mask = (~torch.any(all_nans, axis=-1)).type(torch.int32)

    # If not checking for NaNs, just create single array for nans mask
    else:
        nans_mask = torch.ones(train_batch[PRIO_WEIGHTS].shape,
                               device=parameters.DEVICE)
    ############################## BIER ########################################

    # Policy network evaluation.
    # prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
    policy_t = model.get_policy_output(model_out_t)
    # policy_batchnorm_update_ops = list(
    #    set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

    policy_tp1 = \
        policy.target_model.get_policy_output(target_model_out_tp1)

    # Action outputs.
    if policy.config["smooth_target_policy"]:
        target_noise_clip = policy.config["target_noise_clip"]
        clipped_normal_sample = torch.clamp(
            torch.normal(
                mean=torch.zeros(policy_tp1.size()),
                std=policy.config["target_noise"]).to(policy_tp1.device),
            -target_noise_clip, target_noise_clip)

        policy_tp1_smoothed = torch.min(
            torch.max(
                policy_tp1 + clipped_normal_sample,
                torch.tensor(
                    policy.action_space.low,
                    dtype=torch.float32,
                    device=policy_tp1.device)),
            torch.tensor(
                policy.action_space.high,
                dtype=torch.float32,
                device=policy_tp1.device))
    else:
        # No smoothing, just use deterministic actions.
        policy_tp1_smoothed = policy_tp1

    # Q-net(s) evaluation.
    # prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
    # Q-values for given actions & observations in given current
    q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])

    # Q-values for current policy (no noise) in given current state
    q_t_det_policy = model.get_q_values(model_out_t, policy_t)

    ############################## BIER ########################################
    actor_loss = -torch.mean(q_t_det_policy)
    ############################## BIER ########################################

    if twin_q:
        twin_q_t = model.get_twin_q_values(model_out_t,
                                           train_batch[SampleBatch.ACTIONS])
    # q_batchnorm_update_ops = list(
    #     set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

    # Target q-net(s) evaluation.
    q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
                                             policy_tp1_smoothed)

    if twin_q:
        twin_q_tp1 = policy.target_model.get_twin_q_values(
            target_model_out_tp1, policy_tp1_smoothed)

    q_t_selected = torch.squeeze(q_t, axis=len(q_t.shape) - 1)
    if twin_q:
        twin_q_t_selected = torch.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
        q_tp1 = torch.min(q_tp1, twin_q_tp1)

    q_tp1_best = torch.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
    q_tp1_best_masked = \
        (1.0 - train_batch[SampleBatch.DONES].float()) * \
        q_tp1_best

    # Compute RHS of bellman equation.
    q_t_selected_target = (train_batch[SampleBatch.REWARDS] +
                           gamma**n_step * q_tp1_best_masked).detach()

    # Compute the error (potentially clipped).
    if twin_q:
        td_error = q_t_selected - q_t_selected_target
        twin_td_error = twin_q_t_selected - q_t_selected_target
        if use_huber:
            errors = huber_loss(td_error, huber_threshold) \
                + huber_loss(twin_td_error, huber_threshold)
        else:
            errors = 0.5 * \
                (torch.pow(td_error, 2.0) + torch.pow(twin_td_error, 2.0))
    else:
        td_error = q_t_selected - q_t_selected_target
        if use_huber:
            errors = huber_loss(td_error, huber_threshold)
        else:
            errors = 0.5 * torch.pow(td_error, 2.0)

    ############################## BIER ########################################
    critic_loss = torch.mean(train_batch[PRIO_WEIGHTS] * errors)
    ############################## BIER ########################################

    # Add l2-regularization if required.
    if l2_reg is not None:
        for name, var in policy.model.policy_variables(as_dict=True).items():
            if "bias" not in name:
                actor_loss += (l2_reg * l2_loss(var))
        for name, var in policy.model.q_variables(as_dict=True).items():
            if "bias" not in name:
                critic_loss += (l2_reg * l2_loss(var))

    # Model self-supervised losses.
    if policy.config["use_state_preprocessor"]:
        # Expand input_dict in case custom_loss' need them.
        input_dict[SampleBatch.ACTIONS] = train_batch[SampleBatch.ACTIONS]
        input_dict[SampleBatch.REWARDS] = train_batch[SampleBatch.REWARDS]
        input_dict[SampleBatch.DONES] = train_batch[SampleBatch.DONES]
        input_dict[SampleBatch.NEXT_OBS] = train_batch[SampleBatch.NEXT_OBS]
        [actor_loss, critic_loss] = model.custom_loss(
            [actor_loss, critic_loss], input_dict)

    # Store values for stats function.
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.td_error = td_error
    policy.q_t = q_t

    ############################## BIER ########################################
    # Log every LOG_INTERVAL steps or if there is a NaN
    if parameters.GLOBAL_STEP_COUNT % (parameters.LOG_INTERVAL * parameters.REPLAY_RATIO) == 0:

        # Get quantities and names to be logged
        vals = [train_batch[PRIO_WEIGHTS], errors, actor_loss, td_error, q_t]
        names = ["Priority Weights", "Errors",
                 "Actor Loss", "TD Error", "Q-vals"]

        # Add critic loss
        if isinstance(critic_loss, list):
            vals.extend(critic_loss)
            names.extend(
                ["Critic Loss {}".format(i) for i in range(len(critic_loss))])
        else:
            vals.append(critic_loss)
            names.append("Critic Loss")

        # Loop over values and metrics
        for val, name in zip(vals, names):

            # Compute moments and log to Tensorboard
            if torch.numel(val) > 1:
                mean_val = torch.mean(val)
                var_val = torch.var(val)

                # Compute min and max
                max_val = torch.max(val)
                min_val = torch.min(val)

                values = [mean_val, var_val, max_val, min_val]
                types = ["Mean", "Variance", "Maximum", "Minimum"]

            else:
                # Variables and types
                values = [val.flatten()]
                types = [""]

            # Log to tensorboard
            for v, t in zip(values, types):
                if parameters.TB_WRITER is not None:  # Initialized as None
                    parameters.TB_WRITER.add_scalar(
                        "DDPG Metrics/{}/{}".format(name, t), v,
                        global_step=parameters.GLOBAL_STEP_COUNT)
    ############################## BIER ########################################

    # Return two loss terms (corresponding to the two optimizers, we create).
    return policy.actor_loss, policy.critic_loss


def make_ddpg_optimizers(policy, config):
    """Create separate optimizers for actor & critic losses."""

    # Set epsilons to match tf.keras.optimizers.Adam's epsilon default.
    policy._actor_optimizer = torch.optim.Adam(
        params=policy.model.policy_variables(),
        lr=config["actor_lr"],
        eps=1e-7)

    policy._critic_optimizer = torch.optim.Adam(
        params=policy.model.q_variables(), lr=config["critic_lr"], eps=1e-7)

    # Return them in the same order as the respective loss terms are returned.
    return policy._actor_optimizer, policy._critic_optimizer


def apply_gradients_fn(policy):
    # For policy gradient, update policy net one time v.s.
    # update critic net `policy_delay` time(s).
    if policy.global_step % policy.config["policy_delay"] == 0:
        policy._actor_optimizer.step()

    policy._critic_optimizer.step()

    # Increment global step & apply ops.
    policy.global_step += 1


def build_ddpg_stats(policy, batch):
    stats = {
        "actor_loss": policy.actor_loss,
        "critic_loss": policy.critic_loss,
        "mean_q": torch.mean(policy.q_t),
        "max_q": torch.max(policy.q_t),
        "min_q": torch.min(policy.q_t),
        "mean_td_error": torch.mean(policy.td_error),
        "td_error": policy.td_error,
    }
    return stats


def before_init_fn(policy, obs_space, action_space, config):
    # Create global step for counting the number of update operations.
    policy.global_step = 0


class ComputeTDErrorMixin:
    def __init__(self, loss_fn):
        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask,
                             importance_weights):
            input_dict = self._lazy_tensor_dict({
                SampleBatch.CUR_OBS: obs_t,
                SampleBatch.ACTIONS: act_t,
                SampleBatch.REWARDS: rew_t,
                SampleBatch.NEXT_OBS: obs_tp1,
                SampleBatch.DONES: done_mask,
                PRIO_WEIGHTS: importance_weights,
            })
            # Do forward pass on loss to update td errors attribute
            # (one TD-error value per item in batch to update PR weights).
            loss_fn(self, self.model, None, input_dict)

            # Self.td_error is set within actor_critic_loss call.
            return self.td_error

        self.compute_td_error = compute_td_error


class TargetNetworkMixin:
    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self, tau=None):
        tau = tau or self.config.get("tau")
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        # Full sync from Q-model to target Q-model.
        if tau == 1.0:
            self.target_model.load_state_dict(self.model.state_dict())
        # Partial (soft) sync using tau-synching.
        else:
            model_vars = self.model.variables()
            target_model_vars = self.target_model.variables()
            assert len(model_vars) == len(target_model_vars), \
                (model_vars, target_model_vars)
            for var, var_target in zip(model_vars, target_model_vars):
                var_target.data = tau * var.data + \
                    (1.0 - tau) * var_target.data


def setup_late_mixins(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy, ddpg_actor_critic_loss_likelihood_weighted)
    TargetNetworkMixin.__init__(policy)

############################## BIER ############################################
# Build likelihood-weighted DPPG policy
DDPGTorchPolicyLikelihoodWeighted = build_torch_policy(
    name="DDPGTorchPolicy",
    loss_fn=ddpg_actor_critic_loss_likelihood_weighted,
    get_default_config=lambda: ray.rllib.agents.ddpg.ddpg.DEFAULT_CONFIG,
    stats_fn=build_ddpg_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    extra_grad_process_fn=utils.execution.grad_clipping.apply_grad_clipping,
    optimizer_fn=make_ddpg_optimizers,
    validate_spaces=validate_spaces,
    before_init=before_init_fn,
    before_loss_init=setup_late_mixins,
    action_distribution_fn=get_distribution_inputs_and_class,
    make_model_and_action_dist=build_ddpg_models_and_action_dist,
    apply_gradients_fn=apply_gradients_fn,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
    ])
############################## BIER ############################################

# Build non-likelihood-weighted DPPG policy
DDPGTorchPolicy = build_torch_policy(
    name="DDPGTorchPolicy",
    loss_fn=ddpg_actor_critic_loss,
    get_default_config=lambda: ray.rllib.agents.ddpg.ddpg.DEFAULT_CONFIG,
    stats_fn=build_ddpg_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    extra_grad_process_fn=utils.execution.grad_clipping.apply_grad_clipping,
    optimizer_fn=make_ddpg_optimizers,
    validate_spaces=validate_spaces,
    before_init=before_init_fn,
    before_loss_init=setup_late_mixins,
    action_distribution_fn=get_distribution_inputs_and_class,
    make_model_and_action_dist=build_ddpg_models_and_action_dist,
    apply_gradients_fn=apply_gradients_fn,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
    ])
############################## BIER ############################################
