"""Function to applied customized gradient clipping."""
# External Python packages
import torch
import torch.nn
import numpy as np

# Custom Python packages/modules
import parameters


def apply_grad_clipping(policy, optimizer, loss):
    """Adapted from RLlib's apply_grad_clipping function. Adds a tensorboard
    logger for viewing the gradient on each training step.

    Parameters:
        policy (Policy): Policy object with attributes corresponding to actor
            and critic networks as torch.Module objects.
        optimizer (Optimizer): PyTorch optimizer object containing information
            regarding gradients.
        loss (loss): Loss object used by convention for this function. Note
            that loss is not explicitly utilized in this function.

    Returns:
        info (dict): Dictionary of info returned and used for metric plotting
            and for training the RLlib agents.
    """
    info = {}
    if policy.config["grad_clip"]:
        for i, param_group in enumerate(optimizer.param_groups):
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
            params = list(
                filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                # Compute gradient norm (NOTE: this is unclipped)
                grad_gnorm = torch.nn.utils.clip_grad_norm_(
                    params, policy.config["grad_clip"])

                # Log to info dictionary
                info["gnorm"] = grad_gnorm

                # Write to tensorboard
                time_to_log = parameters.GLOBAL_STEP_COUNT % \
                              (parameters.REPLAY_RATIO * parameters.LOG_INTERVAL) == 0
                if time_to_log:
                    label = "gnorm/Clipped Gradient Norm, " \
                            "Param Group {}".format(i)
                    parameters.TB_WRITER.add_scalar(
                        label, grad_gnorm.detach().cpu().numpy(),
                        global_step=parameters.GLOBAL_STEP_COUNT)
    return info
