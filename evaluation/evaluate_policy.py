"""Script to load a trained policy and evaluate it on a specified environment.

NOTE: This evaluation module is still under development, and is not yet ready
for use.
"""
import argparse
import pickle
from agents.likelihood_weighted.sac import SACTorchPolicy, actor_critic_loss
from ray.rllib.agents.sac.sac_torch_policy import build_sac_model_and_action_dist
import gym
import os

def parse_arguments():
    """Function defining a CLI parser."""
    # Declare a CLI parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-f", "--policy_weights_path", type=str,
                        help="Path to trained policy.")
    parser.add_argument("-env", "--environment", type=str,
                        help="Name of environment to use.")

    # Return parsed arguments
    return parser.parse_args()


def main(args):
    """Main evaluation function given a set of CLI arguments."""
    # Load the model weights
    with open(args.policy_weights_path, "rb") as policy_weights:
        weights = pickle.load(policy_weights)
        policy_weights.close()

    # Set RL environment variable
    os.environ["RL_ENV"] = args.environment
    from agents.likelihood_weighted.sac import DEFAULT_CONFIG

    # Load the environment
    env = gym.make('HalfCheetah-v2')
    state0 = env.reset()

    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space

    # Create a torch policy
    EvalPolicy = SACTorchPolicy.with_updates(
        name="EvalPolicy", loss_fn=actor_critic_loss)
   # policy = EvalPolicy(obs_space, action_space, {})
   # policy.set_weights(weights)

    model, action_dist_class = build_sac_model_and_action_dist(
        EvalPolicy, obs_space, action_space, DEFAULT_CONFIG)

    #CustomPolicy = SACTorchPolicy.with_updates(
    #    name="MyCustomPPOTFPolicy",
    #    loss_fn=actor_critic_loss)
    #print(type(CustomPolicy))
    input_dict = {"obs": state0}
    model(state0)

    # Now run iterations with the trained policy
    for _ in range(1000):
        env.step(env.action_space.sample())  # take a random action
        env.render()
    env.close()


if __name__ == '__main__':
    pass  # TODO(rms): Fix issues with policy loading
    #args = parse_arguments()
    #main(args)
