"""Script to run multiple tune experiments concurrently, i.e. on the same node.
Calls custom_trainer.py with a given set of arguments."""
# Native Python packages
import os
import argparse


def parse_inputs():
    """Used as a CLI parser and argument formatter for use in inputting arguments
    to the custom_training.py file.py.

    Note that default_args lead some behaviors, such as running in local mode,
    to be enabled by default.

    Returns:
        args_list (list): A list of arguments. Each element corresponds to the
            CLI parameters for the given call to custom_training.py.
    """
    # Create argument parser, add arguments, and parse them
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_path", "--input_path", type=str,
                        help="File path location for input arguments")
    args = parser.parse_args()

    # Begin adding arguments
    args_list = []
    keys = ["seed", "env", "custom_replay_buffer", "agent_name",
            "round_robin_weights", "local_dir", "trainer"]
    default_args = ["use_delta", "gaussian_process", "gpytorch", "kneighbors 50",
                   "prioritized_replay", "mixup_interpolation", "train_size 1000",
                   "retrain_interval 1000", "kernel matern", "mean_type zero",
                   "matern_nu 1.5", "global_hyperparams", "use_ard", "local_mode",
                   "normalize", "mc_hyper", "--interp_prio_update", "--use_queue"]

    # Parse arguments from input files
    with open(args.input_path, "r") as inputs:
        for i, line in enumerate(inputs):  # First line is additional arguments
            if i == 0:  # Additional arguments
                added_args = line
            else:
                # Creates string to store CLI config for custom_training.py
                args_list.append("")  # Make sure to start with blank string
                line_args = line.split(" ")

                # Adds arguments for parsed args
                for a, key in zip(line_args, keys):
                    # Stripping ensures no new lines are created
                    args_list[-1] += "--{} {} ".format(key, a.strip())

                # Adds arguments for default args
                for d in default_args:
                    # Stripping ensures no new lines are created
                    args_list[-1] += "--{} ".format(d.strip())

                # Adds final args that are applied to all in call
                args_list[-1] += added_args.strip()

                # Add a sleep value - 30 seconds between each job
                args_list[-1] += " --sleep_time {} ".format(30 * (i-1))

        inputs.close()  # Close file

    return args_list


def call_experiments(args):
    """Function to stitch together arguments for custom_training.py into a single
    concurrent custom_training.py call.

    Parameters:
        args_list (list): A list of arguments. Each element corresponds to the
            CLI parameters for the given call to custom_training.py.
    """
    # Create list to store individual calls
    command_list = [
        "python3 custom_training.py {} & ".format(a.strip()) for a in args]
    command_str = ""  # Initialize command

    # String-concatenate single commands into concurrent command
    for c in command_list:
        command_str += c
    command_str = command_str[:-2]  # Remove final &+space

    # Add final formats
    command_str = "(" + command_str + ")"
    print("Current directory is: {}".format(os.getcwd()))
    print("Command is: {}".format(command_str))

    # Call command to run concurrent experiments
    os.system(command_str)


def main(args):
    """Main invoked script. Calls functions above to run
    concurrent experiments."""
    call_experiments(args)  # Run experiments with parsed arguments


if __name__ == '__main__':
    args_list = parse_inputs()  # Parse arguments and format
    main(args_list)  # Run experiments concurrently
