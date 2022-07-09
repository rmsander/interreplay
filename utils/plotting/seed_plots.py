"""Script for parsing tensorboard data and creating average and confidence
interval (+/- 1 standard deviation) plots and for analyzing ablations using
json result files produced by RLlib.

In addition to containing a script for running these files, this package also
contains utility functions for extracting rewards of single files, parsing
directories, and looping through sets of ablation results to combine multi-seed
results.
"""
# External Python packages
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
import matplotlib
#matplotlib.rcParams["savefig.directory"] = os.path.join("~", "Documents", "meng",
#                                                        "thesis", "neurips", "plots", "td3")

plt.style.use('ggplot')  # Set style to make plots nicer

# Native Python packages
import ast
import argparse
import json
import os


def get_reward_data(f_json, mode="evaluate"):
    """Helper function for extracting reward data from a JSON file written to
    Tensorboard. This function is typically used iteratively to parse
    training and evaluation data from the RL training run.

    Parameters:
        f_json (str): String denoting the file path of the JSON file.
        mode (str): Whether to extract training or evaluation data. Defaults
            to "evaluate" for evaluation data.

    Returns:
        reward_data (dict): A dictionary in which each key is a type of reward
    """
    # Extract reward data
    with open(f_json, "r") as json_file:
        json_data = [json.loads(line) for line in json_file]  # Extract all data
        keys = ["episode_reward_min", "episode_reward_max",
                "episode_reward_mean", "num_steps_trained", "num_steps_sampled"]
        reward_data = {key: [] for key in keys}
        for json_dict in json_data:  # Iterate over dictionaries
            if (mode == "evaluate") and ("evaluation" not in json_dict.keys()):
                continue
            for key in keys:
                if key in ["num_steps_trained", "num_steps_sampled"]:  # X-axes
                    reward_data[key].append(json_dict["info"][key])
                else:
                    if mode == "evaluate":
                        reward_data[key].append(json_dict["evaluation"][key])
                    else:
                        reward_data[key].append(json_dict[key])
    return reward_data


def get_reward_data_smooth_window(f_json, mode="evaluate", window_length=5):
    """Helper function for extracting reward data from a JSON file written to
    Tensorboard. This function is typically used iteratively to parse
    training and evaluation data from the RL training run.

    In addition to extracting reward data, this approach also smooths the
    reward data using an averaging window.

    Parameters:
        f_json (str): String denoting the file path of the JSON file.
        mode (str): Whether to extract training or evaluation data. Defaults
            to "evaluate" for evaluation data.

    Returns:
        reward_data (dict): A dictionary in which each key is a type of reward
    """
    # Extract reward data
    with open(f_json, "r") as json_file:
        json_data = [json.loads(line) for line in json_file]  # Extract all data
        keys = ["episode_reward_min", "episode_reward_max",
                "episode_reward_mean", "num_steps_trained", "num_steps_sampled"]
        reward_data = {key: [] for key in keys}
        reward_data_smoothed = {key: [] for key in keys}
        for json_dict in json_data:  # Iterate over dictionaries
            if (mode == "evaluate") and ("evaluation" not in json_dict.keys()):
                continue
            for key in keys:
                if key in ["num_steps_trained", "num_steps_sampled"]:  # X-axes

                    # True values dictionary
                    reward_data[key].append(json_dict["info"][key])

                    # Smoothed values dictionary
                    reward_data_smoothed[key].append(json_dict["info"][key])

                else:
                    if mode == "evaluate":

                        # Take previous window_length-1 values
                        prev_values = reward_data[key][-(window_length-1):]

                        # Take window average with most recent value
                        window_avg = (np.sum(prev_values) + json_dict["evaluation"][key]) / window_length

                        # Now append to smoothed (not true!) data
                        reward_data_smoothed[key].append(window_avg)

                        # Append true reward for future averaging
                        reward_data[key].append(json_dict["evaluation"][key])
                    else:

                        # Take previous window_length-1 values
                        prev_values = reward_data[key][-(window_length - 1):]

                        # Take window average with most recent value
                        window_avg = (np.sum(prev_values) + json_dict[key]) / window_length

                        # Now append to smoothed (not true!) data
                        reward_data_smoothed[key].append(window_avg)

                        reward_data[key].append(json_dict[key])

    return reward_data, reward_data_smoothed


def merge_reward_data(fnames, mode="evaluate", smoothed=False, window_length=5):
    """Function to extract and merge reward data for plotting standard deviation
    and mean rewards across different random seeds.

    Parameters:
        fnames (list): A list of .json paths corresponding to the rewards weights
            wish to average and find the standard deviation of.
        mode (str): Whether to extract training or evalaution data. Defaults
            to "evaluate" for evalaution data.

    Returns:
        min_len_vals (list): A list of the rewards averaged across different
            random seeds. The length of this will be the minimum of the lengths
            of the experiments.
        max_len_vals (list): A list of the rewards averaged across different
            random seeds. The length of this will be the maximum of the lengths
            of the experiments.
        std_dev_min_len (list): A list of the standard deviation of rewards
            across different random seeds. The length of this will be the
            minimum of the lengths of the experiments.
        std_dev_max_len (list): A list of the standard deviation of rewards
            across different random seeds. The length of this will be the
            maximum of the lengths of the experiments.
        smoothed (bool): Whether to smooth the data. Defaults to False.
        window_length (int): If smoothing, the length of the window to smooth
            over.
    """
    # Get rewards data from all applicable JSON files
    reward_data = []
    for fname in fnames:  # Iterate through filenames

        # Get smoothed or unsmoothed data for plotting
        if smoothed:
            print("USING SMOOTHED DATA")
            _, reward_smoothed = get_reward_data_smooth_window(
                fname, mode=mode, window_length=window_length)
            reward_data.append(reward_smoothed)
        else:
            print("USING UNSMOOTHED DATA")
            reward_data.append(get_reward_data(fname, mode=mode))

    # Compute averaged results for the shortest experiment length
    min_lengths = {}
    max_lengths = {}
    for key in list(reward_data[0].keys()):
        min_lengths[key] = min(
            [len(reward_data[i][key]) for i in range(len(reward_data))])
        max_lengths[key] = max(
            [len(reward_data[i][key]) for i in range(len(reward_data))])

    # Initialize variable to store mean results
    min_len_vals = {}
    max_len_vals = {}
    f_elt_counts = {}
    N_files = len(fnames)

    # Add the results together before averaging
    for f_data in reward_data:
        print("SMOOTHED?")
        print("Length: {}, Value: {}".format(len(f_data["episode_reward_mean"]),
                                             f_data["episode_reward_mean"][-1]))
        for key, value in f_data.items():
            if key not in min_len_vals:
                min_len_vals[key] = np.array(value[:min_lengths[key]])
                max_len_vals[key] = np.array(value)
                f_elt_counts[key] = np.ones(len(value))
            else:
                min_len_vals[key] = np.add(min_len_vals[key],
                                           value[:min_lengths[key]])
                key_len = len(max_len_vals[key])
                val_len = len(value)
                if len(max_len_vals[key]) < len(value):
                    num_to_add = len(value) - len(max_len_vals[key])
                    max_len_vals[key] = np.concatenate([max_len_vals[key],
                                                        np.zeros(num_to_add)])
                    max_len_vals[key] = np.add(max_len_vals[key], value)
                    f_elt_counts[key] = np.concatenate([f_elt_counts[key],
                                                        np.zeros(num_to_add)])
                    f_elt_counts[key] = np.add(f_elt_counts[key], np.ones(len(value)))
                elif len(value) < len(max_len_vals[key]):
                    num_to_add = len(max_len_vals[key]) - len(value)
                    value_counts = np.concatenate([np.ones(len(value)),
                                                   np.zeros(num_to_add)])
                    value = np.concatenate([value, np.zeros(num_to_add)])
                    max_len_vals[key] = np.add(max_len_vals[key], value)
                    f_elt_counts[key] = np.add(f_elt_counts[key], value_counts)
                elif len(value) == len(max_len_vals[key]):
                    max_len_vals[key] = np.add(max_len_vals[key], value)
                    f_elt_counts[key] = np.add(f_elt_counts[key],
                                               np.ones(len(value)))

    # Now average
    for key in list(min_len_vals.keys()):
        min_len_vals[key] = min_len_vals[key] / N_files
        max_len_vals[key] = np.divide(max_len_vals[key], f_elt_counts[key])

    # Now compute standard deviation across each
    std_dev_min_len = {}
    std_dev_max_len = {}
    for key in list(min_len_vals.keys()):
        std_dev_min_len[key] = np.std(
            [reward_data[i][key][:min_lengths[key]] for i in range(N_files)], axis=0)

        # TODO(rms):  Fix max std computation
        num_to_add = len(max_len_vals[key]) - len(std_dev_min_len[key])
        std_dev_max_len[key] = np.concatenate([std_dev_min_len[key],
                                               np.zeros(num_to_add)])

    return min_len_vals, max_len_vals, std_dev_min_len, std_dev_max_len


def merge_reward_data_multiple(all_fnames, mode="evaluate", smoothed=False,
                               window_length=5):
    """Function to extract and merge reward data for plotting standard deviation
    and mean rewards across different random seeds. Designed for plotting
    comparatively across multiple sets of runs/ablations, each of which is
    across several seeds.

    Parameters:
        all_fnames (list): A list of lists of .json paths corresponding to the
            rewards weights wish to average and find the standard deviation of.
        mode (str): Whether to extract training or evaluation data. Defaults
            to "evaluate" for evaluation data.

    Returns:
        all_min_len_vals (list): A list of lists of the rewards averaged across
            different random seeds. The length of this will be the minimum
            of the lengths of the experiments.
        all_max_len_vals (list): A list of lists of the rewards averaged across
            different random seeds. The length of this will be the maximum of
            the lengths of the experiments.
        all_std_dev_min_len (list): A list of lists of the standard deviation of
            rewards across different random seeds. The length of this will be
            the minimum of the lengths of the experiments.
        all_std_dev_max_len (list): A list of lists of the standard deviation of
            rewards across different random seeds. The length of this will be
            the maximum of the lengths of the experiments.
        smoothed (bool): Whether to smooth the data. Defaults to False.
        window_length (int): If smoothing, the length of the window to smooth
            over.
    """
    # Initialize output lists for storing min, max, and standard deviation
    all_min_len_vals = []
    all_max_len_vals = []
    all_std_dev_min_len = []
    all_std_dev_max_len = []

    # Iterate over filenames
    for fnames in all_fnames:

        # Merge reward data for filenames
        min_len_vals, max_len_vals, std_dev_min_len, std_dev_max_len = \
            merge_reward_data(fnames, mode=mode, smoothed=smoothed,
                              window_length=window_length)

        # Add entries for each criteria
        all_min_len_vals.append(min_len_vals)
        all_max_len_vals.append(max_len_vals)
        all_std_dev_min_len.append(std_dev_min_len)
        all_std_dev_max_len.append(std_dev_max_len)

    return all_min_len_vals, all_max_len_vals, \
           all_std_dev_min_len, all_std_dev_max_len


def plot(reward_data_min_len, reward_data_max_len, std_dev_min_len,
         std_dev_max_len, out_path, environment, mode="evaluate"):
    """Function for creating reward plots across multiple random seeds with
    confidence interval plotting.

    Parameters:
        reward_data_min_len (dict): Dictionary corresponding to reward data,
            which is parsed using the functions above. The length is equal to
            the shortest length over all seeds
            (i.e. the shortest-running experiment).
        reward_data_max_len (dict): Dictionary corresponding to reward data,
            which is parsed using the functions above. The length is equal to
            the longest length over all seeds
            (i.e. the longest-running experiment).
        std_dev_min_len (dict): Dictionary corresponding to standard deviation
            data, parsed using the functions above. The length is equal to
            the shortest length over all seeds
            (i.e. the shortest-running experiment).
        std_dev_max_len (dict): Dictionary corresponding to standard deviation
            data, parsed using the functions above. The length is equal to
            the longest length over all seeds
            (i.e. the longest-running experiment).
        out_path (str): An output path where files will be saved once plotted.
        environment (str): Environment name for which data is evaluated.
            Defaults to HalfCheetah-v2.
    """
    # Determine which type of plot to make
    if mode == "evaluate":
        title_reward_type = "Evaluation"
    else:
        title_reward_type = "Training"

    # Loop over min and max lengths for reward
    for reward_data, std_dev, reward_type in zip(
            [reward_data_min_len, reward_data_max_len],
            [std_dev_min_len, std_dev_max_len],
            ["Min Length", "Max Length"]):

        # Loop over all data components
        for x_axis, x_label in zip(
                ["num_steps_trained", "num_steps_sampled"],
                ["Number of Steps Trained", "Environment Interactions"]):

            # Loop over episode reward types (min, max, mean)
            for key in ["episode_reward_min",
                        "episode_reward_max",
                        "episode_reward_mean"]:

                # Extract quantities of interest for plotting
                y_bar = reward_data[key]  # Mean value for reward type
                y_std = std_dev[key]  # Standard deviation for reward type
                x_bar = reward_data[x_axis]  # Number of steps/env interactions

                # Plot confidence interval
                interval_plus = np.add(y_bar, y_std)  # Upper bound
                interval_minus = np.subtract(y_bar, y_std)  # Lower bound

                # Get type of reward for plotting
                if "max" in key:
                    type_reward = "Max"
                elif "mean" in key:
                    type_reward = "Mean"
                elif "min" in key:
                    type_reward = "Min"

                # Create the plot with confidence intervals
                sns.lineplot(data=reward_data, x=x_axis, y=key)
                plt.fill_between(x_bar, interval_plus, interval_minus,
                                 color='gray', alpha=0.2)
                plt.legend()
                plt.xlabel(x_label)
                plt.ylabel("{} Reward".format(title_reward_type))
                plt.title(environment)
                #plt.title("Episodic {} {} Reward By {}, {},".format(
                #    title_reward_type, type_reward, x_label,
                #    reward_type, environment))
                plt.savefig(out_path.format(title_reward_type, key, x_label))
                plt.show()


def plot_multiple(all_rewards_min, all_rewards_max, all_std_dev_min,
                  all_std_dev_max, labels, out_path,
                  environment="HalfCheetah-v2", mode="evaluate", other_means=None,
                  other_stds=None, other_labels=None, legend=False):
    """Function for plotting overlayed results from multiple experiments, each
    across random seeds.

    Parameters:
        all_rewards_min (list): A list of dictionaries corresponding to reward
            data parsed using the functions above. The length is equal to the
            shortest length over all seeds
            (i.e. the shortest-running experiment).
        all_rewards_max (list): A list of dictionaries corresponding to reward
            data parsed using the functions above. The length is equal to the
            longest length over all seeds
            (i.e. the longest-running experiment).
        all_std_dev_min (list): A list of dictionaries corresponding to standard
            deviation data, parsed using the functions above. The length is
            equal to the longest length over all seeds
            (i.e. the shortest-running experiment).
        all_std_dev_max (list): A list of dictionaries corresponding to standard
            deviation data, parsed using the functions above. The length is
            equal to the longest length over all seeds
            (i.e. the longest-running experiment).
        labels (list):  A list of labels corresponding to the experiment names.
            These indices correspond to the indices of the lists all_rewards
            and all_std_dev.
        out_path (str): An output path where files will be saved once plotted.
        environment (str): Environment name for which data is evaluated.
            Defaults to HalfCheetah-v2.
        legend (bool): Whether to use a legend when plotting.
    """
    # Determine which type of plot to make
    if mode == "evaluate":
        title_reward_type = "Evaluation"
    else:
        title_reward_type = "Training"

    # Loop over all reward types and min and max length segments
    for all_rewards, all_std_dev, reward_type in zip(
            [all_rewards_min, all_rewards_max],
            [all_std_dev_min, all_std_dev_max],
            ["Min Length", "Max Length"]):

        # Loop over different values for the x-axis (training steps, env inter.)
        for x_axis, x_label in zip(
                ["num_steps_sampled", "num_steps_trained"],
                ["Environment Interactions", "Steps Trained"]):

            # Loop over different types of reward
            for key in ["episode_reward_min",
                        "episode_reward_max",
                        "episode_reward_mean"]:

                # Generate colors for plotting
                colors = list(mcolors.BASE_COLORS)[:len(all_rewards)]

                # Loop over experiments
                for reward_data, std_dev, c, l in zip(
                        all_rewards, all_std_dev, colors, labels):

                    # Extract quantities of interest for plotting
                    y_bar = reward_data[key]
                    y_std = std_dev[key]
                    x_bar = reward_data[x_axis]

                    # Plot confidence interval
                    interval_plus = np.add(y_bar, y_std)
                    interval_minus = np.subtract(y_bar, y_std)

                    # Get type of reward for plotting
                    if "max" in key:
                        type_reward = "Max"
                    elif "mean" in key:
                        type_reward = "Mean"
                    elif "min" in key:
                        type_reward = "Min"

                    # Create the plot with CI
                    plt.plot(x_bar, y_bar, color=c, label=l)
                    plt.fill_between(x_bar, interval_plus, interval_minus,
                                     color=c, alpha=0.2)

                    # Get length
                    N = len(y_bar)

                    # Lastly, display on console
                    """
                    print("_________________________________")
                    print("Type of Reward: {}, Label: {}".format(
                        type_reward, l))

                    # Print reward at 0.25x
                    print("Mean, Step 5K: {}, Std: {}".format(
                        round(y_bar[min(4, len(y_bar)-1)], 3),
                        round(y_std[min(4, len(y_bar)-1)], 3)))

                    # Print reward at 0.5x
                    print("Mean, Step 10K: {}, Std: {}".format(
                        round(y_bar[min(9, len(y_bar)-1)], 3),
                        round(y_std[min(9, len(y_bar)-1)], 3)))

                    # Print reward at 0.5x
                    print("Mean, Step 20K: {}, Std: {}".format(
                        round(y_bar[min(19, len(y_bar) - 1)], 3),
                        round(y_std[min(19, len(y_bar) - 1)], 3)))
                    """

                    # Print reward at 1.0x
                    print("Mean, Step {}K: {}, Std: {}".format(
                        len(y_bar), round(y_bar[-1], 3), round(y_std[-1], 3)))
                    print("_________________________________")

                # Loop over baselines, if not None
                baseline_colors = list(mcolors.TABLEAU_COLORS)[:len(other_labels)]
                assert len(other_means) == len(other_stds)
                for mu, sigma, label, b_color in zip(
                        other_means, other_stds, other_labels, baseline_colors):

                    # Extract quantities of interest for plotting
                    y_bar = [mu] * len(reward_data[x_axis])
                    y_std = [sigma] * len(reward_data[x_axis])
                    x_bar = reward_data[x_axis]

                    # Plot confidence interval
                    interval_plus = np.add(y_bar, y_std)
                    interval_minus = np.subtract(y_bar, y_std)

                    # Create the plot with CI
                    plt.plot(x_bar, y_bar, color=b_color, label=label, linestyle='--')
                    plt.fill_between(x_bar, interval_plus, interval_minus,
                                     color=b_color, alpha=0.2)

                # After all data is plotted, finalize, save, and close graph
                if legend:
                    plt.legend()
                plt.xlabel(x_label)
                plt.ylabel("{} Reward".format(title_reward_type))
                plt.title(environment)
                #plt.title("{} {} Reward By {} in {}".format(
                #    title_reward_type, type_reward, x_label, environment))
                plt.savefig(out_path.format(title_reward_type, key, x_label))
                plt.show()
                plt.clf()


def parse_arguments():
    """Argument parser for running this as a script to generate results for
    single or multiple experiments.

    Parameters:
        args (Object): Arguments object corresponding to the parsed CLI
            arguments.
    """
    # Create ArgumentParser object.
    parser = argparse.ArgumentParser()

    # Add arg for base file paths
    parser.add_argument("-base_path", "--base_path", type=str,
                        help="Base path location of JSON files")
    parser.add_argument("-base_dir", "--base_dir", type=str,
                        help="Base directory that captures longest common path "
                             "of directories.")
    parser.add_argument("-seeds", "--seed_list", type=str,
                        help="List of seeds we compute results over. These "
                             "seeds should appear as the suffix of each JSON "
                             "file name.")
    parser.add_argument("-env", "--environment", type=str,
                        default="HalfCheetah-v2",
                        help="Name of environment for plots.")
    parser.add_argument("-labels", "--labels", type=str,
                        help="Experiment keys (used for plotting).")
    parser.add_argument("-other_means", "--other_means", type=str, default="[]",
                        help="Means values for plotting other baselines.")
    parser.add_argument("-other_stds", "--other_stds", type=str, default="[]",
                        help="Std values for plotting other baselines.")
    parser.add_argument("-other_labels", "--other_labels", type=str, default="[]",
                        help="Names of.")
    parser.add_argument("-outpath", "--outpath", type=str,
                        default="/home/ryansander/Documents/mit/meng/seed_plotting/{}"
                                "_reward_key-{}-axis-{}.png",
                        help="Path (with extension) to save image files to.")
    parser.add_argument("-wlen", "--window_length", type=int, default=5,
                        help="Length of averaging window over which to smooth"
                             "results.")
    # Parse args
    return parser.parse_args()

def main():
    """Run data merging and plotting."""
    # Get CLI arguments
    args = parse_arguments()
    seeds = ast.literal_eval(args.seed_list)
    base_paths = ast.literal_eval(args.base_path)
    base_dir = args.base_dir
    labels = ast.literal_eval(args.labels)
    window_length = args.window_length

    # Get values for other baselines and labels
    other_means = ast.literal_eval(args.other_means)
    other_stds = ast.literal_eval(args.other_stds)
    other_labels = ast.literal_eval(args.other_labels)

    # Get file paths for extracting and plotting results
    paths = [[os.path.join(base_dir, base_path, "{}_result.json".format(seed)) \
              for seed in seeds] for base_path in base_paths]

    # Extract results
    if len(base_paths) > 1:  # Plot results from more than one experiment
        for mode in ["evaluate", "train"]:

            # Extract and plot smooth results
            print("EXTRACTING AND PLOTTING SMOOTHED RESULTS...")
            all_vals_min, all_vals_max, all_std_min, all_std_max = \
                merge_reward_data_multiple(paths, mode=mode, smoothed=True,
                                           window_length=window_length)
            plot_multiple(all_vals_min, all_vals_max, all_std_min, all_std_max,
                          labels, args.outpath, args.environment, mode=mode,
                          other_means=other_means, other_stds=other_stds,
                          other_labels=other_labels)

            # Extract and plot unsmoothed results
            print("EXTRACTING AND PLOTTING UNSMOOTHED RESULTS...")
            all_vals_min, all_vals_max, all_std_min, all_std_max = \
                merge_reward_data_multiple(paths, mode=mode, smoothed=False)
            plot_multiple(all_vals_min, all_vals_max, all_std_min, all_std_max,
                          labels, args.outpath, args.environment, mode=mode)

    else:  # Plot one result
        paths = paths[0]  # Just "flatten" the list
        for mode in ["evaluate", "train"]:

            # Plot smoothed results
            print("EXTRACTING AND PLOTTING SMOOTHED RESULTS...")
            vals_min, vals_max, std_dev_min, std_dev_max = merge_reward_data(
                paths, mode=mode, smoothed=True, window_length=window_length)
            # Plot results
            plot(vals_min, vals_max, std_dev_min, std_dev_max, args.outpath,
                 args.environment, mode=mode)

            # Plot unsmoothed results
            print("EXTRACTING AND PLOTTING UNSMOOTHED RESULTS...")
            vals_min, vals_max, std_dev_min, std_dev_max = merge_reward_data(
                paths, mode=mode, smoothed=False)
            # Plot results
            plot(vals_min, vals_max, std_dev_min, std_dev_max, args.outpath,
                 args.environment, mode=mode)


if __name__ == '__main__':
    main()
