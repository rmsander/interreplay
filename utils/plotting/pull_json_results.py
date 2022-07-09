"""File to extract JSON results (result.json) files from within folders. These
files are produced from the Ray RLlib framework."""

# Native Python packages
import os
import glob
import argparse


def pull_results_files(results_dir):
    """Function to pull results files from results_dir.

    Parameters:
        results_dir (str): String corresponding to absolute path of directory to
            move results from.
    """
    # Check if path exists
    if not os.path.exists(results_dir):
        raise Exception("Directory {} not found.".format(results_dir))

    # Now find results files
    for subdir in os.listdir(results_dir):

        # Get sub-directory path
        subdir_path = os.path.join(results_dir, subdir)

        # Check if file
        if os.path.isfile(subdir_path):
            continue

        # Get the seed name for distinguishing the different results files
        seed_name = subdir.split("_")[0]

        # Find the result file, get the filename, and create desired target path
        result_paths = glob.glob(subdir_path + '/**/result.json', recursive=True)
        if len(result_paths) > 0:
            result_path = result_paths[0]
        else:
            print("WARNING: No result path found for {} directory".format(subdir))
            continue
        result_file = seed_name + "_" + os.path.split(result_path)[-1]
        target_path = os.path.join(results_dir, result_file)

        # Copy the result file to the base directory
        os.system("cp {} {}".format(result_path, target_path))
        print("Copied {} ---> {}".format(result_path, target_path))


def parse_arguments():
    """Argument parser function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--results_directory", type=str,
                        help="Where to parse results files")
    return parser.parse_args()


def main():
    """Runs the function above to pull results files using CLI inputs."""
    # Parse command-line arguments
    args = parse_arguments()

    # Run function above
    pull_results_files(args.results_directory)
    print("Finished copying results files.")


if __name__ == '__main__':
    main()
