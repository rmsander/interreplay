"""Scripts for evaluating Gaussian Process Regression training on
optimization test functions, such as the Ackley or Rastrigin test functions,
 from the BoTorch library.

To adjust the parameters to observe the effect of modifying the batch size,
number of training samples, or dimensions/distribution of the features/targets,
please adjust the parameters at the top of the main function.
 """
# External Python packages
import torch
from botorch.test_functions.synthetic import Branin, Rosenbrock, Ackley, \
    Bukin, Cosine8, EggHolder, Michalewicz, Rastrigin, SixHumpCamel, \
    ThreeHumpCamel

# Native Python packages
import warnings
warnings.filterwarnings("ignore")

# Custom Python packages/modules
from utils.gpytorch.gpytorch_utils import standardize
from utils.execution.execution_utils import set_seeds
from analytic_evaluation import \
    generate_mean_and_std, create_function_data, evaluate_model, \
    plot_evaluation, extract_lengthscales


def main(test_function, function_name):

    # Display function name
    print("TEST FUNCTION: {}".format(function_name))

    # Set parameters
    SEED = 101  # RNG - need only be an integer > 0
    TWOD = True  # Whether to run test in 2D
    STANDARDIZE = False  # Whether to Z-score standardize features/targets
    set_seeds(SEED)  # Utility function for reproducibility
    N = 2500  # Number of samples in dataset
    EPOCHS = 200  # Duration of training

    # Set other parameters according to provided parameter configuration
    if TWOD:
        # Set dimensions for x and y
        Dx = 2  # Dimensions for x (features)
        Dy = 1  # Dimensions for y (targets)

        # Get means and std of distributions
        mean = torch.zeros(Dx)
        std = torch.ones(Dx)
    else:
        # Set dimensions for x and y
        Dx = 5  # Dimensions for x (features)
        Dy = 10  # Dimensions for y (targets)

        # Get means and std of distributions
        mean, std = generate_mean_and_std(Dx)

    # Create training data
    X, Y = create_function_data((N, Dx), mean, std, fn=test_function, Dy=Dy)

    # Determine if targets should be standardized
    if STANDARDIZE:
        Y_norm, Y_std, Y_mu = standardize(Y)
    else:
        Y_norm, Y_std, Y_mu = Y, None, None

    # Create and train model
    model = create_model(X, Y_norm, epochs=EPOCHS)

    # Evaluate model on generated test data
    X_test, Y_test, preds = evaluate_model(model, (N, Dx), Dy, Y_std, Y_mu, std,
                                           mean, standardize=STANDARDIZE, f=f)
    plot_evaluation(X_test, Y_test, preds, n=n)

    # Extract hyperparameters
    l = extract_lengthscales(model)
    print("___________________________________________________________________")
    print("LENGTHSCALES: {}".format(l))
    print("___________________________________________________________________")


if __name__ == '__main__':

    # Create arrays for test functions and names
    test_functions = [Ackley, Branin, Rosenbrock, Bukin, Cosine8, EggHolder,
                     Michalewicz, Rastrigin, SixHumpCamel, ThreeHumpCamel]
    test_function_names = ["Ackley", "Branin", "Rosenbrock", "Bukin", "Cosine8",
                           "EggHolder", "Michalewicz", "Rastrigin",
                           "SixHumpCamel", "ThreeHumpCamel"]

    # Loop over test functions, and run function approximation with GPR
    for f, n in zip(test_functions, test_function_names):
        main(f, n)
