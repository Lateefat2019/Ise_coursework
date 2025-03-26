import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

def main():
    """
    Parameters:
    systems (list): List of systems containing CSV datasets.
    num_repeats (int): Number of times to repeat the evaluation for avoiding stochastic bias.
    train_frac (float): Fraction of data to use for training.
    random_seed (int): Initial random seed to ensure the results are reproducible
    """

    # Specify the parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 3  # Modify this value to change the number of repetitions
    train_frac = 0.7  # Modify this value to change the training data fraction (e.g., 0.7 for 70%)
    random_seed = 1 # The random seed will be altered for each repeat

    # Initialize models for comparison
    models = {'Random Forest': RandomForestRegressor(random_state=random_seed)}


    for current_system in systems:
        datasets_location = 'datasets/{}'.format(current_system) # Modify this to specify the location of the datasets

        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')] # List all CSV files in the directory

        for csv_file in csv_files:
            print('\n> System: {}, Dataset: {}, Training data fraction: {}, Number of repeats: {}'.format(current_system, csv_file, train_frac, num_repeats))

            data = pd.read_csv(os.path.join(datasets_location, csv_file)) # Load data from CSV file

            # Initialize a dictionary to store results for all models
            metrics = {model_name: {'MAPE': [], 'MAE': [], 'RMSE': []} for model_name in models}

            for current_repeat in range(num_repeats): # Repeat the process n times
                # Randomly split data into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=random_seed*current_repeat) # Change the random seed based on the current repeat
                test_data = data.drop(train_data.index)

                # Split features (X) and target (Y)
                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Iterate over all models
                for model_name, model in models.items():

                    model.fit(training_X, training_Y) # Train the model with the training data

                    predictions = model.predict(testing_X) # Predict the testing data

                    # Calculate evaluation metrics for the current repeat
                    mape = mean_absolute_percentage_error(testing_Y, predictions)
                    mae = mean_absolute_error(testing_Y, predictions)
                    rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                    # Store the metrics for the current model
                    metrics[model_name]['MAPE'].append(mape)
                    metrics[model_name]['MAE'].append(mae)
                    metrics[model_name]['RMSE'].append(rmse)

            # Calculate the average of the metrics for all repeats for each model
            for model_name, model_metrics in metrics.items():
                print(f"Model: {model_name} : Average MAPE: {np.mean(model_metrics['MAPE']):.2f}, Average MAE: {np.mean(model_metrics['MAE']):.2f}, Average RMSE: {np.mean(model_metrics['RMSE']):.2f}")


if __name__ == "__main__":
    main()