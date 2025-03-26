import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
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
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=random_seed),
        'Random Forest': RandomForestRegressor(random_state=random_seed),
        'GBR' : GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }

    # Hyperparameter grids for RandomizedSearchCV
    parameter = {
        'depth': [5, 10, None],
        'split': [2, 5],
        'leaf': [1, 5]
    }


    for current_system in systems:
        datasets_location = 'datasets/{}'.format(current_system)  # Modify this to specify the location of the datasets

        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]  # List all CSV files in the directory

        for csv_file in csv_files:
            print('\n> System: {}, Dataset: {}, Training data fraction: {}, Number of repeats: {}'.format(current_system, csv_file, train_frac, num_repeats))

            data = pd.read_csv(os.path.join(datasets_location, csv_file))  # Load data from CSV file

            # Initialize a dictionary to store results for all models
            metrics = {model_name: {'MAPE': [], 'MAE': [], 'RMSE': []} for model_name in models}

            for current_repeat in range(num_repeats):  # Repeat the process n times
                # Randomly split data into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=random_seed*current_repeat)  # Change the random seed based on the current repeat
                test_data = data.drop(train_data.index)

                # Split features (X) and target (Y)
                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Iterate over all models
                for model_name, model in models.items():
                    # If the model has hyperparameter grid, apply GridSearchCV
                    if model_name in parameter:
                        grid_search = RandomizedSearchCV(model, param_distributions=parameter, n_iter=10, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2, random_state=1)
                        grid_search.fit(training_X, training_Y)  # Fit GridSearchCV model
                        model = grid_search.best_estimator_  # Get the best model from the grid search
                        print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
                    else:
                        model.fit(training_X, training_Y)  # Train the model if no grid search

                    predictions = model.predict(testing_X)  # Predict the testing data

                    # Calculate evaluation metrics for the current repeat
                    mape = mean_absolute_percentage_error(testing_Y, predictions)
                    mae = mean_absolute_error(testing_Y, predictions)
                    rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                    # Store the metrics for the current model
                    metrics[model_name]['MAPE'].append(mape)
                    metrics[model_name]['MAE'].append(mae)
                    metrics[model_name]['RMSE'].append(rmse)

            # Calculate the average of the metrics for all repeats for each model
            print(f"\nResults for {csv_file}:")
            for model_name, model_metrics in metrics.items():
                avg_mape = np.mean(model_metrics['MAPE'])
                avg_mae = np.mean(model_metrics['MAE'])
                avg_rmse = np.mean(model_metrics['RMSE'])

                # Print results
                print(f"Model: {model_name} : Average MAPE: {avg_mape:.2f}, Average MAE: {avg_mae:.2f}, Average RMSE: {avg_rmse:.2f}")
                all_results.append({
                    'System': current_system,
                    'Dataset': csv_file,
                    'Model': model_name,
                    'Avg_MAPE': avg_mape,
                    'Avg_MAE': avg_mae,
                    'Avg_RMSE': avg_rmse
                })

    # Save all results to a single CSV file
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)
    all_results_df = pd.DataFrame(all_results)
    output_path = os.path.join(output_folder, 'all_model_results.csv')
    all_results_df.to_csv(output_path, index=False)
    print(f"\n✅ All results saved to: {output_path}")

if __name__ == "__main__":
    main()
