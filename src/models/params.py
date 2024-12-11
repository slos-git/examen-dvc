
import logging
import pickle

import pandas as pd

from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor

processed_data_relative_path = "./data/processed_data"
models_relative_path = "./models"


def main(processed_data_rel_path=processed_data_relative_path,
         models_rel_path=models_relative_path):
    """
    Find best parameters and save to file.
    """
    logger = logging.getLogger(__name__)
    logger.info('Finding best parameters')

    # Import datasets
    X_train = pd.read_csv(f"{processed_data_rel_path}/X_train_scaled.csv",
                          sep=",")

    X_test = pd.read_csv(f"{processed_data_rel_path}/X_test_scaled.csv",
                         sep=",")

    y_train = pd.read_csv(f"{processed_data_rel_path}/y_train.csv", sep=",")
    y_test = pd.read_csv(f"{processed_data_rel_path}/y_test.csv", sep=",")

    # Define parameter grid
    parameters_rf = {
        'n_estimators': [100, 200, 300],   # , 500, 1000
        'max_depth': [None],               # , 10, 80, 90, 100, 110
        'min_samples_split': [2],          # , 5, 10
        'min_samples_leaf': [1, 2, 4, 8],  # , 16
    }

    # Perform grid search
    grid_rf = model_selection.GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=parameters_rf,
        # scoring='neg_mean_squared_error',
        cv=5, n_jobs=-1, verbose=2)

    print('grid_rf.fit...')
    grid_rf = grid_rf.fit(X_train.values, y_train.values.ravel())

    # Report best score and parameters
    print(f"Best score: {grid_rf.best_score_:.3f}")
    print(grid_rf.best_params_)

    results = pd.DataFrame(grid_rf.cv_results_)[['params',
                                                 'mean_test_score',
                                                 'std_test_score']]
    print(results)

    # Evaluate on test set
    best_model = grid_rf.best_estimator_
    test_score = best_model.score(X_test.values, y_test.values.ravel())
    print(f"Test set R^2 score: {test_score:.3f}")

    # Save model best parameters
    filename = f"{models_rel_path}/parameters.pkl"
    pickle.dump(grid_rf.best_params_, open(filename, 'wb'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
