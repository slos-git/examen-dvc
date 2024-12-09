
import logging
import pandas as pd

from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor


processed_data_relative_path = "./data/processed_data"


def main(processed_data_rel_path=processed_data_relative_path):
    """
    Find best parameters and save to file.
    """
    logger = logging.getLogger(__name__)
    logger.info('Normalizing datasets')

    # Import datasets
    X_train = pd.read_csv(f"{processed_data_rel_path}/X_train.csv", sep=",")
    X_train = X_train.drop(['date'], axis=1)

    X_test = pd.read_csv(f"{processed_data_rel_path}/X_test.csv", sep=",")
    X_test = X_test.drop(['date'], axis=1)

    y_train = pd.read_csv(f"{processed_data_rel_path}/y_train.csv", sep=",")
    y_test = pd.read_csv(f"{processed_data_rel_path}/y_test.csv", sep=",")

    # Define parameter grid
    parameters_rf = {
        'n_estimators': [100, 200, 300, 1000],
        'max_depth': [None, 80, 90, 100, 110],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
    }

    # Perform grid search
    grid_rf = model_selection.GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=parameters_rf,
        scoring='neg_mean_squared_error',
        cv=3, n_jobs=-1, verbose=2)

    grid_rf = grid_rf.fit(X_train.values, y_train.values.ravel())

    # Report best score and parameters
    print(f"Best score: {grid_rf.best_score_:.3f}")
    print(grid_rf.best_params_)
    # {'bootstrap': True, 'max_depth': 80, 'max_features': 3,
    #  'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 300}

    # {'bootstrap': True, 'max_depth': 80, 'max_features': 4,
    #  'min_samples_leaf': 3, 'min_samples_split': 12, 'n_estimators': 100}

    # y_pred = grid_rf.predict(X_test.values)

    results = pd.DataFrame(grid_rf.cv_results_)[['params',
                                                 'mean_test_score',
                                                 'std_test_score']]
    print(results)

    # Evaluate on test set
    best_model = grid_rf.best_estimator_
    test_score = best_model.score(X_test.values, y_test.values.ravel())
    print(f"Test set R^2 score: {test_score:.3f}")

    """
    rf = RandomForestRegressor(bootstrap=False, max_depth=100,
                               max_features=2, min_samples_leaf=6,
                               min_samples_split=12, n_estimators=300)

    rf.fit(X_train.values, y_train.values.ravel())
    y_pred = rf.predict(X_test.values)

    score_rf = rf.score(X_test.values, y_test.values.ravel())
    print(score_rf)
    """


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
