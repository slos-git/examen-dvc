
import logging
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import joblib
import numpy as np

processed_data_relative_path = "./data/processed_data"
models_relative_path = "./models"


def main(processed_data_rel_path=processed_data_relative_path,
         models_rel_path=models_relative_path):
    """
    Train model with best parameters and save model to file.
    """
    logger = logging.getLogger(__name__)
    logger.info('Training model with best parameters')

    print("joblib version : ", joblib.__version__)

    # Import datasets
    X_train = pd.read_csv(f"{processed_data_rel_path}/X_train_scaled.csv",
                          sep=",")

    y_train = pd.read_csv(f"{processed_data_rel_path}/y_train.csv", sep=",")

    y_train = np.ravel(y_train)

    # Get model best parameters
    params_filename = f"{models_rel_path}/parameters.pkl"
    best_params = pickle.load(open(params_filename, 'rb'))

    rf_regressor = RandomForestRegressor(n_jobs=-1)

    rf_regressor.set_params(**best_params)

    # Train the model
    rf_regressor.fit(X_train, y_train)

    # Save the trained model to a file
    model_filename = f"{models_rel_path}/trained_model.joblib"
    joblib.dump(rf_regressor, model_filename)
    print("Model trained and saved successfully.")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
