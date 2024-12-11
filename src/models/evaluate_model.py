
import logging

import pandas as pd
import numpy as np
from joblib import load
import json

from sklearn.metrics import mean_squared_error, r2_score

processed_data_relative_path = "./data/processed_data"
models_relative_path = "./models"
metrics_relative_path = "./metrics"


def main(processed_data_rel_path=processed_data_relative_path,
         models_rel_path=models_relative_path,
         metrics_rel_path=metrics_relative_path):
    """
    Evaluate model.
    """
    logger = logging.getLogger(__name__)
    logger.info('Evaluating model with best parameters')

    # Import datasets
    X_test = pd.read_csv(f"{processed_data_rel_path}/X_test_scaled.csv",
                         sep=",")

    y_test = pd.read_csv(f"{processed_data_rel_path}/y_test.csv", sep=",")

    y_test = np.ravel(y_test)

    # Load the trained model from file
    model_filename = f"{models_rel_path}/trained_model.joblib"
    model = load(model_filename)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save metrics
    metrics = {"mse": mse, "r2": r2}
    metrics_filename = f"{metrics_rel_path}/metrics.json"
    json.dump(metrics, open(metrics_filename, 'w'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
