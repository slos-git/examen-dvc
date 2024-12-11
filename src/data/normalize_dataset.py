
import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler


processed_data_relative_path = "./data/processed_data"


def main(processed_data_rel_path=processed_data_relative_path):
    """
    Normalize train and test datasets and save to files.
    """
    logger = logging.getLogger(__name__)
    logger.info('Normalizing datasets')

    # Import datasets
    X_train = pd.read_csv(f"{processed_data_rel_path}/X_train.csv", sep=",")
    X_test = pd.read_csv(f"{processed_data_rel_path}/X_test.csv", sep=",")

    # Normalize
    scaler = StandardScaler()

    X_train.drop(['date'], axis=1, inplace=True)
    X_columns = X_train.columns

    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train),
                                  columns=X_columns)

    X_test.drop(['date'], axis=1, inplace=True)

    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 columns=X_columns)

    # Save dataframes to their respective output file paths
    for set, filename in zip([X_train_scaled, X_test_scaled],
                             ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(processed_data_relative_path,
                                       f'{filename}.csv')
        set.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
