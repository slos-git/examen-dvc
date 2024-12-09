
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

raw_data_relative_path = "./data/raw_data"
processed_data_relative_path = "./data/processed_data"


def main(raw_data_relative_path=raw_data_relative_path,
         processed_data_relative_path=processed_data_relative_path):
    """
    Split dataset into train and test sets and save to files.
    """
    logger = logging.getLogger(__name__)
    logger.info('Splitting raw dataset')

    # Import dataset
    df_raw = pd.read_csv(f"{raw_data_relative_path}/raw.csv", sep=",")

    # Split data into training and testing sets
    target = df_raw['silica_concentrate']
    features = df_raw.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = \
        train_test_split(features, target,
                         test_size=0.3, random_state=42)

    # Save dataframes to their respective output file paths
    for set, filename in zip([X_train, X_test, y_train, y_test],
                             ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(processed_data_relative_path,
                                       f'{filename}.csv')
        set.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
