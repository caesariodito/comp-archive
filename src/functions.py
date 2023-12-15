import pandas as pd
import prince

from sklearn.preprocessing import OneHotEncoder
from category_encoders import BinaryEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    PowerTransformer,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)


def process_transmission(df):
    # Create a mapping for transmission types
    transmission_mapping = {
        "A": "Automatic",
        "AM": "Automatic",
        "AS": "Automatic",
        "AV": "Automatic",
        "M": "Manual",
    }

    # Extract the transmission type using a regular expression
    df["Transmission Type"] = df["Transmission"].str.extract(r"(\D+)")

    # Apply the mapping to the 'Transmission_Type' column
    df["Transmission Type"] = df["Transmission Type"].map(transmission_mapping)
    # df["Transmission Type General"] = df["Transmission Type"].map(transmission_mapping)

    # Extract the number of gears
    df["Gears"] = df["Transmission"].str.extract("(\d+)").astype("Int64")

    return df


def group_vehicle_classes(df):
    # Create a mapping for vehicle classes
    vehicle_class_mapping = {
        "SUV - SMALL": "SUV",
        "SUV - STANDARD": "SUV",
        "PICKUP TRUCK - STANDARD": "PICKUP TRUCK",
        "PICKUP TRUCK - SMALL": "PICKUP TRUCK",
        "STATION WAGON - SMALL": "STATION WAGON",
        "STATION WAGON - MID-SIZE": "STATION WAGON",
        "VAN - CARGO": "VAN",
        "VAN - PASSENGER": "VAN",
    }

    # Apply the mapping to the 'Vehicle Class' column
    df["Vehicle Class General"] = df["Vehicle Class"].replace(vehicle_class_mapping)

    return df


def group_vehicle_types(df):
    # Create a mapping for vehicle types
    vehicle_type_mapping = {
        "SMALL": "Small",
        "MID-SIZE": "Mid-Size",
        "FULL-SIZE": "Full-Size",
        "PASSENGER": "Passenger",
        "CARGO": "Cargo",
        "STANDARD": "Standard",
    }

    # Create a new column 'Vehicle Type'
    df["Vehicle Type"] = df["Vehicle Class"]

    # Apply the mapping to the 'Vehicle Type' column
    for key, value in vehicle_type_mapping.items():
        mask = df["Vehicle Type"].notna() & df["Vehicle Type"].str.contains(key)
        df.loc[mask, "Vehicle Type"] = value.upper()

    return df


def encode_categorical_features(
    df_train,
    df_test,
    categorical_features_onehot,
    categorical_features_binary,
    categorical_features_ordinal,
):
    if (
        not categorical_features_onehot
        and not categorical_features_binary
        and not categorical_features_ordinal
    ):
        return df_train, df_test

    df_train_categorical_one_hot = df_train[categorical_features_onehot]
    df_train_categorical_binary = df_train[categorical_features_binary]
    df_train_categorical_ordinal = df_train[categorical_features_ordinal]

    df_test_categorical_one_hot = df_test[categorical_features_onehot]
    df_test_categorical_binary = df_test[categorical_features_binary]
    df_test_categorical_ordinal = df_test[categorical_features_ordinal]

    encoder_onehot = OneHotEncoder(sparse_output=False)
    train_onehot_encoded_data = encoder_onehot.fit_transform(
        df_train_categorical_one_hot
    )
    test_onehot_encoded_data = encoder_onehot.transform(df_test_categorical_one_hot)

    # Convert numpy arrays to pandas DataFrames
    train_onehot_encoded_data = pd.DataFrame(
        train_onehot_encoded_data,
        columns=encoder_onehot.get_feature_names_out(categorical_features_onehot),
        index=df_train.index,
    )
    test_onehot_encoded_data = pd.DataFrame(
        test_onehot_encoded_data,
        columns=encoder_onehot.get_feature_names_out(categorical_features_onehot),
        index=df_test.index,
    )

    encoder_binary = BinaryEncoder(cols=categorical_features_binary)
    train_df_binary = encoder_binary.fit_transform(df_train_categorical_binary)
    test_df_binary = encoder_binary.transform(df_test_categorical_binary)

    encoder_ordinal = OrdinalEncoder(cols=categorical_features_ordinal)
    train_df_ordinal = encoder_ordinal.fit_transform(df_train_categorical_ordinal)
    test_df_ordinal = encoder_ordinal.transform(df_test_categorical_ordinal)

    # Merge the one-hot, binary and ordinal encoded dataframes with the original dataframes
    df_train = pd.concat(
        [
            df_train.drop(
                categorical_features_onehot
                + categorical_features_binary
                + categorical_features_ordinal,
                axis=1,
            ),
            train_onehot_encoded_data,
            train_df_binary,
            train_df_ordinal,
        ],
        axis=1,
    )
    df_test = pd.concat(
        [
            df_test.drop(
                categorical_features_onehot
                + categorical_features_binary
                + categorical_features_ordinal,
                axis=1,
            ),
            test_onehot_encoded_data,
            test_df_binary,
            test_df_ordinal,
        ],
        axis=1,
    )

    return df_train, df_test


def perform_mca_dataframe(
    df_train, df_test, mca_columns, n_components=2, n_iter=3, random_state=42
):
    """
    Perform Multiple Correspondence Analysis (MCA) on training and test DataFrames.

    Parameters:
    df_train (pd.DataFrame): The training DataFrame.
    df_test (pd.DataFrame): The test DataFrame.
    mca_columns (list): The columns to perform MCA on.
    n_components (int): The number of components to keep.
    n_iter (int): The number of iterations for the power method.
    random_state (int): The seed of the pseudo random number generator to use when shuffling the data.

    Returns:
    df_train_transformed (pd.DataFrame): The transformed training DataFrame.
    df_test_transformed (pd.DataFrame): The transformed test DataFrame.
    """
    mca = prince.MCA(
        n_components=n_components, n_iter=n_iter, random_state=random_state
    )
    mca = mca.fit(df_train[mca_columns])

    df_train_transformed = df_train.copy()
    df_train_transformed[mca_columns] = mca.transform(df_train[mca_columns])

    df_test_transformed = df_test.copy()
    df_test_transformed[mca_columns] = mca.transform(df_test[mca_columns])

    return df_train_transformed, df_test_transformed


def perform_pca_dataframe(
    df_train, df_test, pca_columns, n_components=2, random_state=42
):
    """
    Perform Principal Component Analysis (PCA) on training and test DataFrames.

    Parameters:
    df_train (pd.DataFrame): The training DataFrame.
    df_test (pd.DataFrame): The test DataFrame.
    pca_columns (list): The columns to perform PCA on.
    n_components (int): The number of components to keep.
    random_state (int): The seed of the pseudo random number generator to use when shuffling the data.

    Returns:
    df_train_transformed (pd.DataFrame): The transformed training DataFrame.
    df_test_transformed (pd.DataFrame): The transformed test DataFrame.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(df_train[pca_columns])

    df_train_transformed = df_train.copy()
    df_train_transformed[pca_columns] = pca.transform(df_train[pca_columns])

    df_test_transformed = df_test.copy()
    df_test_transformed[pca_columns] = pca.transform(df_test[pca_columns])

    return df_train_transformed, df_test_transformed


def power_transform_dataframe(df_train, df_test, columns_to_transform):
    transformer = PowerTransformer()

    transformed_data_train = transformer.fit_transform(df_train[columns_to_transform])
    transformed_df_train = pd.DataFrame(
        transformed_data_train, columns=columns_to_transform, index=df_train.index
    )
    df_train[columns_to_transform] = transformed_df_train

    transformed_data_test = transformer.transform(df_test[columns_to_transform])
    transformed_df_test = pd.DataFrame(
        transformed_data_test, columns=columns_to_transform, index=df_test.index
    )
    df_test[columns_to_transform] = transformed_df_test

    return df_train, df_test


def minmax_transform_dataframe(df_train, df_test, columns_to_transform):
    scaler = MinMaxScaler()

    transformed_data_train = scaler.fit_transform(df_train[columns_to_transform])
    transformed_df_train = pd.DataFrame(
        transformed_data_train, columns=columns_to_transform, index=df_train.index
    )
    df_train[columns_to_transform] = transformed_df_train

    transformed_data_test = scaler.transform(df_test[columns_to_transform])
    transformed_df_test = pd.DataFrame(
        transformed_data_test, columns=columns_to_transform, index=df_test.index
    )
    df_test[columns_to_transform] = transformed_df_test

    return df_train, df_test


def standard_scale_dataframe(df_train, df_test, columns_to_transform):
    scaler = StandardScaler()

    transformed_data_train = scaler.fit_transform(df_train[columns_to_transform])
    transformed_df_train = pd.DataFrame(
        transformed_data_train, columns=columns_to_transform, index=df_train.index
    )
    df_train[columns_to_transform] = transformed_df_train

    transformed_data_test = scaler.transform(df_test[columns_to_transform])
    transformed_df_test = pd.DataFrame(
        transformed_data_test, columns=columns_to_transform, index=df_test.index
    )
    df_test[columns_to_transform] = transformed_df_test

    return df_train, df_test


def robust_transform_dataframe(df_train, df_test, columns_to_transform):
    scaler = RobustScaler()

    transformed_data_train = scaler.fit_transform(df_train[columns_to_transform])
    transformed_df_train = pd.DataFrame(
        transformed_data_train, columns=columns_to_transform, index=df_train.index
    )
    df_train[columns_to_transform] = transformed_df_train

    transformed_data_test = scaler.transform(df_test[columns_to_transform])
    transformed_df_test = pd.DataFrame(
        transformed_data_test, columns=columns_to_transform, index=df_test.index
    )
    df_test[columns_to_transform] = transformed_df_test

    return df_train, df_test
