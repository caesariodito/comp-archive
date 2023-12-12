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
    df["Transmission_Type"] = df["Transmission"].str.extract(r"(\D+)")

    # Apply the mapping to the 'Transmission_Type' column
    df["Transmission_Type"] = df["Transmission_Type"].map(transmission_mapping)

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
