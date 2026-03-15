def split_data(data: str, train_ratio: float = 0.9, val_ratio: float = 0.1) -> tuple[str, str, str]:
    """
    Splits data (string, list, or tensor) into training, validation, and testing sets
    based on the provided ratios. The ratios must sum to 1.0.

    Args:
        data: The string data to be split.
        train_ratio: Float representing the proportion of data for training.
        val_ratio: Float representing the proportion of data for validation.

    Returns:
        tuple: (train_data, val_data, test_data)
    """

    total_len = len(data)

    # Calculate absolute split indices based on the ratios
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    # Slice the data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data
