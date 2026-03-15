from learning_llms_from_first_principles.utils.data_utils import split_data


def test_split_data_string_default() -> None:
    data = "0123456789"
    train, val, test = split_data(data)
    assert train == "012345678"
    assert val == "9"
    assert test == ""


def test_split_data_string_custom_ratios() -> None:
    data = "0123456789"
    train, val, test = split_data(data, train_ratio=0.5, val_ratio=0.3)
    assert train == "01234"
    assert val == "567"
    assert test == "89"


def test_split_data_string_zero_test_ratio() -> None:
    data = "0123456789"
    train, val, test = split_data(data, train_ratio=0.8, val_ratio=0.2)
    assert train == "01234567"
    assert val == "89"
    assert test == ""


def test_split_data_empty() -> None:
    data = ""
    train, val, test = split_data(data)
    assert train == ""
    assert val == ""
    assert test == ""
