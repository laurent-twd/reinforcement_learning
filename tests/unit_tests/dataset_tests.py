from src.data.trading_dataset import TradingDataset
import pandas as pd


def get_dummy_dataset():

    M = 100
    path_to_data = "/Users/laurentthanwerdas/Downloads/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv"
    df = pd.read_csv(path_to_data).tail(M)
    columns = ["High", "Low", "Close", "Open"]
    df = df.dropna(subset=columns)
    prices = df[["High", "Low", "Close"]].values

    n_assets = 1
    n_channels = 3
    window = 30

    dataset = TradingDataset(
        df=prices, n_assets=n_assets, n_channels=n_channels, window=window, norm=None
    )
    return dataset


def test_dataset():

    dataset = get_dummy_dataset()
    assert dataset[52][2] == dataset[53][1]
    assert (dataset[62][0][1:] == dataset[63][0][:-1]).all()
    assert dataset[36][0][-1][-1] == dataset[36][1]
