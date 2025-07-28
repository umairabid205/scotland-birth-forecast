import pandas as pd

def load_birth_data(path="data/raw/births.csv"):
    return pd.read_csv(path, parse_dates=['date'])