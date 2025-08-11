import pandas as pd

def preprocess_transactions(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    return df
