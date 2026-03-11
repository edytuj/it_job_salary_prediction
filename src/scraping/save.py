import pandas as pd


def save_to_csv(data, path):

    df = pd.DataFrame(data)

    df.to_csv(path, index=False)
