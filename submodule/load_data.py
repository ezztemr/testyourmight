import pandas as pd

def load_data():
    df = pd.read_csv("data/heart.csv")
    df.head().T
    
    return df
