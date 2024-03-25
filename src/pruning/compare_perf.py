import pandas as pd

def compare_performances(path_normal, path_pruned):
    df1 = pd.read_csv(f"{path_normal}\\chexpert\\outputs.csv")
    df2 = pd.read_csv(f"{path_pruned}\\chexpert\\outputs.csv")
    # Concatenate based on 'filename'
    result = pd.merge(df1, df2, on='filename', how='outer')
