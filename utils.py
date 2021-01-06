import numpy as np
import dask.dataframe as dd

def train_test_split_by_time(df, timestamp, user):
    df['train_mask'] = np.ones((len(df),), dtype=np.bool)
    df['test_mask'] = np.zeros((len(df),), dtype=np.bool)
    #df = dd.from_pandas(df, npartitions=10)
    def train_test_split(df):
        df = df.sort_values([timestamp])
        if df.shape[0] == 1:
            df.iloc[-1, -1] = True
        if df.shape[0] >= 2:
            df.iloc[-1, -2] = False
            df.iloc[-1, -1] = True
        return df
    df = df.groupby(user, group_keys=False).apply(train_test_split).compute(scheduler='processes').sort_index()
    print(df[df[user] == df[user].unique()[0]].sort_values(timestamp))
    return df['train_mask'].to_numpy().nonzero()[0], \
           df['val_mask'].to_numpy().nonzero()[0], \
           df['test_mask'].to_numpy().nonzero()[0]