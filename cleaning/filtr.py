import pandas as pd
import numpy as np

def filter_date(master_df, cutoff_hi=180, cutoff_low=0):
    """
    This function filter rows of master_df based of 
    difference between the intake survey and the start 
    of the session. The hi cutoff is the max number of 
    days before the session, and low is the bound that 
    limits the other side. Nonnegative values are up to
    the first appointment, negative is after.
    """
    master_df_ = master_df.copy()
    rows = master_df.shape[0]
    mask_1 = master_df_['DateDifference'] < cutoff_hi
    mask_2 = master_df_['DateDifference'] >= cutoff_low
    master_df_ = master_df_.loc[ mask_1 & mask_2, :]

    print(f'Survey not within {cutoff_hi} and {cutoff_low} days before', rows - master_df_.shape[0])
    print(f"Number of clients: {master_df['PatientID'].nunique()}")
    return master_df_

