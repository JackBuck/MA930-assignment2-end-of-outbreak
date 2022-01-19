from datetime import timedelta

import numpy as np


def extract_infection_times(obs_df):
    """
    Extract an array of infection times (with repeats) from an observations dataframe

    Args:
        obs_df: An observations dataframe with columns for 'time' and 'num_cases'

    Returns:
        (first_date, last_possible_offset, offsets).
            `first_date` is the time of the first observation in the dataframe (not
                necessarily the date of the first infection).
            `last_possible_offset` is the offset (in days) from the first_date
                corresponding to the last date in the observations dataframe
            `offsets` is the numpy array of offsets (in days) of each infection (with
                repeats) from the first_date.
    """
    first_date = obs_df["time"].min()
    last_date = obs_df["time"].max()

    offsets = []
    for row in obs_df.itertuples(index=False):
        for _ in range(row.num_cases):
            offsets.append(int((row.time - first_date) / timedelta(days=1)))

    offsets = sorted(offsets)
    offsets = np.asarray(offsets)

    last_possible_offset = int((last_date - first_date) / timedelta(days=1))

    return first_date, last_possible_offset, offsets
