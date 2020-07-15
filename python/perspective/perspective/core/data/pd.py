################################################################################
#
# Copyright (c) 2019, the Perspective Authors.
#
# This file is part of the Perspective library, distributed under the terms of
# the Apache License 2.0.  The full license can be found in the LICENSE file.
#

import re
import numpy as np
import pandas as pd

LEVEL_REGEX = re.compile(r'level_([0-9]\d*)')


def _parse_datetime_index(index):
    '''Given an instance of `pandas.DatetimeIndex`, parse its `freq` and
    return a `numpy.dtype` that corresponds to the unit it should be parsed in.

    Because `pandas.DataFrame`s cannot store datetimes in anything other than
    `datetime64[ns]`, we need to examine the `DatetimeIndex` itself to
    understand what unit it needs to be parsed as.

    Args:
        index (pandas.DatetimeIndex)

    Returns:
        `numpy.dtype`: a datetime64 dtype with the correct units depending on
            `index.freq`.
    '''
    if index.freq is None:
        return np.dtype("datetime64[ns]")

    freq = str(index.freq).lower()
    new_type = None

    if any(s in freq for s in ["businessday", "day"]) or freq == "sm" or freq == "sms":
        # days
        new_type = "D"
    elif freq == "w" or "week" in freq:
        # weeks
        new_type = "W"
    elif any(s in freq for s in ["month", "quarter"]):
        # months
        new_type = "M"
    elif "year" in freq or freq == "a":
        new_type = "Y"
    else:
        # default to datetime
        new_type = "ns"

    return np.dtype("datetime64[{0}]".format(new_type))


def deconstruct_pandas(data):
    '''Given a dataframe, flatten it by resetting the index and memoizing the
    pivots that were applied.

    Args:
        data (pandas.dataframe): a Pandas DataFrame to parse

    Returns:
        (pandas.DataFrame, dict): a Pandas DataFrame and a dictionary containing
            optional members `columns`, `row_pivots`, and `column_pivots`.
    '''
    kwargs = {}

    # handle series first
    if isinstance(data, pd.Series):
        flattened = data.reset_index()

        if isinstance(data, pd.Series):
            # preserve name from series
            flattened.name = data.name

            # make sure all columns are strings
            flattened.columns = [str(c) for c in flattened.columns]
        return flattened, kwargs

    # Decompose Period index to timestamps
    if isinstance(data.index, pd.PeriodIndex):
        data.index = data.index.to_timestamp()

    # collect original index names
    if isinstance(data.index, pd.MultiIndex):
        index_names_orig = data.index.names
    else:
        if data.index.name is None:
            data.index.name = 'index'
        index_names_orig = [data.index.name]

    # collect original column names
    columns_names_orig = data.columns.names if isinstance(data.columns, pd.MultiIndex) else data.columns

    if isinstance(data.index, pd.MultiIndex):
        # row multiindex turns into row pivots
        data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        # column multiindex turns into column_pivots
        s = data.unstack()
        s.name = ' '
        data = s.reset_index()

        data.columns = [str(_) for _ in data.columns]
        column_pivots = list(_ for _ in columns_names_orig if _ is not None)

        kwargs['column_pivots'] = [_ for _ in data.columns if LEVEL_REGEX.match(_)] + column_pivots
        kwargs['row_pivots'] = index_names_orig
        kwargs['columns'] = [' ']

    return data, kwargs
