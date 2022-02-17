# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from __future__ import annotations
from typing import NamedTuple, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..compute import ShapeletsArray, array as scarray, DataTypeLike
import numpy as np
import pathlib
import os


class Dataset(NamedTuple):
    file: str
    description: str
    data_frequency: str
    source: str


datasets = {
    'solar_forecast': Dataset('entoe_2016_es_solar_forecast.gz',
                              'Spanish solar energy production forecast for 2016 in 1h periods',
                              '1h', 'https://transparency.entsoe.eu/'),

    'wind_forecast': Dataset('entoe_2016_es_wind_forecast.gs',
                             'Spanish wind energy production forecast for 2016 in 1h periods',
                             '1h', 'https://transparency.entsoe.eu/'),

    'load': Dataset('entoe_2016_es_load.gz',
                    'Spanish energy load for 2016 in 1h periods',
                    '1h', 'https://transparency.entsoe.eu/'),

    'load_forecast': Dataset('entoe_2016_es_load_forecast.gz',
                             'Spanish energy load forecast for 2016 in 1h periods',
                             '1h', 'https://transparency.entsoe.eu/'),

    'day_ahead_prices': Dataset('entoe_2016_es_day_ahead_prices.gz',
                                'Spanish energy day ahead prices for 2016 in 1h periods',
                                '1h', 'https://transparency.entsoe.eu/'),

    'italian_power_demand': Dataset('italian_power_demand.gz',
                                    'Power demand by a small italian city in 1h intervals during 3 years, beginning on Jan 1 st 1995.',
                                    '1h', 'https://www.cs.ucr.edu/~eamonn/Time_Series_Snippets_10pages.pdf'),
}

DSKeys = Literal['solar_forecast', 'wind_forecast', 'load',
                 'load_forecast', 'day_ahead_prices',
                 'italian_power_demand']


def dataset_info(ds: DSKeys) -> Optional[Dataset]:
    found = ds in datasets
    if found:
        return datasets[ds]
    return None


def load_dataset(ds: Union[DSKeys, str], dtype: DataTypeLike = "float32") -> ShapeletsArray:
    current_path = pathlib.Path(__file__).parent.absolute()

    if ds in datasets:
        file_name = datasets[ds].file
    else:
        file_name = str(ds)

    file = os.path.join(current_path, file_name)
    if not file.endswith('.gz'):
        file += '.gz'

    if not os.path.exists(file):
        raise FileNotFoundError(f'Unable to find file {ds}')

    nparray = np.loadtxt(file)
    return scarray(nparray, dtype=dtype)


def load_mat(file_name: str, dtype: DataTypeLike = "float32") -> ShapeletsArray:
    # TODO this is needs testing
    current_path = pathlib.Path(__file__).parent.absolute()
    file = os.path.join(current_path, file_name)
    if not os.path.exists(file):
        raise FileNotFoundError(f'Unable to find file {ds}')
    nparray = np.loadtxt(file)
    return scarray(nparray, dtype=dtype)


__all__ = ['load_dataset', 'load_mat', 'dataset_info', 'DSKeys', 'Dataset']
