#! /usr/bin/env python3
# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

# %%
from entsoe import EntsoePandasClient, Area
import pandas as pd
import numpy as np
import os

client = EntsoePandasClient(api_key='7f316466-ba56-4680-b178-d9138aec6d16')
base_dir = '/Users/justo.ruiz/Development/shapelets/solo_comprobacion/modules/shapelets/data'

# %%
year = '2016'
country = 'es'
params = {
    'country_code': Area.ES,
    'start': pd.Timestamp(f'{year}0101', tz='Europe/Brussels'),
    'end': pd.Timestamp(f'{year}1231', tz='Europe/Brussels'),
}


def save(query: str, data):
    path = os.path.join(base_dir, f'entoe_{year}_{country}_{query}.gz')
    np.savetxt(path, data)


# %%
day_ahead_prices = client.query_day_ahead_prices(**params)
save('day_ahead_prices', day_ahead_prices)

# %%
load = client.query_load(**params)
save('load', load)

# %%
load_forecast = client.query_load_forecast(**params)
save('load_forecast', load_forecast)

# %%
wind_solar_forecast = client.query_wind_and_solar_forecast(**params)
save('solar_forecast', wind_solar_forecast.to_numpy()[:, 0])
save('wind_forecast', wind_solar_forecast.to_numpy()[:, 1])

# %%
client.query_pri
