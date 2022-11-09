import numpy as np
import pandas as pd
import re
import types
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("final_data.csv")

df.index = pd.PeriodIndex(df.sasdate.tolist(), freq='M')
df.drop('sasdate', axis=1, inplace=True)

factors = 5
# Construct the variable => list of factors dictionary
#factors = {row['description']: ['Global', row['group']]
#           for ix, row in groups.iterrows()}

# Check that we have the desired output for "Real personal income"
#print(factors['Real Personal Income'])

factor_multiplicities = {'Global': 2}

factor_orders = {'Global': 4}

# Get the baseline monthly and quarterly datasets
start = '2000'
endog_m = dta['2020-02'].dta_m.loc[start:, :]
gdp_description = defn_q.loc['GDPC1', 'description']
endog_q = dta['2020-02'].dta_q.loc[start:, [gdp_description]]

# Construct the dynamic factor model
model = sm.tsa.DynamicFactorMQ(
    endog_m, endog_quarterly=endog_q,
    factors=factors, factor_orders=factor_orders,
    factor_multiplicities=factor_multiplicities)
