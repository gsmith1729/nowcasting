import numpy as np
import pandas as pd
import re
import types
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

#import the data
df = pd.read_csv("final_data.csv",dayfirst=True,index_col='date', parse_dates = True)
start="01/01/1980"
endsample="01/12/2016"
end="01/01/2021"
# change the index to the date
df.index = pd.DatetimeIndex(df.index).to_period('M')


#endog_vars=["RPI","W875RX1","DPCERA3M086SBEA","CMRMTSPLx.x","RETAILx","INDPRO.x","IPFPNSS","IPFINAL.x","IPCONGD.x","IPDCONGD.x","IPNCONGD.x","IPBUSEQ.x","IPMAT.x","IPMANSICS.x","IPB51222S.x","IPFUELS.x","CUMFNS.x","HWI","HWIURATIO","CLF16OV","CE16OV.x","UNRATE.x","A823RL1Q225SBEA","USGOVT.y","USTRADE.y","USWTRADE.y","CES9091000001"]
endog_vars=["RPI","W875RX1","DPCERA3M086SBEA","CMRMTSPLx.x","RETAILx","IPDMAT.x","CLF16OV","CE16OV.x","USGOVT.y","USTRADE.y","GDPC1in"]
# specify the variables on which the factor structure will depend
endog=df.loc[start:end, endog_vars]

# fit the factor model
mod=sm.tsa.DynamicFactorMQ(endog, k_endog_monthly=8, factors=2, factor_multiplicities=2, factor_orders=1, idiosyncratic_ar1=False)#, standardization=False)
res=mod.fit()

#print some stuff
print(res.summary())

# plot to compare the in sample performance
def seriesplot(series):
    plt.plot(res.predict()[series].values)
    plt.plot(df.loc[start:end, series].values,markersize = 5, marker="o")
    plt.show()
#seriesplot("GDPC1in")
plt.plot(res.predict()["GDPC1in"].values)
plt.plot(df.loc[start:end, "GDPC1in"].values,markersize = 5, marker="o")
plt.plot(df.loc[start:end, "GDPC1out"].values,markersize = 5, marker="o")
plt.show()
