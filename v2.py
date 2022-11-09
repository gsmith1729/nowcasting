import numpy as np
import pandas as pd
import re
import types
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

#import the data
df = pd.read_csv("final_data.csv")

# change the index to the date
df.index=df["date"]
df.index = pd.DatetimeIndex(df.index).to_period('M')
df.drop('date', inplace=True, axis=1)

#endog_vars=["RPI","W875RX1","DPCERA3M086SBEA","CMRMTSPLx.x","RETAILx","INDPRO.x","IPFPNSS","IPFINAL.x","IPCONGD.x","IPDCONGD.x","IPNCONGD.x","IPBUSEQ.x","IPMAT.x","IPMANSICS.x","IPB51222S.x","IPFUELS.x","CUMFNS.x","HWI","HWIURATIO","CLF16OV","CE16OV.x","UNRATE.x","A823RL1Q225SBEA","USGOVT.y","USTRADE.y","USWTRADE.y","CES9091000001"]
endog_vars=["RPI","W875RX1","DPCERA3M086SBEA","CMRMTSPLx.x","RETAILx","HWI","HWIURATIO","CLF16OV","CE16OV.x","USGOVT.y","USTRADE.y"]
# specify the variables on which the factor structure will depend
endog=df.loc["01/01/1995":"01/01/2018", endog_vars]

# fit the factor model
mod=sm.tsa.DynamicFactor(endog, k_factors=5, factor_order=1, error_orders=1)
initial_res=mod.fit(method="powell", disp=False, cov_type='none')
res=mod.fit(initial_res.params,disp=False)

#print some stuff
print(res.summary(separate_params=False))

# regress gdp on the factors
factors=res.factors.filtered
gdp=df.loc["01/01/1995":"01/01/2018", "GDPC1"]
fs=[]
b=~np.isnan(gdp.values)
for factor in factors:
    fs.append(factor[b])
gdpf=gdp.values[b]
fs=np.asarray(fs)
ols=sm.regression.linear_model.OLS(np.transpose(gdpf),np.transpose(fs))
result=ols.fit()
# produce some graphs to show off our great nowcasting skill
plt.plot(result.params[0]*fs[0]+result.params[1]*fs[1]+result.params[2]*fs[2])
plt.plot(gdpf)
plt.show()


