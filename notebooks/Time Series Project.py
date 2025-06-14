#!/usr/bin/env python
# coding: utf-8

# In[210]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox


# In[44]:


data=pd.read_csv('C:/Users/User/Downloads/G17MVSFTTRUCKS.csv',header=0,index_col=0,parse_dates=True)


# In[45]:


data.head()


# In[51]:


len(data)


# In[46]:


data.describe()


# In[ ]:


Time series plot


# In[85]:


data.plot()
plt.title('Regular Seasonal Factors: Total Truck Production')
plt.xlabel('Date', size = 12)
plt.ylabel('Truck Production', size  = 12)
plt.rc("figure", figsize=(10,8))


# In[ ]:


We can observe from the above plot that the data is stationary.

Next we shall perform the Augmented Dickey Fuller test to ensure our data is stationary or non stationary.


# In[86]:


from statsmodels.tsa.stattools import adfuller


# In[87]:


adfuller(data)


# In[ ]:


The Augmented Dickey fuller test has a P value of greater less than 0.05 which seems to suggest that the time series is stationary. so we will move ahead to next step to plot the acf and pacf and find the models


# In[89]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


# In[22]:


import pandas
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


#Test
autocorrelation_plot(data_q)
pyplot.show()


# In[331]:


plot_acf(data,lags=150)


# In[94]:


plot_pacf(data)


# In[96]:


from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
stepwise_fit=auto_arima(data['G17MVSFTTRUCKS'],trace=True,suppress_warnings=True)


# In[98]:


stepwise_fit.summary()


# In[412]:


model=ARIMA(data['G17MVSFTTRUCKS'],order=(4,0,4))
model=model.fit()
model.summary()


# In[99]:


print(data.shape)
train= data.iloc[:-30]
test= data.iloc[-30:]
print(train.shape,test.shape)


# In[ ]:


model=ARIMA(train['G17MVSFTTRUCKS'],order=(5,0,3))
model=model.fit()
model.summary()


# In[370]:


start=len(train)
end=len(train)+len(test)
predictions = model.predict(start=start, end=end, dynamic=False, typ='levels')


# In[102]:


pred.plot(legend=True)
test['G17MVSFTTRUCKS'].plot(legend=True)


# In[39]:


#Non-Seasonal Data


# In[227]:


data2=pd.read_csv('C:/Users/User/Downloads/Electric_Production.csv',header=0,parse_dates=True)
print(data2)


# In[423]:


data2=pd.read_csv('C:/Users/User/Downloads/Electric_Production.csv',header=0,index_col=0,parse_dates=True)
data2


# In[128]:


data2.dtypes=='object'
num_vars=data2.columns[data2.dtypes!='object']
data2[num_vars] 


# In[129]:


#Checking the null values in our dataset
data2[num_vars].isnull()


# In[130]:


data2.isnull().sum()


# In[131]:


data2.head()


# In[132]:


data2.describe()


# In[133]:


data2.info()


# In[ ]:


Time series plot


# In[424]:


y=data2.IPG2211A2N
y.plot()
plt.rc("figure", figsize=(10,8))
plt.title('Electricity Production')
plt.xlabel('Date', size = 12)
plt.ylabel('Electricity Production', size  = 12)
plt.rc("figure", figsize=(10,8))


# In[ ]:


Looking at the plot we can observe there is an **upward trend** over the period of time.


Stationarity Test

We can observe from the above plot that the Electric production is fairly seasonal with a upward trend.

Next we shall perform the Augmented Dickey Fuller test to see if the trend and level of the electric production is stationary or non stationary.


# In[ ]:


The augmented Dickey-Fuller (ADF) test statistic is the t-statistic of the estimated coefficient of a from the method of least squares regression. However, the ADF test statistic
is not approximately t-distributed under the null hypothesis; instead, it has a certain nonstandard large-sample distribution under the null hypothesis of a unit root.


# In[229]:


from statsmodels.tsa.stattools import adfuller


# In[293]:


data2=pd.read_csv('C:/Users/User/Downloads/Electric_Production.csv',header=0,index_col=0)

adfuller(data2)


# In[ ]:


The Augmented Dickey fuller test has a P value of greater than 0.05 which seems to suggest that the time series is non-stationary. The p-value is obtained is greater than significance level of 0.05 and the ADF statistic is higher than any of the critical values.

Clearly, there is no reason to reject the null hypothesis. So, the time series is in fact non-stationary.

We can clearly see a trend in data so let us perform some more formal test of stationarity.


# In[291]:


first_difference=adfuller(data2.IPG2211A2N.diff().dropna())
print(f'ADF Statistics:{result[0]}')
print(f'p-value:{result[1]}')


# In[ ]:


The p-value:2.9951614981156204e-09 is less than 0.05. Now the series looks stationary with a 1-order difference, so q will be 1 for our model. I am also testing another method two make data stationary. Converting Non-stationary data to Stationary


# In[236]:


#Method2 only for Test
#Differencing and then log transformation
#Making data Stationary using Differencing
from statsmodels.tsa.stattools import adfuller
y1=y-y.shift(1)
y1=y1[1:]
y2=np.log(y1-np.min(y1)+1)


# In[237]:


result=adfuller(y1)
print('ADF statistic: %f' %result[0])
print('p-value: %f' %result[1])
print('critical values:')

for key,value in result[4].items():
    print('\t%s: %.3f' % (key,value))


# In[238]:


y1.plot()


# In[ ]:


The data looks stationary now but we can also try First Log Transformation and then differencing


# In[239]:


from statsmodels.tsa.stattools import adfuller
y1=np.log(y)
y2=y1-y1.shift(1)
y2.dropna(inplace=True)


# In[240]:


result=adfuller(y2)

print('ADF statistic: %f' %result[0])
print('p-value: %f' %result[1])
print('critical values:')

for key,value in result[4].items():
    print('\t%s: %.3f' % (key,value))


# In[241]:


y2.plot()


# In[142]:


print("Data Shape: {}".format(data2.shape))
value_1 = data2[0:198]
value_2 = data2[199:398]


# In[242]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


# In[247]:


plot_acf(data2,lags=20)


# In[246]:


plot_pacf(data2,lags=40)


# In[ ]:


Forecasting models

"Pmdarima" is a useful package to help us find the best parameters for SARIMA model. 


# In[157]:


get_ipython().system('pip install pmdarima')


# In[158]:


from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


ARIMA Forecast to get best p,d,q,P,D,Q values

#auto_arima(df['Monthly beer production'], seasonal=True, m=12,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4).summary()
As we can see best arima model chosen by auto_arima() is SARIMAX(2, 1, 1)x(4, 0, 3, 12)


# In[253]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[ ]:


The model summary provides lot of information. The table in the middle is the coefficients table where the values under ‘coef’ are the weights of the respective terms.

The coefficient of the MA3 term is close to zero and the P-Value in ‘P>|z|’ column is highly insignificant. It should ideally be less than 0.05 for the respective X to be significant.

So, we will rebuild the model without the MA2 term.


# In[280]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA


# In[283]:


from statsmodels.tsa.arima_model import ARIMA

model1=ARIMA(data2['IPG2211A2N'],order=(2,1,4))
model1_fit=model1.fit()
print(model1_fit.summary())


# In[ ]:


When we look at the plot we can sey there is a seasonality in data. That is why we will use SARIMA (Seasonal ARIMA) instead of ARIMA.

There are four seasonal elements that are not part of ARIMA that must be configured; they are:
P: Seasonal autoregressive order.
D: Seasonal difference order.
Q: Seasonal moving average order.
m: The number of time steps for a single seasonal period.
    
Now we will try to get best p,d,q,P,D,Q values baed on the acf and pacf plots


# In[284]:


#ARIMA(1,1,1)(3,1,1)[6]  
model2=SARIMAX(data2,order=(1,1,1), seasonal_order=(3,1,1,6))
results=model.fit()
results.summary()


# In[ ]:


The model summary provides lot of information. The table in the middle is the coefficients table where the values under ‘coef’ are the weights of the respective terms.

The coefficient of the MA3 term is close to zero and the P-Value in ‘P>|z|’ column is highly insignificant. It should ideally be less than 0.05 for the respective X to be significant.

So, we will rebuild the model.


# In[287]:


#ARIMAX(4, 1, 4)x(4, 0, [1], 12) 
model2=SARIMAX(data2,order=(4,1,4), seasonal_order=(4,0,1,12))
results=model.fit()
results.summary()


# In[326]:


model=SARIMAX(data2,order=(2,1,2),  seasonal_order=(1, 1, 2, 12))
results=model.fit()


# In[327]:


results.summary()


# In[295]:


results.plot_diagnostics(figsize=(8,8))
plt.show()


# In[ ]:


All the 4 plots indicates a good fit of the SARIMA model on the given time serie.The model AIC has slightly reduced, which is good. The p-values of the AR1 and MA1 terms have improved and are highly significant (<< 0.05).

Let’s plot the residuals to ensure there are no patterns (that is, look for constant mean and variance).


# In[379]:


# Plotting residual 
residuals = pd.DataFrame(model2_fit.resid)
plt.figure(figsize=(18, 8))
plt.plot(residuals)
plt.title('Residuals', fontsize=16)
plt.xlabel("IPG2211A2N", fontsize=14)
plt.ylabel("Amount of residuals", fontsize=14)
plt.xticks(np.arange(0, len(data2.IPG2211A2N)+1, 45), labels=[data2.IPG2211A2N[i] for i in range(0, len(data2.IPG2211A2N)+1, 45)], fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[381]:


residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# In[ ]:


The residual errors seem fine with near zero mean and uniform variance


# In[391]:


import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(train['IPG2211A2N'],order=(4,0,2),seasonal_order=(1,1,1,12))
results=model.fit()
results.summary()


# In[392]:


train['forecast']=results.predict(start=300,end=384,dynamic=True)
train[['IPG2211A2N','forecast']].plot(figsize=(10,6))


# In[ ]:


Prediction


# In[344]:


train= data2.iloc[:-30]
test= data2.iloc[-30:]


# In[360]:


#I train the model

model = SARIMAX(data2['IPG2211A2N'],order=(2,1,2),  seasonal_order=(1, 1, 2, 12))
results = model.fit()


# In[357]:


start=len(train)
end=len(train)+len(test)
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels')
print(predictions)


# In[358]:


predictions.plot(legend=True)


# In[422]:


import arch
from arch import arch_model

am = arch_model(returns)
res = am.fit(update_freq=5)
print(res.summary())


# In[ ]:




