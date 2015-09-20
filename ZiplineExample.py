__author__ = 'tom'

import warnings
warnings.simplefilter('ignore')
import zipline
import pytz
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import pymc3 as pm

# zipline returns a dataframe
data = zipline.data.load_from_yahoo(stocks=['GLD', 'GDX'], end=dt.datetime(2015, 3, 15, 0, 0, 0, 0, pytz.utc)).dropna()
# print(data.info)

data.plot(figsize=(8,4))
plt.show()

# see how correlated two stocks are
print(data.corr())

# need to convert dates to ordinal date representation for matplotlib to use
# needed to add _mpl_repr() to fix a bug with date2num not accepting datetime64 types
mpl_dates = mpl.dates.date2num(data.index._mpl_repr())

# scatter plot of GDX and GLD
plt.figure(figsize=(8, 4))
plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o')
plt.grid(True)
plt.xlabel('GDX')
plt.ylabel('GLD')
plt.colorbar(ticks=mpl.dates.DayLocator(interval=250), format=mpl.dates.DateFormatter('%d %b %y'))

plt.show()

# Bayesian regression on basis of two time series
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=20)
    beta = pm.Normal('beta', mu=0, sd=20)
    sigma = pm.Uniform('sigma', lower=0, upper=50)


    y_est = alpha + beta + data['GDX'].values

    likelihood = pm.Normal('GLD', mu=y_est, sd=sigma, observed=data['GLD'].values)

    start = pm.find_MAP()
    step = pm.NUTS(state=start)
    trace = pm.sample(100, step, start=start, progressbar=False)

fig = pm.traceplot(trace)
plt.figure(figsize=(8,8))
plt.show()

# add regression lines to scatter plot
plt.figure(figsize=(8,4))
plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o')
plt.grid(True)
plt.xlabel('GDX')
plt.ylabel('GLD')
for i in range(len(trace)):
    plt.plot(data['GDX'], trace['alpha'][i] + trace['beta'][i] * data['GDX'])
plt.colorbar(ticks=mpl.dates.DayLocator(interval=250), format=mpl.dates.DateFormatter('%d %b %y'))
plt.show()

# ...continue with updating regression line over time...
