# importing the libraries
import pandas as pd # This Library used for data manipulation and analysis
import numpy as np # it is used for mathematical operations

def get_data(name):
  """Read the data from excel file. Drop an empty column called 'Unnamed: 4'. Remove all non-countries starting from index 217.
  Returns the original and modified data frames"""
  dat_act = pd.read_excel(name) # reading file in to excel
  dat_trf = dat_act.drop('Unnamed: 4', axis=1) # dropping unwanted columns
  dat_trf = pd.DataFrame.transpose(dat_trf.iloc[:217, :]) # transpose data
  return dat_act, dat_trf

dat_act, dat_trf = get_data('D:\Ramya\ADS1 Last Assign/Data_Extract_FromWorld Development Indicators.xlsx')

df = pd.DataFrame.transpose(dat_trf) # Dataframe transpose
df.head()

df.shape

# removing non-countries after Zimbabwe
df.iloc[216]

#no missing values
df.isna().sum()

#removing commas in numbers
df = df.replace(',','', regex=True)
df.head()

#datatype is still object and not int or float
df.info()

import re

def missing_dat(val):
  """
      Replace all dots with zero to indicate they are missing.
      All non-dots are made integers. 
      Modified item is returned
  """
  try:
    s = int(val)
  except:
    s = re.sub(r".*", '0', val)
  return s

df['1990'] = [int(missing_dat(str(val)[:-2])) for val in df['1990']]
df['2000'] = [int(missing_dat(str(val)[:-2])) for val in df['2000']]
df['2012'] = [int(missing_dat(str(val)[:-2])) for val in df['2012']]
df['2013'] = [int(missing_dat(str(val)[:-2])) for val in df['2013']]
df['2014'] = [int(missing_dat(str(val)[:-2])) for val in df['2014']]
df['2015'] = [int(missing_dat(str(val)[:-2])) for val in df['2015']]
df['2016'] = [int(missing_dat(str(val)[:-2])) for val in df['2016']]
df['2017'] = [int(missing_dat(str(val)[:-2])) for val in df['2017']]
df['2018'] = [int(missing_dat(str(val)[:-2])) for val in df['2018']]
df['2019'] = [int(missing_dat(str(val)[:-2])) for val in df['2019']]
df['2020'] = [int(missing_dat(str(val)[:-2])) for val in df['2020']]
df['2021'] = [int(missing_dat(str(val)[:-2])) for val in df['2021']]

df.head()

# all columns are now type int
df.info()

def norm(array):
  """ 
      Returns array normalised to [0,1]. Array can be a numpy
      array or a column of a dataframe
  """
  min_val = np.min(array)
  max_val = np.max(array)
  scaled = (array-min_val) / (max_val-min_val)
  return scaled

def norm_df(df, first=0, last=None):
  """
      Returns all columns of the dataframe normalised to [0,1] with the exception of 
      the first (containing the names) Calls function norm to do the normalisation of
      one column, but doing all in one function is also fine. First, last: columns from
      first to last (including) are normalised. Defaulted to all. None is the empty entry. 
      The default corresponds
  """
  # iterate over all numerical columns
  for col in df.columns[first:last]: # excluding the first column
    df[col] = norm(df[col])
  return df

# obtain columns for fitting
df_fit = df[["2020", "2021"]].copy()
#using normalization function on those columns
df_fit = norm_df(df_fit)
df_fit.describe()


# importing sklearn libraries
import sklearn.cluster as cluster
import sklearn.metrics as skmet

for ic in range(2, 10):
  # set up kmeans and fit
  kmeans = cluster.KMeans(n_clusters=ic)
  kmeans.fit(df_fit)
  # extract labels and calculate silhoutte score
  labels = kmeans.labels_
  print (ic, skmet.silhouette_score(df_fit, labels))

"""
    Selecting 2 clusters as the optimal number (highest silhouette score)

    Plot on normalized values
"""

import matplotlib.pyplot as plt

#Plot for 2 clusters
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
# Individual colours can be assigned to symbols. The label l is used to the select the l-th number from the colour table.
plt.scatter(df_fit["2020"], df_fit["2021"], c=labels, cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(2):
  xc, yc = cen[ic,:]
  plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("2020")
plt.ylabel("2021")
plt.title("2 clusters on normalized values")
plt.show()

""" 
    Plot on original values
    
"""

cols = df[["2020", "2021"]].copy()
cols['clust_num'] = kmeans.labels_
cols.head()

#Plot for 2 clusters
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
# Individual colours can be assigned to symbols. The label l is used to the select the l-th number from the colour table.
plt.scatter(cols["2020"], cols["2021"], c=cols['clust_num'], cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(2):
  xc, yc = cen[ic,:]
  plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("2020")
plt.ylabel("2021")
plt.title("2 clusters on original values")
plt.show()

cols['clust_num'].value_counts()

cols[cols['clust_num'] == 1]

""" 
    Curve fitting
    
"""

def exp_growth(t, scale, growth):
  """ 
      Computes exponential function with scale and growth as free parameters
  """
  f = scale * np.exp(growth * (t-2000))
  return f

"""
    Picking China from cluster 1 and Andorra from cluster 0
    
"""

chn = df[df['Country Name'] == 'China']
adr = df[df['Country Name'] == 'Andorra']

chn = chn.drop('Country Name', axis=1)
chn = chn.T.reset_index()
chn.columns = ['Date', 'GDP']
temp = []
for i in chn['Date']:
  temp.append(int(i[:4]))
chn['Date'] = temp

adr.drop('Country Name', axis=1, inplace=True)
adr = adr.T.reset_index()
adr.columns = ['Date', 'GDP']

temp = []
for i in adr['Date']:
  temp.append(int(i[:4]))
adr['Date'] = temp

import scipy.optimize as opt

"""
    China fitting

    First fit attempt of the exponential function with defaul initial parameters
    
"""

# fit exponential growth
popt, covar = opt.curve_fit(exp_growth, chn["Date"], chn["GDP"])

print("Fit parameter", popt)
# use *popt to pass on the fit parameters
chn["gdp_exp"] = exp_growth(chn["Date"], *popt)
plt.figure()
plt.plot(chn["Date"], chn["GDP"], label="data")
plt.plot(chn["Date"], chn["gdp_exp"], label="fit")
plt.legend()
plt.title("First fit attempt")
plt.xlabel("year")
plt.ylabel("GDP")
plt.show()
print()

"""Finding a start approximation"""

# find a feasible start value the pedestrian way
popt = [7e12, 0.04]
chn["gdp_exp"] = exp_growth(chn["Date"], *popt)
plt.figure()
plt.plot(chn["Date"], chn["GDP"], label="data")
plt.plot(chn["Date"], chn["gdp_exp"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("GDP")
plt.title("Improved start value")
plt.show()

# fit exponential growth
popt, covar = opt.curve_fit(exp_growth, chn["Date"],
chn["GDP"], p0=[7e12, 0.04])
# much better
print("Fit parameter", popt)
chn["gdp_exp"] = exp_growth(chn["Date"], *popt)
plt.figure()
plt.plot(chn["Date"], chn["GDP"], label="data")
plt.plot(chn["Date"], chn["gdp_exp"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("GDP")
plt.title("Final fit exponential growth")
plt.show()
print()

"""
    Estimate lower and upper limits of the
    confidence range
"""

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper

# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)
low, up = err_ranges(chn["Date"], exp_growth, popt, sigma)
plt.figure()
plt.title("lower and upper limits")
plt.plot(chn["Date"], chn["GDP"], label="data")
plt.plot(chn["Date"], chn["gdp_exp"], label="fit")
plt.fill_between(chn["Date"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("GDP")
plt.show()

"""Give ranges """

print("Forecasted GDP")
low, up = err_ranges(2030, exp_growth, popt, sigma)
print("2030 between ", low, "and", up)
low, up = err_ranges(2040, exp_growth, popt, sigma)
print("2040 between ", low, "and", up)
low, up = err_ranges(2050, exp_growth, popt, sigma)
print("2050 between ", low, "and", up)

"""Andorra Fitting"""

# fit exponential growth
popt_adr, covar_adr = opt.curve_fit(exp_growth, adr["Date"], adr["GDP"])

print("Fit parameter", popt_adr)
# use *popt to pass on the fit parameters
adr["gdp_exp"] = exp_growth(adr["Date"], *popt_adr)
plt.figure()
plt.plot(adr["Date"], adr["GDP"], label="data")
plt.plot(adr["Date"], adr["gdp_exp"], label="fit")
plt.legend()
plt.title("First fit attempt")
plt.xlabel("year")
plt.ylabel("GDP")
plt.show()
print()

adr

"""Finding a start approximation"""

# find a feasible start value the pedestrian way
popt_adr = [2e9, 0.04]
adr["gdp_exp"] = exp_growth(adr["Date"], *popt_adr)
plt.figure()
plt.plot(adr["Date"], adr["GDP"], label="data")
plt.plot(adr["Date"], adr["gdp_exp"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("GDP")
plt.title("Improved start value")
plt.show()

# fit exponential growth
popt_adr, covar_adr = opt.curve_fit(exp_growth, adr["Date"], adr["GDP"], p0=[2e9, 0.04])
# much better
print("Fit parameter", popt_adr)
adr["gdp_exp"] = exp_growth(adr["Date"], *popt_adr)
plt.figure()
plt.plot(adr["Date"], adr["GDP"], label="data")
plt.plot(adr["Date"], adr["gdp_exp"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("GDP")
plt.title("Final fit exponential growth")
plt.show()
print()

# extract the sigmas from the diagonal of the covariance matrix
sigma_adr = np.sqrt(np.diag(covar_adr))
print(sigma_adr)
low, up = err_ranges(adr["Date"], exp_growth, popt_adr, sigma_adr)
plt.figure()
plt.title("lower and upper limits")
plt.plot(adr["Date"], adr["GDP"], label="data")
plt.plot(adr["Date"], adr["gdp_exp"], label="fit")
plt.fill_between(adr["Date"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("GDP")
plt.show()


