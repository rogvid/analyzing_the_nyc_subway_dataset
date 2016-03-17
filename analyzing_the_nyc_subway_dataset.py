import datetime
import pandas as pd
import pandasql
import csv
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import sys
import os
import pickle
from geopy.geocoders import Nominatim
import matplotlib
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import shapefile
from matplotlib.collections import LineCollection
from matplotlib import cm
import country_basemaps

sns.set(style='white')

# plt.style.use("ggplot")

new_style = {'grid': False} #Remove grid
matplotlib.rc('axes', **new_style)
from matplotlib import rcParams
rcParams['figure.figsize'] = (8.5, 8.5) #Size of figure
rcParams['figure.dpi'] = 250

fig = plt.figure(1)
ax = fig.add_subplot(111)

#r = shapefile.Reader(r"./data/USA_adm")
#shapes = r.shapes()
#records = r.records()

m = Basemap(resolution='f', projection='merc', llcrnrlat=40.61, urcrnrlat=40.91, llcrnrlon=-74.06, urcrnrlon=-73.77,  area_thresh=0.1)
m.drawcoastlines()
m.fillcontinents(color='black', lake_color="white")

# pd.options.display.mpl_style = 'default' #Better Styling
#Inline Plotting for Ipython Notebook

if os.path.exists("./geopy_cache.p"):
    cache = pickle.load(open("./geopy_cache.p", "rb"))
else:
    cache = None

def address2location(address):
    """
    Converts address + city abbrevation into location object.
    """
    geolocator = Nominatim()
    location = geolocator.geocode(address)
    return location


def geocache(search_term):
    if cache is None:
        location = address2location(search_term)
        cache = {search_term.lower(): location}
        pickle.dump(cache, open("./geopy_cache.p", "wb"))
    elif search_term.lower() not in cache.keys():
        location = address2location(search_term)
        cache = {search_term.lower(): location}
        pickle.dump(cache, open("./geopy_cache.p", "wb"))
    else:
        location = cache[search_term]
    return location


def boothAddress2latlon(boothAddress):
    """
    Takes as input booth address and returns a lat lon of that station
    """
    return geocache(boothAddress).point[:-1]


df = pd.read_csv("./data/turnstile_weather_v2.csv", sep=",")
# latlon = pd.DataFrame(df.groupby(["latitude", "longitude"])["ENTRIESn_hourly"].count())
newdf = pd.DataFrame({'meanENTRIESn_hourly': df.groupby(["latitude", "longitude"])["ENTRIESn_hourly"].mean()}).reset_index()
xpt, ypt = np.array(m(df['longitude'].values, df['latitude'].values))
# xpt, ypt = newdf['latitude'].values, newdf['longitude'].values
m.scatter(x=xpt[(df['rain']==0).values], y=ypt[(df['rain']==0).values],
          s=df[(df['rain']==0).values]['ENTRIESn_hourly']/(len(df) - df['rain'].sum())*25.0,
          marker='o',
          color="#f6ca32",
          zorder=10,
          alpha=0.2)

sns.despine(left=True, bottom=True)
plt.savefig("./NYC_latlon_ENTRIESn_hourly_no_rain.png", dpi=600, bbox_inches='tight')
plt.show()

fig = plt.figure(2)
ax = fig.add_subplot(111)

m.drawcoastlines()
m.fillcontinents(color='black', lake_color="white")

m.scatter(x=xpt[(df['rain']==1).values], y=ypt[(df['rain']==1).values],
          s=df[(df['rain']==1).values]['ENTRIESn_hourly']/df['rain'].sum()*25.0,
          marker='o',
          color="#32caf6",
          zorder=10,
          alpha=0.2)

sns.despine(left=True, bottom=True)
plt.savefig("./NYC_latlon_ENTRIESn_hourly_rain.png", dpi=600, bbox_inches='tight')
plt.show()
# ax.scatter(x=newdf['latitude'], y=newdf['longitude'], s=newdf['meanENTRIESn_hourly']/100.0, color="#32caf6", zorder=10)
# plt.xlim(-74.06, -73.77)
# plt.ylim(40.61, 40.91)
# P=df[["ENTRIESn_hourly", "latitude", "longitude"]].plot(kind='scatter', x='longitude', y='latitude', color='#32caf6',xlim=(-74.06,-73.77),ylim=(40.61, 40.91),s=df['ENTRIESn_hourly'] / 1000.0,alpha=.6, ax=ax)
# P=df["ENTRIESn_hourly"].plot(color='#32caf6',xlim=(-74.06,-73.77),ylim=(40.61, 40.91),s=.02,alpha=.6)
# P.set_axis_bgcolor('white') #Background Color
# fig = plt.figure()
# ax = fig.add_subplot(111)
# columns = ["day_week", "hour", "ENTRIESn_hourly", "fog", "rain", "meanprecipi", "meantempi"]
# scatter_matrix(df[columns], alpha=0.2, figsize=(8, 8), diagonal="kde")
# df[["ENTRIESn_hourly", "fog"]].groupby("fog").plot(kind="hist", bins=10, alpha=0.5)
# df[["ENTRIESn_hourly", "rain"]].groupby("rain").plot(kind="hist", alpha=0.5)
# df.plot(kind="scatter", x='meanprecipi', y='ENTRIESn_hourly', alpha=0.5)
# dfh = pd.melt(df, id_vars=['hour'], value_vars=['ENTRIESn_hourly'], value_name="# Entries").groupby("hour")
# dfh.mean().plot(kind='bar', yerr=dfh.std())
# dfdw = pd.melt(df, id_vars=['day_week'], value_vars=['ENTRIESn_hourly'], value_name="# Entries").groupby("day_week")
# dfdw.mean().plot(kind='bar', yerr=dfdw.std())
# ax = sns.violinplot(x="day_week", y="ENTRIESn_hourly", data=df, inner='quartile', hue='rain', split=True, scale='count', scale_hue=False)
