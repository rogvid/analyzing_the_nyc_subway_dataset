from geopy.geocoders import Nominatim
from mpl_toolkits.basemap import Basemap


def draw_latlon(llclat, urclat, llclon, urclon, rsphere=6371200, resolution='h', area_thresh=0.1, projection='merc'):
    m = Basemap(llcrnrlat=llclat, urcrnrlat=urclat,
                llcrnrlon=llclon, urcrnrlon=urclon,
                rsphere=rsphere, resolution=resolution,
                area_thresh=area_thresh, projection=projection)
    m.drawcoastlines()
    m.drawcountries()


def latlon_basemap(llclat, urclat, llclon, urclon, rsphere=6371200, resolution='h', area_thresh=0.1, projection='merc'):
    m = Basemap(llcrnrlat=llclat, urcrnrlat=urclat,
                llcrnrlon=llclon, urcrnrlon=urclon,
                rsphere=rsphere, resolution=resolution,
                area_thresh=area_thresh, projection=projection)
    return m


def draw_country(country, rsphere=6371200, resolution='h', area_thresh=0.1, projection='merc'):
    geolocator = Nominatim()
    location = geolocator.geocode(country)
    raw = location.raw
    latlon = map(float, raw['boundingbox'])
    llclat = latlon[0]
    urclat = latlon[1]
    llclon = latlon[2]
    urclon = latlon[3]
    m = Basemap(llcrnrlat=llclat, urcrnrlat=urclat,
                llcrnrlon=llclon, urcrnrlon=urclon,
                rsphere=rsphere, resolution=resolution,
                area_thresh=area_thresh, projection=projection)
    m.drawcoastlines()
    m.drawcountries()


def country_basemap(country, rsphere=6371200, resolution='h', area_thresh=0.1, projection='merc'):
    geolocator = Nominatim()
    location = geolocator.geocode(country)
    raw = location.raw
    latlon = map(float, raw['boundingbox'])
    llclat = latlon[0]
    urclat = latlon[1]
    llclon = latlon[2]
    urclon = latlon[3]
    m = Basemap(llcrnrlat=llclat, urcrnrlat=urclat,
                llcrnrlon=llclon, urcrnrlon=urclon,
                rsphere=rsphere, resolution=resolution,
                area_thresh=area_thresh, projection=projection)
    return m
