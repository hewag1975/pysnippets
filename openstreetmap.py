# openstreetmap

## OSMnx
# https://pygis.io/docs/d_access_osm.html
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

place = "Forchheim"
area = ox.geocode_to_gdf(place)

area.head()
# type(area)

area.plot()
plt.show()

tags = {'building': True}
bld = ox.geometries_from_place(place, tags=tags)
bld.head()
bld.shape

bld.plot()
plt.show()