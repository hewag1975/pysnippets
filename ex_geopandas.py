
# source: https://towardsdatascience.com/geospatial-data-analysis-with-geopandas-876cb72721cb

# import pandas as pd
# import geopandas as gpd
#
# df = pd.read_csv('data/CARVAN_OASA_England_Northern_Ireland_Scotland_Wales_Descriptions.csv')
# # df = clean_df(df)
# df.sample(3)
#
# df_geo = gpd.read_file('data/infuse_oa_lyr_2011/infuse_oa_lyr_2011.shp')
# gpd.GeoDataFrame(df_geo)
# df_geo.sample(3)


# https://geopandas.org/en/stable/getting_started/introduction.html
# sudo apt install python3-tk
# pip install pyqt5
# import matplotlib.pyplot as plt
# plt.show()

import geopandas as gpd
from geodatasets import get_path

# read file
path_to_data = get_path("nybb")
gdf = gpd.read_file(path_to_data)

gdf.set_index("BoroName")
gdf["area"] = gdf.area
gdf["area"]

# gdf["centroid"] = gdf.centroid

gdf.plot("area", legend=True)
gdf.explore("area", legend=False)





