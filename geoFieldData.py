import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt


def get_data_slice(data, lat_range, lon_range, level_index):
    """
    Extracts a slice from the netCDF data based on latitude, longitude, and level.

    Parameters:
    - data: The netCDF dataset.
    - lat_range: Tuple of (start_lat, end_lat) indices.
    - lon_range: Tuple of (start_lon, end_lon) indices.
    - level_index: Index of the level.

    Returns:
    - DataFrame containing longitude, latitude, and temperature.
    """
    air_data = data.variables['air']
    lats = data.variables['lat'][lat_range[0]:lat_range[1]]
    lons = data.variables['lon'][lon_range[0]:lon_range[1]]
    slice_data = air_data[0, level_index, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]]

    df = pd.DataFrame({
        'longitude': lons.repeat(len(lats)),
        'latitude': np.tile(lats, len(lons)),
        'temperature': slice_data.flatten()
    })

    return df


def plot_on_basemap(df):
    """
    Plots the data from the DataFrame on a basemap.

    Parameters:
    - df: DataFrame containing longitude, latitude, and temperature.

    Returns:
    - Matplotlib plot.
    """
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    gdf.crs = "EPSG:4326"
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(column='temperature', ax=ax, legend=True, cmap='viridis', markersize=100,
             legend_kwds={'label': "Temperature (degK)"})
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_title('Air Temperature Heatmap')
    plt.show()
