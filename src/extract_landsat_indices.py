import rasterio
import pandas as pd
from rasterio.transform import rowcol
from rasterio.warp import transform
import numpy as np

COORDINATES_FILE = "trap_coordinates.csv"
OUTPUT_FILE = "coordinates_with_indices.csv"

BAND_FILES = {
    'B2': 'LE07_L2SP_229082_20231005_20231031_02_T1_SR_B2.TIF',
    'B3': 'LE07_L2SP_229082_20231005_20231031_02_T1_SR_B3.TIF',
    'B4': 'LE07_L2SP_229082_20231005_20231031_02_T1_SR_B4.TIF',
    'B5': 'LE07_L2SP_229082_20231005_20231031_02_T1_SR_B5.TIF',
}


def extract_pixel_value(band, lon, lat):
    lon, lat = transform('EPSG:4326', band.crs, [lon], [lat])
    row, col = rowcol(band.transform, lon[0], lat[0])
    try:
        return band.read(1)[row, col]
    except IndexError:
        return np.nan


df = pd.read_csv(COORDINATES_FILE)

bands = {name: rasterio.open(path) for name, path in BAND_FILES.items()}

ndvi_vals, ndbi_vals, ndwi_vals = [], [], []

for _, row in df.iterrows():
    lat, lon = row['Latitude'], row['Longitude']

    b2 = extract_pixel_value(bands['B2'], lon, lat)
    b3 = extract_pixel_value(bands['B3'], lon, lat)
    b4 = extract_pixel_value(bands['B4'], lon, lat)
    b5 = extract_pixel_value(bands['B5'], lon, lat)

    ndvi = (b4 - b3) / (b4 + b3) if (b4 + b3) != 0 else np.nan
    ndbi = (b5 - b4) / (b5 + b4) if (b5 + b4) != 0 else np.nan
    ndwi = (b2 - b4) / (b2 + b4) if (b2 + b4) != 0 else np.nan

    ndvi_vals.append(ndvi)
    ndbi_vals.append(ndbi)
    ndwi_vals.append(ndwi)

df['NDVI'] = ndvi_vals
df['NDBI'] = ndbi_vals
df['NDWI'] = ndwi_vals

df.to_csv(OUTPUT_FILE, index=False)
