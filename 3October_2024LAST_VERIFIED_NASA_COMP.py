# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:42:59 2024

@author: Moham
"""


############New_Verified_ LAST 

#### Downloading to google drive

import ee
import geemap
import time
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Step 1: Authenticate and Initialize Earth Engine
ee.Authenticate()
ee.Initialize()

# Step 2: Define AOI for Künzelsau, Germany using a 10 km buffer
aoi = ee.Geometry.Point([9.6833, 49.2806]).buffer(10000)

# Load Landsat Collection 2 data for the specified AOI
landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                      .filterBounds(aoi) \
                      .filterDate('2021-01-01', '2024-10-01') \
                      .filter(ee.Filter.lt('CLOUD_COVER', 15))

# Check if the collection contains images
collection_size = landsat_collection.size().getInfo()
print(f"Number of images in the collection: {collection_size}")

if collection_size == 0:
    raise ValueError("No images found in the specified collection for the given AOI and date range.")

# Create a visualization map using geemap
Map = geemap.Map(center=(49.2806, 9.6833), zoom=8)
Map.addLayer(landsat_collection.median(), {'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'max': 3000}, "Landsat 8 SR")
Map.addLayerControl()
Map

# Use the median image from the collection and unmask it to fill empty pixels
image = landsat_collection.median().unmask()

# Export the unmasked image to Google Drive
export_task = ee.batch.Export.image.toDrive(
    image=image,
    description='Landsat_Export',
    folder='LANDSAT_TEST1_October',  # Change to 'None' to export to the root Google Drive directory
    fileNamePrefix='Landsat_8_SR',
    region=aoi,
    scale=30,  # Use 30 meters per pixel resolution
    crs='EPSG:4326',  # Use standard WGS84 projection
    maxPixels=1e9  # Max pixels allowed in export
)
export_task.start()

# Monitor the export task
while export_task.active():
    print('Exporting...')
    time.sleep(30)
print('Export completed!')

# Step 3: Authenticate and create the PyDrive client
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates a local webserver and automatically handles authentication
drive = GoogleDrive(gauth)

# Step 4: Search for the exported file in Google Drive
file_list = drive.ListFile({'q': "title contains 'Landsat_8_SR' and trashed=false"}).GetList()
for file in file_list:
    print(f"Downloading {file['title']} from Google Drive...")
    file.GetContentFile(r'C:\Users\Moham\Desktop\PhD Data\NASA_Comp\{}'.format(file['title']))

print('File downloaded to C:\\Users\\Moham\\Desktop\\PhD Data\\NASA_Comp')



##### No Grid Cells (Also No Labels)


import geemap
import matplotlib.pyplot as plt
import rasterio
import numpy as np

# Step 1: Load the Landsat image using rasterio
image_path = r"C:\Users\Moham\Desktop\PhD Data\NASA_Comp\Landsat_8_SR.tif"
with rasterio.open(image_path) as src:
    bands = src.read()  # Read all bands
    profile = src.profile  # Get metadata of the image

# Step 2: Create a visualization map using geemap
Map = geemap.Map(center=(49.2806, 9.6833), zoom=8)

# 1. **RGB Visualization**
rgb = np.dstack((bands[3], bands[2], bands[1]))  # Band 4 (Red), Band 3 (Green), Band 2 (Blue)

# 2. **False Color Composite (NIR, Red, Green)**
false_color = np.dstack((bands[4], bands[3], bands[2]))  # Band 5 (NIR), Band 4 (Red), Band 3 (Green)



# Use Matplotlib to display NDVI
plt.figure(figsize=(10, 6))
plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)  # Color palette for NDVI
plt.colorbar(label='NDVI')
plt.title('NDVI Visualization')
plt.axis('off')
plt.show()

# 4. **EVI Calculation and Visualization**
evi = (2.5 * (bands[4] - bands[3])) / (bands[4] + 6 * bands[3] - 7.5 * bands[1] + 1)  # EVI formula

# Use Matplotlib to display EVI
plt.figure(figsize=(10, 6))
plt.imshow(evi, cmap='RdYlGn', vmin=-1, vmax=1)  # Color palette for EVI
plt.colorbar(label='EVI')
plt.title('EVI Visualization')
plt.axis('off')
plt.show()

# 5. **Custom Colormap Visualization (using Band 4)**
plt.figure(figsize=(10, 6))
plt.imshow(bands[3], cmap='plasma')  # Using Band 4 for custom colormap
plt.colorbar()
plt.title('Custom Colormap Visualization of Band 4 (Red)')
plt.axis('off')
plt.show()









import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the Landsat image using rasterio
image_path = r"C:\Users\Moham\Desktop\PhD Data\NASA_Comp\Landsat_8_SR.tif"
with rasterio.open(image_path) as src:
    profile = src.profile  # Get metadata of the image
    bands = src.read()  # Read all bands

# Define the AOI based on the image bounds
aoi_bounds = src.bounds
print(f"AOI Bounds: {aoi_bounds}")

# Step 2: Create a 3x3 Grid over the AOI
# Convert bounds to numpy array
min_x, min_y, max_x, max_y = aoi_bounds
grid_size = 3  # 3x3 grid
x_coords = np.linspace(min_x, max_x, grid_size + 1)
y_coords = np.linspace(min_y, max_y, grid_size + 1)

# Create grid cells
grid_cells = []
for i in range(grid_size):
    for j in range(grid_size):
        cell = {
            'min_x': x_coords[i],
            'min_y': y_coords[j],
            'max_x': x_coords[i + 1],
            'max_y': y_coords[j + 1],
            'center_x': (x_coords[i] + x_coords[i + 1]) / 2,
            'center_y': (y_coords[j] + y_coords[j + 1]) / 2,
        }
        grid_cells.append(cell)

# Step 3: Define a function to add the grid overlay
def add_grid_overlay(ax, grid_cells, color='red', linewidth=2):
    for cell in grid_cells:
        rectangle = plt.Rectangle((cell['min_x'], cell['min_y']),
                                  cell['max_x'] - cell['min_x'],
                                  cell['max_y'] - cell['min_y'],
                                  linewidth=linewidth, edgecolor=color, facecolor='none')
        ax.add_patch(rectangle)

# Step 4: Visualize the original image with the grid cells
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(bands[0], cmap='gray', extent=aoi_bounds)  # Display the first band of the Landsat image
add_grid_overlay(ax, grid_cells)
plt.title('3x3 Grid Cells Over AOI (Original Image)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()



# Step 7: NDVI Calculation and Visualization
ndvi = (bands[4] - bands[3]) / (bands[4] + bands[3])  # NDVI = (NIR - Red) / (NIR + Red)
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(ndvi, cmap='RdYlGn', extent=aoi_bounds, vmin=-1, vmax=1)  # Color palette for NDVI
plt.colorbar(label='NDVI')
add_grid_overlay(ax, grid_cells)
plt.title('NDVI Visualization with Grid Overlay')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()

# Step 8: EVI Calculation and Visualization
evi = (2.5 * (bands[4] - bands[3])) / (bands[4] + 6 * bands[3] - 7.5 * bands[1] + 1)  # EVI formula
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(evi, cmap='RdYlGn', extent=aoi_bounds, vmin=-1, vmax=1)  # Color palette for EVI
plt.colorbar(label='EVI')
add_grid_overlay(ax, grid_cells)
plt.title('EVI Visualization with Grid Overlay')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()

# Step 9: Custom Colormap Visualization using Band 4 (Red)
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(bands[3], cmap='plasma', extent=aoi_bounds)  # Using Band 4 for custom colormap
plt.colorbar(label='Reflectance')
add_grid_overlay(ax, grid_cells)
plt.title('Custom Colormap Visualization of Band 4 (Red) with Grid Overlay')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()


# Step 4: Define a function to download data (if required)
def download_data(cell):
    # This function would include the logic to download data based on cell coordinates
    # Example: Use Earth Engine or other APIs to download data for the specific cell
    print(f"Downloading data for cell with coordinates: {cell}")

# Uncomment to download data for each grid cell
# for cell in grid_cells:
    # download_data(cell)


import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the Landsat image using rasterio
image_path = r"C:\Users\Moham\Desktop\PhD Data\NASA_Comp\Landsat_8_SR.tif"
with rasterio.open(image_path) as src:
    profile = src.profile  # Get metadata of the image
    bands = src.read()  # Read all bands

# Define the AOI based on the image bounds
aoi_bounds = src.bounds
min_x, min_y, max_x, max_y = aoi_bounds

# Create a 3x3 Grid over the AOI
grid_size = 3
x_coords = np.linspace(min_x, max_x, grid_size + 1)
y_coords = np.linspace(min_y, max_y, grid_size + 1)

# Create grid cells
grid_cells = []
for i in range(grid_size):
    for j in range(grid_size):
        cell = {
            'min_x': x_coords[i],
            'min_y': y_coords[j],
            'max_x': x_coords[i + 1],
            'max_y': y_coords[j + 1],
            'center_x': (x_coords[i] + x_coords[i + 1]) / 2,
            'center_y': (y_coords[j] + y_coords[j + 1]) / 2,
        }
        grid_cells.append(cell)

# Define a function to add the grid overlay
def add_grid_overlay(ax, grid_cells, color='red', linewidth=2):
    for cell in grid_cells:
        rectangle = plt.Rectangle((cell['min_x'], cell['min_y']),
                                  cell['max_x'] - cell['min_x'],
                                  cell['max_y'] - cell['min_y'],
                                  linewidth=linewidth, edgecolor=color, facecolor='none')
        ax.add_patch(rectangle)

# Step 4: Visualize the original image with the grid cells
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(bands[0], cmap='gray', extent=aoi_bounds)  # Display the first band of the Landsat image
add_grid_overlay(ax, grid_cells)
plt.title('3x3 Grid Cells Over AOI (Original Image)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()

# Step 7: NDVI Calculation and Visualization
ndvi = (bands[4] - bands[3]) / (bands[4] + bands[3])  # NDVI = (NIR - Red) / (NIR + Red)
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(ndvi, cmap='viridis', extent=aoi_bounds, vmin=0, vmax=1)  # Using viridis colormap for better NDVI sensitivity
plt.colorbar(label='NDVI (0 to 1)')
add_grid_overlay(ax, grid_cells)
plt.title('Enhanced NDVI Visualization with Grid Overlay (Viridis Colormap)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()

# Step 8: EVI Calculation and Visualization
evi = (2.5 * (bands[4] - bands[3])) / (bands[4] + 6 * bands[3] - 7.5 * bands[1] + 1)  # EVI formula
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(evi, cmap='viridis', extent=aoi_bounds, vmin=-1, vmax=1)  # Color palette for EVI
plt.colorbar(label='EVI')
add_grid_overlay(ax, grid_cells)
plt.title('Enhanced EVI Visualization with Grid Overlay')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()

# Step 9: Custom Colormap Visualization using Band 4 (Red)
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(bands[3], cmap='magma', extent=aoi_bounds)  # Using Band 4 for custom colormap
plt.colorbar(label='Reflectance')
add_grid_overlay(ax, grid_cells)
plt.title('Custom Colormap Visualization of Band 4 (Red) with Grid Overlay')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()




##### Labelling the Figures! (Verified)
## 
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the Landsat image using rasterio
image_path = r"C:\Users\Moham\Desktop\PhD Data\NASA_Comp\Landsat_8_SR.tif"
with rasterio.open(image_path) as src:
    profile = src.profile  # Get metadata of the image
    bands = src.read()  # Read all bands

# Define the AOI based on the image bounds
aoi_bounds = src.bounds

# Create the NDVI
ndvi = (bands[4] - bands[3]) / (bands[4] + bands[3])  # NDVI = (NIR - Red) / (NIR + Red)

# Define NDVI ranges and assign labels
ndvi_classified = np.zeros_like(ndvi)
ndvi_classified[(ndvi >= -1) & (ndvi < 0)] = 0  # Water bodies
ndvi_classified[(ndvi >= 0) & (ndvi < 0.2)] = 1  # Barren Land
ndvi_classified[(ndvi >= 0.2) & (ndvi < 0.4)] = 2  # Sparse Vegetation
ndvi_classified[(ndvi >= 0.4) & (ndvi < 0.6)] = 3  # Moderate Vegetation
ndvi_classified[(ndvi >= 0.6) & (ndvi < 0.8)] = 4  # Dense Vegetation
ndvi_classified[(ndvi >= 0.8) & (ndvi <= 1)] = 5   # Very Dense Vegetation

# Corrected colormap with recognized color names
colors = ['blue', 'sandybrown', 'yellow', 'lightgreen', 'green', 'darkgreen']
cmap = ListedColormap(colors)

# Plot the Classified NDVI Map
fig, ax = plt.subplots(figsize=(10, 10))
im = plt.imshow(ndvi_classified, cmap=cmap, extent=aoi_bounds)
plt.colorbar(im, ticks=[0, 1, 2, 3, 4, 5], orientation='vertical', label='NDVI Classes')
ax.set_title('Classified NDVI Visualization with Corrected Colors')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()

# Add legend with class names
legend_labels = ['Water', 'Barren Land', 'Sparse Vegetation', 'Moderate Vegetation', 'Dense Vegetation', 'Very Dense Vegetation']
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(ndvi_classified, cmap=cmap, extent=aoi_bounds)
cbar = plt.colorbar(im, ticks=[0, 1, 2, 3, 4, 5], orientation='vertical')
cbar.ax.set_yticklabels(legend_labels)
ax.set_title('Classified NDVI with Legend')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()



import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the Landsat image using rasterio
image_path = r"C:\Users\Moham\Desktop\PhD Data\NASA_Comp\Landsat_8_SR.tif"
with rasterio.open(image_path) as src:
    bands = src.read()  # Read all bands

# Define the AOI based on the image bounds
aoi_bounds = src.bounds

# Define a function to classify an index based on thresholds
def classify_index(index_array, thresholds):
    classified_array = np.zeros_like(index_array)
    for i, (low, high) in enumerate(thresholds):
        classified_array[(index_array >= low) & (index_array < high)] = i
    return classified_array

# Define a function to plot classified maps with legends
def plot_classified_map(classified_array, title, extent, cmap, legend_labels):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(classified_array, cmap=cmap, extent=extent)
    cbar = plt.colorbar(im, ticks=np.arange(len(legend_labels)), orientation='vertical')
    cbar.ax.set_yticklabels(legend_labels)
    ax.set_title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid()
    plt.show()

# Define color maps and labels
ndvi_colors = ['blue', 'sandybrown', 'yellow', 'lightgreen', 'green', 'darkgreen']
ndvi_labels = ['Water', 'Barren Land', 'Sparse Vegetation', 'Moderate Vegetation', 'Dense Vegetation', 'Very Dense Vegetation']
evi_colors = ['blue', 'tan', 'lightyellow', 'lightgreen', 'darkgreen']
evi_labels = ['Water', 'Low Vegetation', 'Moderate Vegetation', 'High Vegetation', 'Very High Vegetation']
band4_colors = ['purple', 'blue', 'cyan', 'yellow', 'orange', 'red']
band4_labels = ['Low Reflectance', 'Medium Low', 'Medium', 'Medium High', 'High', 'Very High']

# Create NDVI, EVI, and custom Band 4 classifications
ndvi = (bands[4] - bands[3]) / (bands[4] + bands[3])  # NDVI = (NIR - Red) / (NIR + Red)
evi = (2.5 * (bands[4] - bands[3])) / (bands[4] + 6 * bands[3] - 7.5 * bands[1] + 1)  # EVI formula
band4 = bands[3]  # Band 4 (Red)

# Define classification thresholds for each index
ndvi_thresholds = [(-1, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]
evi_thresholds = [(-1, 0), (0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1)]
band4_thresholds = [(8000, 10000), (10000, 14000), (14000, 18000), (18000, 20000), (20000, 22000), (22000, 24000)]

# Classify each index
ndvi_classified = classify_index(ndvi, ndvi_thresholds)
evi_classified = classify_index(evi, evi_thresholds)
band4_classified = classify_index(band4, band4_thresholds)

# Plot classified NDVI with labels
ndvi_cmap = ListedColormap(ndvi_colors)
plot_classified_map(ndvi_classified, 'Classified NDVI Visualization', aoi_bounds, ndvi_cmap, ndvi_labels)

# Plot classified EVI with labels
evi_cmap = ListedColormap(evi_colors)
plot_classified_map(evi_classified, 'Classified EVI Visualization', aoi_bounds, evi_cmap, evi_labels)

# Plot classified Band 4 with labels
band4_cmap = ListedColormap(band4_colors)
plot_classified_map(band4_classified, 'Classified Band 4 (Red) Reflectance', aoi_bounds, band4_cmap, band4_labels)






import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the Landsat image using rasterio
image_path = r"C:\Users\Moham\Desktop\PhD Data\NASA_Comp\Landsat_8_SR.tif"
with rasterio.open(image_path) as src:
    bands = src.read()  # Read all bands

# Define the AOI based on the image bounds
aoi_bounds = src.bounds
print(f"AOI Bounds: {aoi_bounds}")

# Define a function to classify an index based on thresholds
def classify_index(index_array, thresholds):
    classified_array = np.zeros_like(index_array)
    for i, (low, high) in enumerate(thresholds):
        classified_array[(index_array >= low) & (index_array < high)] = i
    return classified_array

# Define a function to plot classified maps with legends
def plot_classified_map(classified_array, title, extent, cmap, legend_labels):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(classified_array, cmap=cmap, extent=extent, vmin=0, vmax=len(legend_labels)-1)
    cbar = plt.colorbar(im, ticks=np.arange(len(legend_labels)), orientation='vertical')
    cbar.ax.set_yticklabels(legend_labels)
    ax.set_title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid()
    plt.show()

# Define NDVI and NDBI color maps and labels based on NASA standards
ndvi_colors = ['blue', 'sandybrown', 'yellow', 'lightgreen', 'green', 'darkgreen']
ndvi_labels = ['Water', 'Barren Land', 'Sparse Vegetation', 'Moderate Vegetation', 'Dense Vegetation', 'Very Dense Vegetation']

ndbi_colors = ['blue', 'lightyellow', 'orange', 'red']
ndbi_labels = ['Water/Bare Land', 'Low Built-Up Area', 'Moderate Built-Up Area', 'High Built-Up Area']

# Calculate NDVI and NDBI
ndvi = (bands[4] - bands[3]) / (bands[4] + bands[3])  # NDVI = (NIR - Red) / (NIR + Red)
ndbi = (bands[6] - bands[4]) / (bands[6] + bands[4])  # NDBI = (SWIR - NIR) / (SWIR + NIR)

# **Clip the NDVI and NDBI values to a valid range [-1, 1]**
ndvi = np.clip(ndvi, -1, 1)
ndbi = np.clip(ndbi, -1, 1)

# Define refined classification thresholds for NDVI and NDBI
ndvi_thresholds = [(-1, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]
ndbi_thresholds = [(-1, 0), (0, 0.15), (0.15, 0.3), (0.3, 1)]  # Adjusted NDBI thresholds for better differentiation

# Classify NDVI and NDBI based on thresholds
ndvi_classified = classify_index(ndvi, ndvi_thresholds)
ndbi_classified = classify_index(ndbi, ndbi_thresholds)

# Plot classified NDVI with NASA-based labels
ndvi_cmap = ListedColormap(ndvi_colors)
plot_classified_map(ndvi_classified, 'Classified NDVI (NASA Standard Labels)', aoi_bounds, ndvi_cmap, ndvi_labels)

# Plot classified NDBI with refined labels and thresholds
ndbi_cmap = ListedColormap(ndbi_colors)
plot_classified_map(ndbi_classified, 'Refined Classified NDBI Visualization', aoi_bounds, ndbi_cmap, ndbi_labels)
