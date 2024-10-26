import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from rasterio.features import rasterize

df = gpd.read_file('dataset/train/osm/9.geojson')

df = df.to_crs(4326)
df.tags.unique()


def transform_arr(arr):
    """From torch format to pyplot format"""
    return np.transpose(arr[::-1], (1, 2, 0))


def load_raster(src):
    """Load a raster image"""
    data = src.read([1, 2, 3])
    return transform_arr(data / 1500)


def load_mask(path):
    """Load a mask"""
    with rasterio.open(path) as src:
        data = src.read(1)
        return data


def get_mask(gj, tif_image):
    return rasterize(
        [(geom, 1) for geom in gj.geometry],
        out_shape=(tif_image.height, tif_image.width),
        transform=tif_image.transform,
        all_touched=True,
        fill=0,
        default_value=1
    )


fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

image = rasterio.open('dataset/train/images/9_1.tif')
before_gt_mask = load_mask('dataset/train/masks/9_1.tif')
after_gt_mask = load_mask('dataset/train/masks/9_2.tif')

house_mask = get_mask(df, image)

raster = load_raster(image)

masks = np.array(house_mask, )

ax.imshow(
    0.5 * np.transpose(
        np.stack(
            [house_mask,
             after_gt_mask,
             before_gt_mask],
            axis=0),
        (1, 2, 0)
    )
    + 0.5 * raster)

fig.savefig('figure.png', bbox_inches='tight')
