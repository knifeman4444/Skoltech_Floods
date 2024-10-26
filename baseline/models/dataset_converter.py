import numpy as np
from typing import Tuple
import os
import rasterio
import pathlib

#from models.load_elevations import load_and_add_elevation_data
from models.load_elevations import load_and_add_osm_data

BANDS_S2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
OUR_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

def from_worldfloods(image: np.ndarray, mask: np.ndarray, elevation=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert image from WorldFloods format to ours
    
    - Take needed channels
    - Remove clouds and invalid data
    """
    
    indices = [BANDS_S2.index(band) for band in OUR_BANDS]
    if elevation:
        indices.append(-1)
    
    image = image[indices, :, :]
    mask[mask == 0] = 0 # Invalid data
    mask[mask == 1] = 0 # Land
    mask[mask == 2] = 1 # Water
    mask[mask == 3] = 0 # Clouds
    
    return image, mask

def is_tile_valid(mask: np.ndarray) -> bool:
    """
    Apply to unprocessed mask
    
    Checks that there are not too many clouds, not too much invalid data
    """
    
    max_clouds = 0.1
    max_invalid = 0.1
    n_pixels = mask.size
    n_clouds = np.sum(mask == 3)
    n_invalid = np.sum(mask == 0)
    n_water = np.sum(mask == 2)
    
    return n_clouds / n_pixels < max_clouds and n_invalid / n_pixels < max_invalid and (
        n_water / n_pixels >= 0.05 and n_water / n_pixels <= 0.95
    )


def load_from_folder(folder: str, split="test", filename=None, elevation=False, osm=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image and mask from a given folder
    
    Returns unprocessed np.ndarrays with all the channels
    Mask contains 0 for invalid data, 1 for land, 2 for water, 3 for clouds 
    """
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist")
    folder = os.path.join(folder, split)
    
    folder_S2 = os.path.join(folder, "S2")
    folder_gt = os.path.join(folder, "gt")
    
    if filename is None:
        # Choose at random from folder_S2
        filename = np.random.choice(os.listdir(folder_S2))
        print(f"Chosen filename: {filename}")
    
    if not elevation and not osm:
        image = rasterio.open(os.path.join(folder_S2, filename)).read()
    if osm:
        image = load_and_add_osm_data(os.path.join(folder_S2, filename))
    else:
        # image = load_and_add_elevation_data(os.path.join(folder_S2, filename))
        raise NotImplementedError("Elevation data is not supported yet")
    mask = rasterio.open(os.path.join(folder_gt, filename)).read()[1][np.newaxis, :, :]
    
    return image, mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    image, mask = load_from_folder("/home/kuzakov-dn/other/Skoltech_Floods/dataset/WorldFloodsv2", elevation=False)
    print(image.shape, mask.shape)
    
    # Taking random NxN tile
    N = 1024
    x = np.random.randint(0, image.shape[2] - N)
    y = np.random.randint(0, image.shape[1] - N)
    s_image = image[:, y:y+N, x:x+N]
    s_mask = mask[0][y:y+N, x:x+N]
    i = 0
    while not is_tile_valid(s_mask):
        x = np.random.randint(0, image.shape[2] - N)
        y = np.random.randint(0, image.shape[1] - N)
        s_image = image[:, y:y+N, x:x+N]
        s_mask = mask[0][y:y+N, x:x+N]
        i += 1
        if i > 100:
            raise ValueError("Could not find a valid tile")
        
    image, mask = from_worldfloods(s_image, s_mask)
    print(mask.shape)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow((image[[2, 1, 0], :, :].transpose(1, 2, 0) / 3_500).clip(0, 1))
    axs[1].imshow(mask[:, :])
    #plt.show()
    plt.savefig('my_plot.png')
    
    
    