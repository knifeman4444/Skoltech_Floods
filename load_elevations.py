import rasterio
import elevation
from rasterio.enums import Resampling
import os
import argparse
import glob

"""
Requires "elevation" package to be installed

It, in turn, requires "gdal" to be installed on the system
Instructions for macOS: https://mits003.github.io/studio_null/2021/07/install-gdal-on-macos/
"""
        
def add_elevation_data(image_path: str, output_path: str):
    """
    Adds elevation data to a given image and saves the result.
    
    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the image with elevation data.
    Raises:
        ValueError: If the input image does not have exactly 10 channels
        (for example if you already added elevation data to it, but forgot)
    """
    
    with rasterio.open(image_path) as src_image:
        bounds = src_image.bounds
        
        # Check that image has exactly 10 channels
        if src_image.count != 10:
            raise ValueError("Image to add elevation data should have exactly 10 channels")
        
        src_image_data = src_image.read(indexes=1)
        
        # Download elevation data to a temporary file
        elevation.clip(bounds, output='/tmp/.elevation.tif')
        
        # Resample to needed resolution
        with rasterio.open('/tmp/.elevation.tif') as src_elevation:
            elevation_resampled = src_elevation.read(
                out_shape=(1, src_image_data.shape[0], src_image_data.shape[1]),
                resampling=Resampling.bilinear
            )
            
        elevation_resampled = elevation_resampled.astype(src_image_data.dtype)
        elevation_resampled = elevation_resampled[0]
            
        profile = src_image.profile
        profile.update(count=11)
        
        # Save
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(10):
                dst.write(src_image.read(indexes=i+1), i+1)
            dst.write(elevation_resampled, 11)
            
        # Remove temporary file
        os.remove('/tmp/.elevation.tif')


def process_images(input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if input_dir == output_dir:
        raise ValueError("Input and output directories are the same")

    for image_path in glob.glob(os.path.join(input_dir, '*.tif')):
        print(f"Processing {image_path}")
        
        image_name = os.path.basename(image_path)
        elevation_output_path = os.path.join(output_dir, image_name)
        add_elevation_data(image_path, elevation_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download elevation data")
    parser.add_argument('input_dir', type=str, help="Directory with reference images")
    parser.add_argument('output_dir', type=str, help="Directory to save elevation profiles")
    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir)