import os
import rasterio
import argparse
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely import affinity
from shapely import Point

from sklearn.metrics import f1_score


def flooded_houses(
    osm_path: str, 
    lats: np.ndarray, 
    lons: np.ndarray, 
    pred: np.ndarray, 
    ground_truth: np.ndarray 
):
    gdf = gpd.read_file(osm_path)
    gdf = gdf.to_crs(4326)
    gdf.tags.unique()

    flooded_pred = []
    flooded_gt = []
    pred = pred.flatten()  # Flatten the prediction array
    ground_truth = ground_truth.flatten()  # Flatten the ground_truth array
    
    for _, row in gdf.iterrows():
        polygon = row.geometry
        # Scale the polygon for more accurate coverage
        scaled_polygon = affinity.scale(polygon, xfact=1.5, yfact=1.5)
        
        # Get the polygon's bounding box (xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = scaled_polygon.bounds

        # Find the indices of points that fall inside the bounding box of the polygon
        selected_indices = np.where((ymin <= lats) & (lats <= ymax) & (xmin <= lons) & (lons <= xmax))
        
        lats_to_check = lats[selected_indices]
        lons_to_check = lons[selected_indices]
        flood_pred_to_check = pred[selected_indices]
        flood_gt_to_check = ground_truth[selected_indices]

        # Check if at least one point inside the polygon is flooded in the prediction mask
        is_flooded_pred = any(
            flood_pred_to_check[i] and scaled_polygon.contains(Point(lons_to_check[i], lats_to_check[i]))
            for i in range(len(flood_pred_to_check))
        )

        # Check if at least one point inside the polygon is flooded in the ground truth mask
        is_flooded_gt = any(
            flood_gt_to_check[i] and scaled_polygon.contains(Point(lons_to_check[i], lats_to_check[i]))
            for i in range(len(flood_gt_to_check))
        )

        flooded_pred.append(1 if is_flooded_pred else 0)
        flooded_gt.append(1 if is_flooded_gt else 0)

    return f1_score(flooded_gt, flooded_pred, average='macro')


def calculate_f1_score(masks):
    """Calculate the F1 score between corresponding images in the lists."""
    f1_scores = []
    
    for mask1, mask2 in masks:
        # Calculate F1 score
        f1 = f1_score(mask1, mask2, average='macro')
        f1_scores.append(f1)
    
    # Calculate average F1 score across all image pairs
    average_f1 = np.mean(f1_scores)
    return average_f1


def get_score(masks, osm_path):

    assert masks[-1][0].shape == (512, 512), "last image should be 9_2"
    assert masks[-1][1].shape == (512, 512), "last image should be 9_2"
    assert masks[-2][0].shape == (512, 512), "the penultimate one should be 9_1"
    assert masks[-2][1].shape == (512, 512), "the penultimate one should be 9_1"
    assert masks[-2][0].sum() < masks[-1][0].sum(), "the right order"

    f1_water = calculate_f1_score(masks)

    pre_lats = np.load('bad_val_saves/pre_lats.npy')
    pre_lons = np.load('bad_val_saves/pre_lons.npy')
    post_lats = np.load('bad_val_saves/post_lats.npy')
    post_lons = np.load('bad_val_saves/post_lons.npy')

    pre_f1 = flooded_houses(osm_path, pre_lats, pre_lons, masks[-2][1], masks[-2][0])
    post_f1 = flooded_houses(osm_path, post_lats, post_lons, masks[-1][1], masks[-1][0])
    avg_f1_business = (pre_f1 + post_f1) / 2
    total_f1 = (f1_water + avg_f1_business) / 2

    return {'total_f1': total_f1, 'f1_water': f1_water, 'avg_f1_business': avg_f1_business, 'pre_f1': pre_f1, 'post_f1': post_f1}
