import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from tqdm import tqdm
import os

def calc_dist_angle_mat(info_path, out_path):
    df = pd.read_csv(info_path)
    coordinate = [eval(coord) for coord in df['coordinates']]
    lats = [float(coord[1]) for coord in coordinate]
    lons = [float(coord[0]) for coord in coordinate]
    dist_angle_mat = np.zeros((len(lons), len(lons), 2))

    for i in tqdm(range(len(lons))):
        for j in range(len(lons)):
            dist = Geodesic.WGS84.Inverse(lats[i], lons[i], lats[j], lons[j])
            dist_angle_mat[i, j, 0] = dist["s12"] / 1000.0  # distance, km
            dist_angle_mat[i, j, 1] = dist["azi1"]  # azimuth at the first point in degrees

    print(dist_angle_mat.shape)
    # print(dist_angle_mat)
    np.save(out_path, dist_angle_mat)

if __name__ == "__main__":
    raw_dir = "raw_data"
    clean_dir = "clean_data"
    dataset_name = 'yelp_la'
    output_folder = f'{clean_dir}/yelp/{dataset_name}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raw_path = f"{raw_dir}/yelp/{dataset_name}/{dataset_name}.geo"
    # clean_path = f"{clean_dir}/{dataset_name}"

    out_name = "dist_angle_mat.npy"
    out_path = f"{output_folder}/{out_name}"
    # print(raw_path,out_path)
    calc_dist_angle_mat(raw_path, out_path)

    raw_dir = "raw_data"
    clean_dir = "clean_data"
    dataset_name = 'yelp_nv'
    output_folder = f'{clean_dir}/yelp/{dataset_name}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raw_path = f"{raw_dir}/yelp/{dataset_name}/{dataset_name}.geo"
    # clean_path = f"{clean_dir}/{dataset_name}"

    out_name = "dist_angle_mat.npy"
    out_path = f"{output_folder}/{out_name}"
    # print(raw_path, out_path)
    calc_dist_angle_mat(raw_path, out_path)




