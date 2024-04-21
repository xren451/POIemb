import os
from collections import defaultdict

import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from tqdm import tqdm


def filter_location_checkin(df, threshold=5):
    venue_counts = df['location'].value_counts()
    filtered_venues = venue_counts[venue_counts >= threshold].index
    df = df[df['location'].isin(filtered_venues)]
    return df


def frequency_of_occurrence(df, hours_gap=None):
    """
    1 前 3 后
    matrix[1][3]
    """

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by=['entity_id', 'time'])
    if hours_gap:
        df['sequence_gap'] = df.groupby('entity_id')['time'].diff().gt(pd.Timedelta(hours=hours_gap))

        df['sequence_id'] = df.groupby('entity_id')['sequence_gap'].cumsum().fillna(0)
    else:
        df['sequence_id'] = 0  # If no hours_gap, use a single sequence for each user

    pairwise_counts = defaultdict(int)
    grouped = df.groupby(['entity_id', 'sequence_id'])['location'].apply(list).reset_index()
    for _, row in grouped.iterrows():
        venues_list = row['location']

        for i, v1 in enumerate(venues_list):
            for j, v2 in enumerate(venues_list):
                if i < j:  # To ensure we only count when v1 appears before v2
                    pairwise_counts[(v1, v2)] += 1
    venues = df['location'].unique()
    n = len(venues)
    matrix = np.zeros((n, n))
    for (i, j), count in pairwise_counts.items():
        fraction = count / len(grouped)
        matrix[i, j] = fraction  # Asymmetric matrix


    return matrix




def generate_freq_matrix(raw_path,out_path):
    df = pd.read_csv(raw_path)
    # checkin_threshold = 5
    # dyna_data_filtered = filter_location_checkin(dyna_data, threshold=checkin_threshold)
    freq_matrix = frequency_of_occurrence(df, hours_gap=48)
    np.save(out_path, freq_matrix)
    print(freq_matrix.shape)

    return freq_matrix

# generate_freq_matrix(dyna_data)

def calc_dist_angle_mat(df, out_path):

    lons, lats = df["Long"].values, df["Lat"].values
    dist_angle_mat = np.zeros((len(lons), len(lons), 2))

    for i in tqdm(range(len(lons))):
        for j in range(len(lons)):
            dist = Geodesic.WGS84.Inverse(lats[i], lons[i], lats[j], lons[j])
            dist_angle_mat[i, j, 0] = dist["s12"] / 1000.0  # distance, km
            dist_angle_mat[i, j, 1] = dist["azi1"]  # azimuth at the first point in degrees

    print(dist_angle_mat.shape)
    # print(dist_angle_mat)
    np.save(out_path, dist_angle_mat)
    return dist_angle_mat

# out_path = 'clean_data/foursquare_nyc/checkin0hourgap48/relativePosition_matrix.npy'
# dist_angle_matrix = calc_dist_angle_mat(geo_data, out_path)


if __name__ == "__main__":
    raw_dir = "raw_data/yelp"
    clean_dir = "clean_data/yelp"
    dataset_name = 'yelp_nv'
    output_folder = f'{clean_dir}/{dataset_name}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raw_path = f"{raw_dir}/{dataset_name}/{dataset_name}.dyna"
    # clean_path = f"{clean_dir}/{dataset_name}"

    out_name = "occ_mat.npy"
    out_path = f"{output_folder}/{out_name}"
    # print(raw_path,out_path)
    generate_freq_matrix(raw_path, out_path)