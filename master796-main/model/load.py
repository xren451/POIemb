import os
from collections import defaultdict

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import networkx as nx
from tqdm import tqdm
from torch_geometric.data import Data, NeighborSampler
import gc

# Configurations


# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_sequences(df):
    df['seq'] = df[test_column_name].apply(lambda x: tokenizer.encode(x, truncation=True, max_length=512))
    target_seqs = [torch.tensor(seq[1:-1]).to(device) for seq in df['seq'].values]
    return target_seqs


def one_hot_encoding(target_seqs):
    one_hot_targets = torch.zeros(len(target_seqs), tokenizer.vocab_size).to(device)
    for i, seq in tqdm(enumerate(target_seqs)):
        if seq.nelement() != 0:
            one_hot_targets[i, seq] = 1
    return one_hot_targets


def load_numpy_data(filepath):
    with open(filepath, "rb") as f:
        return np.load(f)


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

def compute_stats(matrix, axis=(0, 1)):
    """
    Compute mean, std, max, and min of the matrix along the given axis.
    """
    return {
        "mean": np.mean(matrix, axis=axis),
        "std": np.std(matrix, axis=axis),
        "max": np.max(matrix, axis=axis),
        "min": np.min(matrix, axis=axis)
    }


def standardize_matrix(matrix):
    """
    Standardize the matrix using mean and std.
    """
    stats = compute_stats(matrix)
    return (matrix - stats["mean"]) / stats["std"]


def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()


def construct_graph(position_matrix, freq_matrix_tensor, text_features, num_neighbors=5):
    G = nx.DiGraph()
    num_nodes = len(position_matrix)
    position_matrix_np = position_matrix.cpu().numpy()

    for i in tqdm(range(num_nodes)):
        G.add_node(i, text_features=text_features[i].cpu().numpy())
        distances_i = np.argsort(position_matrix_np[i, :, 0])[:num_neighbors + 1]
        neighbors = [node for node in distances_i if node != i][:num_neighbors]
        r_ij_batch = position_matrix[i, neighbors]
        f_ij_batch = freq_matrix_tensor[i, neighbors]
        for j, neighbor in enumerate(neighbors):
            G.add_edge(i, neighbor, p_feature=r_ij_batch[j], f_feature=f_ij_batch[j])
    return G


def get_file_paths(dataset_name):
    if dataset_name in ['foursquare_nyc', 'foursquare_tky']:
        base_path = os.path.join('..', 'clean_data', dataset_name)
        geo_path = os.path.join('..', 'raw_data', dataset_name, 'filtered', f"{dataset_name}.geo")
        test_column = 'venue_category_name'
    elif dataset_name in ['yelp_la', 'yelp_nv']:
        base_path = os.path.join('..', 'clean_data', 'yelp', dataset_name)
        geo_path = os.path.join('..', 'raw_data', 'yelp', dataset_name, f"{dataset_name}.geo")
        test_column = 'combined_text'  # Here, use the combined column
    elif dataset_name in ['instagram']:
        base_path = os.path.join('..', 'clean_data', dataset_name)
        geo_path = os.path.join('..', 'raw_data', dataset_name, f"{dataset_name}.geo")
        test_column = 'combined_text'  # Here, use the combined column
    else:
        raise ValueError(f"Unrecognized dataset_name: {dataset_name}")

    relative_position_file_path = os.path.join(base_path, 'dist_angle_mat.npy')
    freq_matrix_file_path = os.path.join(base_path, 'occ_mat.npy')
    output_folder = os.path.join(base_path, 'dataLoader')

    return relative_position_file_path, freq_matrix_file_path, geo_path, output_folder, test_column


def main():
    # Prepare data
    target_seqs = get_sequences(df_geo)
    one_hot_targets = one_hot_encoding(target_seqs)
    relative_position = load_numpy_data(relative_position_file_path)
    freq_matrix = load_numpy_data(freq_matrix_file_path)
    relative_position = standardize_matrix(relative_position)
    testtxt = df_geo[test_column_name]
    txt_embedding = testtxt.apply(get_bert_embeddings)
    txt_embedding = np.vstack(txt_embedding)

    # Convert data to tensors
    relative_position_tensor = torch.tensor(relative_position, device=device)
    freq_matrix_tensor = torch.tensor(freq_matrix, device=device)
    txt_embedding_tensor = torch.tensor(txt_embedding, device=device)

    # Construct graph
    G_optimized = construct_graph(relative_position_tensor, freq_matrix_tensor, txt_embedding_tensor, num_neighbors)
    torch.cuda.empty_cache()
    gc.collect()
    # Convert node and edge features for PyTorch Geometric
    x = [G_optimized.nodes[node]['text_features'] for node in G_optimized.nodes()]
    x = torch.tensor(np.stack(x), dtype=torch.float32)
    edge_index, edge_position_attr, edge_frequency_attr = [], [], []
    for u, v, data in G_optimized.edges(data=True):
        edge_index.append([u, v])
        edge_position_attr.append(data['p_feature'].cpu())
        edge_frequency_attr.append(data['f_feature'].cpu())

    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.int64)
    edge_position_attr = torch.tensor(np.stack(edge_position_attr), dtype=torch.float32).to(device)
    edge_frequency_attr = torch.stack(edge_frequency_attr).to(device)

    # edge_frequency_attr = torch.tensor(np.stack(edge_frequency_attr), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_position_attr=edge_position_attr,
                edge_frequency_attr=edge_frequency_attr)
    return data, one_hot_targets, relative_position_tensor

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_neighbors = 5
    dataset_name = 'yelp_la'

    relative_position_file_path, freq_matrix_file_path, geo_path, output_folder, test_column_name = get_file_paths(dataset_name)
    df_geo = pd.read_csv(geo_path)
    # If yelp dataset, concatenate 'category' and 'text'
    if dataset_name in ['yelp_la', 'yelp_nv']:
        df_geo['combined_text'] = df_geo['categories'].astype(str) + ' ' + df_geo['text'].astype(str)
    if dataset_name in 'instagram':
        df_geo['combined_text'] = df_geo['poi_name'].astype(str) + ' ' + df_geo['text'].astype(str)

    processed_data, one_hot_targets, relative_position_tensor = main()
    data_to_save = {
        "graph_data": processed_data,
        "one_hot_targets": one_hot_targets,
        "relative_position_tensor": relative_position_tensor
    }

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    out_name = "data.pt"
    out_path = f"{output_folder}/{out_name}"
    torch.save(data_to_save, out_path)
    print('preprocessed data finished----')
