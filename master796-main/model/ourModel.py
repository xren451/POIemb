import torch
from torch import nn, scatter_add
from torch.optim import Adam
from torch.nn import functional as F
from torch_geometric.data import Data
import os
import numpy as np
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class twoFCN(nn.Module):
    def __init__(self, d_e, device):
        super(twoFCN, self).__init__()
        self.d_e = d_e
        self.fc1 = nn.Linear(2, d_e).to(device)
        self.fc2 = nn.Linear(d_e, d_e).to(device)

    def forward(self, r_ij):
        r_ij = r_ij.float()
        c_ij = F.relu(self.fc1(r_ij))
        c_ij = self.fc2(c_ij)
        return c_ij


#     encoder for freq
class FCN(nn.Module):
    def __init__(self, d_e, device):
        super(FCN, self).__init__()
        self.d_e = d_e
        self.fc1 = nn.Linear(1, d_e).to(device)
        self.fc2 = nn.Linear(d_e, d_e).to(device)

    def forward(self, r_ij):
        r_ij = r_ij.float()
        c_ij = F.relu(self.fc1(r_ij))
        c_ij = self.fc2(c_ij)
        return c_ij



class MultiHeadGATLayer(nn.Module):
    
    def __init__(self, in_features, out_features, num_heads=2, C_dim=16, device=None):
        super(MultiHeadGATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = num_heads
        self.C_dim = C_dim

        self.fc_q = nn.Linear(
            self.in_features, self.out_features * self.n_heads, bias=False)
        self.fc_k = nn.Linear(
            self.in_features, self.out_features * self.n_heads, bias=False)
        self.fc_v = nn.Linear(
            self.in_features, self.out_features * self.n_heads, bias=False)
        self.fc_c = nn.Linear(C_dim, self.out_features, bias=False)
        self.fc_f = nn.Linear(C_dim, self.out_features, bias=False)

        self.fcn = FCN(C_dim, device)

        self.spatial_encoder = twoFCN(C_dim, device)

        self.fc_out = nn.Linear(
            self.n_heads * self.out_features, self.out_features, bias=False)

    def forward(self, x, edge_index, edge_attr1, edge_attr2):
        Q = self.fc_q(x).view(-1, self.n_heads, self.out_features)
        K = self.fc_k(x).view(-1, self.n_heads, self.out_features)
        V = self.fc_v(x).view(-1, self.n_heads, self.out_features)

        edge_attr1 = self.spatial_encoder(edge_attr1)
        edge_attr1 = self.fc_c(edge_attr1.view(-1, self.C_dim))
        edge_attr1 = edge_attr1.unsqueeze(1).expand(-1, self.n_heads, -1)

        edge_attr2 = edge_attr2.unsqueeze(-1)
        edge_attr2 = self.fcn(edge_attr2)
        edge_attr2 = self.fc_f(edge_attr2.view(-1, self.C_dim))
        edge_attr2 = edge_attr2.unsqueeze(1).expand(-1, self.n_heads, -1)

        edge = edge_index.t().contiguous()
        q, k, v = Q[edge[:, 0]], K[edge[:, 1]], V[edge[:, 1]]

        print(Q.shape, K.shape, V.shape)
        print(edge_attr1.shape)
        print(edge_attr2.shape)
        print(q.shape, k.shape, v.shape)

        attn_score = (q * k * edge_attr1 * edge_attr2).sum(dim=-1) / self.out_features ** 0.5
        attn_weights = F.softmax(attn_score, dim=-1).unsqueeze(-1)

        output = attn_weights * v
        output_list = []
        for i in range(self.n_heads):
            output_head_i = torch.zeros_like(x[:, 0:self.out_features]).scatter_add(
                0, edge[:, 0].view(-1, 1).expand(-1, self.out_features), output[:, i, :])
            output_list.append(output_head_i)
        output = torch.stack(output_list, dim=1)

        output = output.view(-1, self.n_heads * self.out_features)
        output = self.fc_out(output)
        return output



class GATGraphSAGE(nn.Module):
    def __init__(self, in_features, edge_features, out_features, num_heads=2, C_dim=16, agg_hidden=768, device=None):
        super(GATGraphSAGE, self).__init__()

        self.gat_layer = MultiHeadGATLayer(
            in_features + edge_features, agg_hidden, num_heads, C_dim, device)  # Note the adjustment in input features
        self.fc = nn.Linear(agg_hidden + in_features + edge_features, out_features)  # Adjustment here too

        self.two_fcn = twoFCN(C_dim, device)
        self.fcn = FCN(C_dim, device)

    def forward(self, x, edge_index, edge_attr1, edge_attr2):
        edge_index_t = edge_index.t().contiguous()

        encoded_edge_attr1 = self.two_fcn(edge_attr1)
        encoded_edge_attr2 = self.fcn(edge_attr2.unsqueeze(-1))

        summed_edge_attr1 = torch.zeros(x.size(0), encoded_edge_attr1.size(1)).to(x.device)
        summed_edge_attr1.scatter_add_(0, edge_index_t[:, 1].unsqueeze(-1).expand(-1, encoded_edge_attr1.size(1)), encoded_edge_attr1)

        summed_edge_attr2 = torch.zeros(x.size(0), encoded_edge_attr2.size(1)).to(x.device)
        summed_edge_attr2.scatter_add_(0, edge_index_t[:, 1].unsqueeze(-1).expand(-1, encoded_edge_attr2.size(1)), encoded_edge_attr2)
        # Aggregating edge features
        summed_edge_attr = torch.cat([summed_edge_attr1, summed_edge_attr2], dim=1)
        enhanced_x = torch.cat([x, summed_edge_attr], dim=-1)

        # Aggregating neighbor information
        aggregated = self.gat_layer(enhanced_x, edge_index, edge_attr1, edge_attr2)

        # Concatenate aggregated information with own enhanced features and aggregated edge features
        combined = torch.cat([enhanced_x, aggregated], dim=-1)

        # Apply fully connected layer
        out = self.fc(combined)
        return out


class Decoder(nn.Module):
    def __init__(self, in_features, out_features_text, out_features_spatial):
        super(Decoder, self).__init__()
        self.text_decoder = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, out_features_text),
            nn.Sigmoid()
        )
        self.spatial_decoder = nn.Sequential(
            nn.Linear(in_features * 2, in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features_spatial)
        )

    def forward_text(self, node_embedding):
        text_features = self.text_decoder(node_embedding)
        return text_features

    def forward_spatial(self, node_i_embedding, node_j_embedding):
        concatenated_embeddings = torch.cat(
            [node_i_embedding, node_j_embedding], dim=-1)
        spatial_features = self.spatial_decoder(concatenated_embeddings)
        return spatial_features

class OccDecoder(nn.Module):
    def __init__(self, in_features):
        super(OccDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            nn.ReLU(),
            nn.Linear(in_features//2, 1)  # 输出一个值，即occ
        )

    def forward(self, node_i_embedding, node_j_embedding):
        concatenated_embeddings = torch.cat([node_i_embedding, node_j_embedding], dim=-1)
        occ_prediction = self.decoder(concatenated_embeddings)
        return occ_prediction


def train_one_epoch(
        model, text_decoder, spatial_decoder,occ_decoder, data,
        relative_position_tensor, freq_matrix_tensor, one_hot_targets, optimizer, log_vars_optimizer
):
    model.train()
    text_decoder.train()
    spatial_decoder.train()
    occ_decoder.train()
    # Forward pass through GAT
    node_embeddings = model(data.x, data.edge_index, data.edge_position_attr, data.edge_frequency_attr)
    edge_i = node_embeddings[data.edge_index[0]]
    edge_j = node_embeddings[data.edge_index[1]]

    # Forward pass through decoder
    predicted_text_features = text_decoder.forward_text(node_embeddings)
    predicted_spatial_features = spatial_decoder.forward_spatial(edge_i, edge_j)

    # Extract true spatial features
    edge_indices = torch.stack([data.edge_index[0], data.edge_index[1]], dim=0)
    true_spatial_features = relative_position_tensor[edge_indices[0], edge_indices[1]]

    true_occ_features = freq_matrix_tensor[edge_indices[0], edge_indices[1]]
    predicted_occ_values = occ_decoder(edge_i, edge_j).squeeze()  # 预测的occ值
    occ_loss = F.mse_loss(predicted_occ_values, true_occ_features)

    # Compute loss
    text_loss = F.binary_cross_entropy(predicted_text_features, one_hot_targets)
    spatial_loss = (predicted_spatial_features - true_spatial_features).pow(2).mean(dim=0)

    mse_distance, mse_azimuth = spatial_loss

    # Compute weighted losses
    weighted_text_loss = torch.exp(-log_vars[0]) * text_loss + log_vars[0]
    weighted_spatial_loss = torch.exp(-log_vars[1]) * spatial_loss.sum() + log_vars[1]

    weighted_occ_loss = torch.exp(-log_vars[2]) * occ_loss + log_vars[2]
    total_loss = weighted_text_loss + weighted_spatial_loss + weighted_occ_loss

    # Optimization step
    optimizer.zero_grad()
    log_vars_optimizer.zero_grad()

    total_loss.backward()

    optimizer.step()
    log_vars_optimizer.step()

    return total_loss.item(), mse_distance.item(), mse_azimuth.item(), occ_loss.item(), predicted_text_features, node_embeddings


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = 'yelp_la'

    clean_dir = f'../clean_data/yelp/{dataset_name}'
    input_path = os.path.join(clean_dir, 'dataLoader', 'data_15.pt')

    loaded_data_dict = torch.load(input_path)
    data = loaded_data_dict["graph_data"].to(device)
    one_hot_targets = loaded_data_dict["one_hot_targets"].to(device)
    relative_position_tensor = loaded_data_dict["relative_position_tensor"].to(device)

    freq_matrix_file_path = os.path.join(clean_dir, 'occ_mat.npy')
    freq_matrix = np.load(freq_matrix_file_path)
    freq_matrix_tensor = torch.tensor(freq_matrix, device=device).float()

    output_folder = f'{clean_dir}/embeddings'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    out_name = "emb_decode_15_epoch100.pth"
    out_path = f"{output_folder}/{dataset_name}_{out_name}"

    emb_size = 500
    vocab_size = 30522
    num_epoch = 100
    edge_feature_size = 32
    model = GATGraphSAGE(data.x.size(1), edge_feature_size, emb_size, device=device).to(device)
    text_decoder = Decoder(emb_size, vocab_size, 0).to(device)
    spatial_decoder = Decoder(emb_size, 0, 2).to(device)
    occ_decoder = OccDecoder(emb_size* 2).to(device)

    optimizer = Adam(
        list(model.parameters()) + list(text_decoder.parameters()) + list(spatial_decoder.parameters()),
        lr=0.001
    )
    log_vars = nn.Parameter(torch.zeros(3), requires_grad=True)
    log_vars_optimizer = Adam([log_vars], lr=0.001)

    for epoch in tqdm(range(num_epoch)):
        loss, mse_distance, mse_azimuth,mse_occ,predicted_text_features,node_embeddings = train_one_epoch(
            model, text_decoder, spatial_decoder, occ_decoder, data,
            relative_position_tensor,freq_matrix_tensor, one_hot_targets, optimizer, log_vars_optimizer
        )
        print(f'Epoch [{epoch + 1}/50], Loss: {loss:.4f}, Distance: {mse_distance:.4f}, Azimuth: {mse_azimuth:.4f}, Occ: {mse_occ:.4f}')

    torch.save(model.state_dict(), f'pretrained_model_state_{dataset_name}.pth')
    model.eval()
    with torch.no_grad():
        poi_emb = model(data.x, data.edge_index, data.edge_position_attr, data.edge_frequency_attr)
    torch.save(poi_emb,out_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        predicted_text_features = text_decoder.forward_text(poi_emb)
    probs = torch.nn.functional.softmax(predicted_text_features, dim=0)
    top_k_values, top_k_indices = torch.topk(probs, 5)  # 取top 5
    top_k_tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in top_k_indices.tolist()]
    for i in range(5):
        print(f"Top {i + 1} words: {top_k_tokens[i]}")

    """
    pre-train cross city
    """
    
    if dataset_name == 'foursquare_tky':
        pretrained_dataset_name = 'foursquare_nyc'
        clean_dir = f'../clean_data/{pretrained_dataset_name}/filtered'
    if dataset_name == 'foursquare_nyc':
        pretrained_dataset_name = 'foursquare_tky'
        clean_dir = f'../clean_data/{pretrained_dataset_name}/filtered'
    if dataset_name == 'yelp_la':
        pretrained_dataset_name = 'yelp_nv'
        clean_dir = f'../clean_data/yelp/{pretrained_dataset_name}'
    if dataset_name == 'yelp_nv':
        pretrained_dataset_name = 'yelp_la'
        clean_dir = f'../clean_data/yelp/{pretrained_dataset_name}'

    input_path = os.path.join(clean_dir, 'dataLoader', 'data.pt')
    loaded_data_dict = torch.load(input_path)
    pre_trained_data = loaded_data_dict["graph_data"].to(device)

    model = GATGraphSAGE(pre_trained_data.x.size(1), edge_feature_size, emb_size, device=device)
    model.load_state_dict(torch.load(f'pretrained_model_state_{dataset_name}.pth'))
    model.eval() 
    model = model.to(device)

    with torch.no_grad():
        embeddings = model(pre_trained_data.x, pre_trained_data.edge_index, pre_trained_data.edge_position_attr, pre_trained_data.edge_frequency_attr)
    print(embeddings.shape)
    print(embeddings)

    output_folder = f'{clean_dir}/embeddings'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    out_name = "emb_decode_from_pre_trained_15_epoch100.pth"
    out_path = f"{output_folder}/{pretrained_dataset_name}_{out_name}"
    torch.save(embeddings, out_path)