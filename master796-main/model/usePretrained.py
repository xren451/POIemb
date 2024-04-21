import torch
from torch import nn, scatter_add
from torch.optim import Adam
from torch.nn import functional as F
from torch_geometric.data import Data
import os

from torch_geometric.utils import add_self_loops
from tqdm import tqdm
from transformers import BertTokenizer
from model.trainModelPredict import GATGraphSAGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = 'foursquare_nyc'
clean_dir = f'../clean_data/{dataset_name}/filtered'
input_path = os.path.join(clean_dir, 'dataLoader', 'data.pt')
loaded_data_dict = torch.load(input_path)
data = loaded_data_dict["graph_data"].to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emb_size = 500
vocab_size = 30522
num_epoch = 50
edge_feature_size = 2
model = GATGraphSAGE(data.x.size(1), edge_feature_size, emb_size, device=device)

model.load_state_dict(torch.load('pretrained_model_state_foursquare_tky.pth'))
model.eval()  # 设置为评估模式
model = model.to(device)
with torch.no_grad():
    embeddings = model(data.x, data.edge_index, data.edge_position_attr, data.edge_frequency_attr)

print(embeddings.shape)
print(embeddings)

output_folder = f'{clean_dir}/embeddings'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
out_name = "emb_from_pre_trained.pth"
out_path = f"{output_folder}/{dataset_name}_{out_name}"

torch.save(embeddings, out_path)