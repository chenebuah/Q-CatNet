## CONTRIBUTORS: * Ericsson Chenebuah, Michel Nganbe, David Liu and Alain Tchagang
# Department of Mechanical Engineering, University of Ottawa, 75 Laurier Ave. East, Ottawa, ON, K1N 6N5 Canada
# Digital Technologies Research Centre, National Research Council of Canada, 1200 Montréal Road, Ottawa, ON, K1A 0R6 Canada
# * email: echen013@uottawa.ca

# Saved Pytorch Models can be downloaded from the "Saved Models Folder".

#! pip install torch_geometric
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_geometric.nn import CGConv, global_mean_pool

# Model 1: Density of States Predictive Model
class CGCNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, target_size):
        super(CGCNN, self).__init__()
        self.conv1 = CGConv(num_node_features, num_edge_features)
        self.conv2 = CGConv(num_node_features, num_edge_features)
        self.conv3 = CGConv(num_node_features, num_edge_features)
        self.fc1 = torch.nn.Linear(num_node_features, num_node_features)
        self.fc2 = torch.nn.Linear(num_node_features+180, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(4096, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dropout3 = nn.Dropout(0.2)
        self.fc7 = torch.nn.Linear(2048, target_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        x, edge_index, edge_attr, glob_attr = data.x, data.edge_index, data.edge_attr, data.glob_attr
        x = F.silu(self.conv1(x, edge_index, edge_attr))
        x = F.silu(self.conv2(x, edge_index, edge_attr))
        x = F.silu(self.conv3(x, edge_index, edge_attr))
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc1(x))
        x = global_mean_pool(x, data.batch)
        x = torch.cat([x, glob_attr], dim=1)
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = self.fc7(x)
        x = self.sigmoid(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dos = CGCNN(num_node_features=230, num_edge_features=60, target_size=1024).to(device)
model_dos.load_state_dict(torch.load("dos_model.pth", map_location=torch.device('cpu')))

# Model 2: Adsorbate Binding Energy Predicitive Model
class CGCNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, target_size):
        super(CGCNN, self).__init__()
        self.conv1 = CGConv(num_node_features, num_edge_features)
        self.conv2 = CGConv(num_node_features, num_edge_features)
        self.conv3 = CGConv(num_node_features, num_edge_features)
        self.fc1 = torch.nn.Linear(num_node_features, num_node_features)
        self.fc2 = torch.nn.Linear(num_node_features+1204, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = torch.nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = torch.nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.2)
        self.fc6 = torch.nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.dropout6 = nn.Dropout(0.2)
        self.fc7 = torch.nn.Linear(64, target_size)

    def forward(self, data):
        x, edge_index, edge_attr, glob_attr = data.x, data.edge_index, data.edge_attr, data.glob_attr
        x = F.silu(self.conv1(x, edge_index, edge_attr))
        x = F.silu(self.conv2(x, edge_index, edge_attr))
        x = F.silu(self.conv3(x, edge_index, edge_attr))
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc1(x))
        x = global_mean_pool(x, data.batch)
        glob_attr = glob_attr.reshape(data.y.size(0), 1204)
        x = torch.cat([x, glob_attr], dim=1)
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = F.silu(self.fc4(x))
        x = F.silu(self.fc5(x))
        x = F.silu(self.fc6(x))
        x = self.fc7(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ads = CGCNN(num_node_features=230, num_edge_features=60, target_size=1).to(device)
model_ads.load_state_dict(torch.load("ads_model.pth", map_location=torch.device('cpu')))

# Model 3: DFT Energies Predicitve Model
class CGCNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, target_size):
        super(CGCNN, self).__init__()
        self.conv1 = CGConv(num_node_features, num_edge_features)
        self.conv2 = CGConv(num_node_features, num_edge_features)
        self.conv3 = CGConv(num_node_features, num_edge_features)
        self.fc1 = torch.nn.Linear(num_node_features, num_node_features)
        self.fc2 = torch.nn.Linear(num_node_features+180, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = torch.nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = torch.nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.2)
        self.fc6 = torch.nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.dropout6 = nn.Dropout(0.2)
        self.fc7 = torch.nn.Linear(64, target_size)

    def forward(self, data):
        x, edge_index, edge_attr, glob_attr = data.x, data.edge_index, data.edge_attr, data.glob_attr
        x = F.silu(self.conv1(x, edge_index, edge_attr))
        x = F.silu(self.conv2(x, edge_index, edge_attr))
        x = F.silu(self.conv3(x, edge_index, edge_attr))
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc1(x))
        x = global_mean_pool(x, data.batch)
        glob_attr = glob_attr.reshape(1, 180)
        x = torch.cat([x, glob_attr], dim=1)
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = F.silu(self.fc4(x))
        x = F.silu(self.fc5(x))
        x = F.silu(self.fc6(x))
        x = self.fc7(x)
        return x

# NOTE !! Saved Pytorch Models can be downloaded from the "Saved Models Folder"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_Ef = CGCNN(num_node_features=230, num_edge_features=60, target_size=1).to(device)
model_Ef.load_state_dict(torch.load("Ef_model.pth", map_location=torch.device('cpu')))

model_Ehull = CGCNN(num_node_features=230, num_edge_features=60, target_size=1).to(device)
model_Ehull.load_state_dict(torch.load("Ehull_model.pth", map_location=torch.device('cpu')))

model_Eg = CGCNN(num_node_features=230, num_edge_features=60, target_size=1).to(device)
model_Eg.load_state_dict(torch.load("Eg_model.pth", map_location=torch.device('cpu')))

model_Etot = CGCNN(num_node_features=230, num_edge_features=60, target_size=1).to(device)
model_Etot.load_state_dict(torch.load("Etot_model.pth", map_location=torch.device('cpu')))
