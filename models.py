import torch
import torch.nn as nn
import os
from torch.optim import SGD

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        device,
        dropout=0.0,
        num_hidden=2,
        model_name="mlp"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_hidden = num_hidden
        self.linear_layers_list = nn.ModuleList()
        self.linear_layers_list.append(nn.Linear(input_size, hidden_size, bias=False).to(device))
        for _ in range(1, self.num_hidden-1):
            self.linear_layers_list.append(nn.Linear(hidden_size, hidden_size, bias=False).to(device))
        self.linear_layers_list.append(nn.Linear(hidden_size, output_size, bias=False).to(device))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def load_model_from(self, path, device):
        self.load_state_dict(torch.load(path))
        self.to(device)
        self.eval()

    def save(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        model_path = os.path.join(output_dir, os.path.basename(output_dir) + ".pth")
        device = self.linear_layers_list[0].weight.device # hacky way to get the device
        self.to("cpu")
        torch.save(self.state_dict(), model_path)
        self.to(device)
        
    def forward(self, x, adj_hat = None, **kwargs):
        for i in range(self.num_hidden-1):
            x = self.linear_layers_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.linear_layers_list[self.num_hidden-1](x)
        return x        


class MLPPool(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_hidden, device, dropout=0.0, model_name="mlppool"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_hidden = num_hidden
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_layers_list = nn.ModuleList()
        self.linear_layers_list.append(nn.Linear(input_size, hidden_size, bias=False).to(device))  
        
        for _ in range(1, self.num_hidden - 1):
            self.linear_layers_list.append(nn.Linear(hidden_size, hidden_size, bias=False).to(device))
        
        self.linear_layers_list.append(nn.Linear(hidden_size, output_size, bias=False).to(device))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.device = device
    
    def load_model_from(self, path, device):
        self.load_state_dict(torch.load(path))
        self.to(device)
        self.eval()
        
    def save(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        model_path = os.path.join(output_dir, os.path.basename(output_dir) + ".pth")
        device = self.linear_layers_list[0].weight.device # hacky way to get the device
        self.to("cpu")
        torch.save(self.state_dict(), model_path)
        self.to(device)   
    
    def forward(self, x, adj_hat=None, **kwargs):
        x = self.linear_layers_list[0](x)
        x = torch.sparse.mm(adj_hat, x)
        x = self.relu(x)
        x = self.dropout(x)
        
        for i in range(1, self.num_hidden-1):
            x = self.linear_layers_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.linear_layers_list[-1](x)
        return x

class GCN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_hidden, device, dropout=0.0, model_name="gcn"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_hidden = num_hidden
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_layers_list = nn.ModuleList()
        self.linear_layers_list.append(nn.Linear(input_size, hidden_size, bias=False).to(device))
        
        for _ in range(1, self.num_hidden - 1):
            self.linear_layers_list.append(nn.Linear(hidden_size, hidden_size, bias=False).to(device))
        
        self.linear_layers_list.append(nn.Linear(hidden_size, output_size, bias=False).to(device))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.device = device

    def load_model_from(self, path, device):
        self.load_state_dict(torch.load(path))
        self.to(device)
        self.eval()
        
    def save(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        model_path = os.path.join(output_dir, os.path.basename(output_dir) + ".pth")
        device = self.linear_layers_list[0].weight.device # hacky way to get the device
        self.to("cpu")
        torch.save(self.state_dict(), model_path)
        self.to(device)            
        
    def forward(self, x, adjacency_hat):
        x = x.to(self.device)
        
        for i in range(self.num_hidden-1):
            x = self.linear_layers_list[i](x)
            x = torch.sparse.mm(adjacency_hat, x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.linear_layers_list[self.num_hidden-1](x)
        x = torch.sparse.mm(adjacency_hat, x)
        
        return x
        
def train(model, data, device, optimizer, epochs):
    model = model.to(device)
    x = data["x"]
    y = data["y"]
    train_mask = data["train_mask"]
    adj_hat = data["adj"]
    
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        y_train = y[train_mask]
        y_hat = model(x, adj_hat)[train_mask]
        loss = nn.CrossEntropyLoss()(y_hat, y_train)
        loss.backward()
        optimizer.step()
        
def test(model, data, device):
    model = model.to(device)
    model.eval()
    
    x = data["x"]
    y = data["y"]
    val_mask = data["val_mask"]
    adj_hat = data["adj"]        
    y_val = y[val_mask]
    y_hat = model(x, adj_hat)[val_mask]
    loss = nn.CrossEntropyLoss()(y_hat, y_val)
    correct = int((y_hat.argmax(dim=-1) == y_val).sum())
    total_examples = val_mask.sum().item()
    
    loss = float(loss)
    acc = correct / total_examples
    
    return loss, acc

def get_optimizer(model, optimizer_name, learning_rate, momentum=0.9, weight_decay=5e-4):
    if optimizer_name == "sgd":
        return SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    raise ValueError("Unknown optimizer")