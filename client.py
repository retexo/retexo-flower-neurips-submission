import os
import torch
import torch.nn as nn 
from torch.optim import SGD
from collections import OrderedDict
from argparse import ArgumentParser
from load_data import load_data, node_indices
from models import MLP, MLPPool, train, test, get_optimizer
from custom_btserver import CustomBluetoothServer
from custom_btclient import CustomBluetoothClient
import flwr as fl   
import numpy as np
import pickle as pkl
import time
import pdb
import struct
import tracemalloc

class GNNClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        cid,
        data, 
        epochs,
        device,
        total_rounds=400,
        is_bluetooth_server=False,
        bluetooth_server_name=None
    ):
        self.cid = cid
        self.data = data
        self.device = device
        self.epochs = epochs
        self.round = 0
        self.total_rounds = total_rounds
        self.indices_received = None
        self.indices_sent = None
        self.logits_received = None
        self.logits_sent = None
        self.logits = None
        self.bluetooth_server = None 
        self.bluetooth_client = None
      
        def data_received_server(data):
            deserialized_a = pkl.loads(data)
            print("received data over bluetooth")
            
            command = deserialized_a["comm"]
            
            if command == 0:  
                self.indices_received = deserialized_a["data"]    
                serialized_b = pkl.dumps({"data": self.indices_sent, "comm": 0})
                serialized_b = struct.pack('>I', len(serialized_b)) + serialized_b
                self.bluetooth_server.send(serialized_b)
                
                self.logits_sent = self.logits[node_indices(self.data, torch.from_numpy(self.indices_received))].detach().numpy()
            else:
                self.logits_received = deserialized_a["data"]
                serialized_b = pkl.dumps({"data": self.logits_sent, "comm": 1})
                serialized_b = struct.pack('>I', len(serialized_b)) + serialized_b
                self.bluetooth_server.send(serialized_b)

                self.logits[node_indices(self.data, torch.from_numpy(self.indices_sent))] = torch.from_numpy(self.logits_received).detach()
        
        def data_received_client(data):
            deserialized_a = pkl.loads(data)
            print("received data over bluetooth")
            
            command = deserialized_a["comm"]
            
            if command == 0:
                self.indices_received = deserialized_a["data"]
                
                self.logits_sent = self.logits[node_indices(self.data, torch.from_numpy(self.indices_received))].detach().numpy()
                serialized_b = pkl.dumps({"data": self.logits_sent, "comm": 1})
                serialized_b = struct.pack('>I', len(serialized_b)) + serialized_b
                self.bluetooth_client.send(serialized_b) 
            else:
                self.logits_received = deserialized_a["data"]
                self.logits[node_indices(self.data, torch.from_numpy(self.indices_sent))] = torch.from_numpy(self.logits_received).detach()    
            
        if model == "mlp":
            self.model = MLP(input_size=self.data["x"].size(1),
                            hidden_size=256,
                            output_size=7,
                            device=device,
                            dropout=0.0,
                            num_hidden=2)  
            
        elif model == "mlppool_1":
            self.data["x"] = torch.load("logits/mlp_logits.pt").detach()
            self.logits = self.data["x"]
            first_hop_neighborhood = torch.logical_xor(self.data["nodes"], self.data["first_hop_nodes"])
            first_hop_nei_indices = (first_hop_neighborhood == True).nonzero(as_tuple=True)[0]
            self.indices_sent = self.data["index_orig"][first_hop_nei_indices].cpu().numpy()
            
            if is_bluetooth_server:
                self.bluetooth_server = CustomBluetoothServer(
                    data_received_server,
                    encoding=None,
                    auto_start=False,
                    when_client_connects= lambda: print("client connected"),
                    when_client_disconnects= lambda: print("client disconnected")
                )
                
                self.bluetooth_server.start()
                
            else:
                self.bluetooth_client = CustomBluetoothClient(bluetooth_server_name, data_received_client, encoding=None)
                serialized_a = pkl.dumps({"data": self.indices_sent, "comm": 0})  
                serialized_a = struct.pack('>I', len(serialized_a)) + serialized_a              
                self.bluetooth_client.send(serialized_a)                                
                            
            self.data["x"] = self.logits    
            self.model = MLPPool(input_size=self.data["x"].size(1),
                            hidden_size=256,
                            output_size=7,
                            device=device,
                            dropout=0.0,
                            num_hidden=2) 
            
        elif model == "mlppool_2":
            self.data["x"] = torch.load("logits/mlppool_logits.pt").detach()
            self.logits = self.data["x"]
            first_hop_neighborhood = torch.logical_xor(self.data["nodes"], self.data["first_hop_nodes"])
            first_hop_nei_indices = (first_hop_neighborhood == True).nonzero(as_tuple=True)[0]
            self.indices_sent = self.data["index_orig"][first_hop_nei_indices].cpu().numpy()
            
            if is_bluetooth_server:
                self.bluetooth_server = CustomBluetoothServer(
                    data_received_server,
                    encoding=None,
                    auto_start=False,
                    when_client_connects= lambda: print("client connected"),
                    when_client_disconnects= lambda: print("client disconnected")
                )
                
                self.bluetooth_server.start()
                
            else:
                self.bluetooth_client = CustomBluetoothClient(bluetooth_server_name, data_received_client, encoding=None)
                serialized_a = pkl.dumps({"data": self.indices_sent, "comm": 0})  
                serialized_a = struct.pack('>I', len(serialized_a)) + serialized_a              
                self.bluetooth_client.send(serialized_a)                                
                                        
            self.data["x"] = self.logits 
            self.model = MLPPool(input_size=self.data["x"].size(1),
                                 hidden_size=256,
                                 output_size=7,
                                 device=device,
                                 dropout=0.0,
                                 num_hidden=2)
            
        self.optimizer = get_optimizer(self.model,"sgd",learning_rate=0.05)
    
    def get_dataloaders():
        pass 
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.data, self.device, self.optimizer, epochs=1)
        self.round += 1
        if self.round == self.total_rounds:
            final_logits = self.model(self.data["x"], self.data["adj"])
            torch.save(final_logits, f"logits/{self.model.model_name}_logits.pt")
        return self.get_parameters(config={}), self.data["train_mask"].sum().item(), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = test(self.model, self.data, self.device)
        return loss, self.data["val_mask"].sum().item(), {"accuracy": acc}
        
if __name__ == "__main__":
    total_time_start = time.time()
    tracemalloc.start()
    data_dir = "data"
    dataset = "cora"
    
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="mlp"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="[::]:8080"
    )
    parser.add_argument(
        "--cid",
        type=int,
        default=0,
        metavar="N"
    )
    parser.add_argument(
        "--nb_clients",
        type=int,
        default=2
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )
    parser.add_argument(
        '--bluetooth_server', 
        action='store_true', 
        default=False
    )
    parser.add_argument(
        '--bluetooth_server_name',
        type=str,
        default="ubuntupi2-desktop"
    )
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=400
    )
    
    args = parser.parse_args()
    data = load_data(data_dir, dataset, args.cid, args.nb_clients)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = GNNClient(model= args.model, cid=args.cid, data=data, total_rounds=args.num_rounds, epochs=args.epochs, device=device, is_bluetooth_server=args.bluetooth_server, bluetooth_server_name=args.bluetooth_server_name)
    
    time.sleep(120)
    
    fl_training_start = time.time()  
    
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
    
    fl_training_end = time.time()
    fl_training_time = fl_training_end - fl_training_start
    print(f"Training time was {fl_training_time} seconds")
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Peak memory usage was {peak / 10**6} MB")
    
    total_time = time.time() - total_time_start
    print(f"Total time of execution was {total_time}")
    
    tracemalloc.stop()
