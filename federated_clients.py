import sys, os, time
import pandas as pd

from os.path import join as path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from configuration import Configuration
from flwr.client import NumPyClient
import torch
from torch import nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import copy

import matplotlib.pyplot as plt
import seaborn as sns

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

class FQDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class FederatedClient(NumPyClient):
    def __init__(self, model, client_id, distribution_case, features, fedprox_mu=None):
        self.client_id = client_id
        self.model = model
        self.c = Configuration()
        self.distribution_case = distribution_case
        
        self.device = "mps"
        self.random_state = self.c.random_state
        self.epochs = 10
        self.batch_size = 64
        self.lr = 0.001

        self.model.to(self.device)

        self.features = features
        self.columns = self.features + self.c.appl
        self.loss_fn = nn.CrossEntropyLoss()
        self.fedprox_mu = fedprox_mu
        
    # Returns current local model parameters
    def get_parameters(self, config):
        return get_parameters(self.model)

    # Fit local model (train, validate, test)
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        dl_train, dl_val = self._load_train_data(config["round"])
        
        ### VALIDATE GLOBAL MODEL ###
        
        best_val_loss_epoch = 0
        best_val_loss_model_parameters = get_parameters(self.model)
        best_val_loss, best_val_acc, best_val_precision, best_val_recall, best_val_f1 = self._validate(dl_val)
        global_params = copy.deepcopy(self.model).parameters()
        
        ### TRAIN ###
        train_loss = 0
        train_time = 0
        train_acc = 0
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epochs):
            
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            # Training loop
            t_start = time.time()
            for x, y in dl_train:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                
                # If FedProx, calculate and add the proximal term.
                if self.fedprox_mu is not None:
                    proximal_term = 0
                    for local_weights, global_weights in zip(self.model.parameters(), global_params):
                        proximal_term += (local_weights - global_weights).norm(2)
                        
                    loss += (self.fedprox_mu / 2) * proximal_term
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == y).sum().item()
                total_train += y.size(0)
                    
            train_time += time.time() - t_start
            train_loss += train_loss/len(dl_train)
            train_acc += correct_train/total_train
            
            val_metric_loss, val_metric_acc, val_metric_precision, val_metric_recall, val_metric_f1 = self._validate(dl_val)
            if val_metric_loss < best_val_loss:
                best_val_loss = val_metric_loss
                best_val_acc = val_metric_acc
                best_val_precision = val_metric_precision
                best_val_recall = val_metric_recall
                best_val_f1 = val_metric_f1
                best_val_loss_epoch = epoch+1
                best_val_loss_model_parameters = get_parameters(self.model)
        
        train_loss /= self.epochs
        train_time /= self.epochs
        train_acc /= self.epochs
        
        if best_val_loss_epoch == 0:
            raise Exception("No validation loss improvement in client {}".format(self.client_id))
        
        return best_val_loss_model_parameters, len(dl_train), {
            "client_id": self.client_id,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_time": train_time,
            "val_metric_loss": best_val_loss,
            "val_metric_acc": best_val_acc,
            "val_metric_recall": best_val_recall,
            "val_metric_precision": best_val_precision,
            "val_metric_f1": best_val_f1,
            "best_val_loss_epoch": best_val_loss_epoch
        }

    # Evaluate global model
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        dl_test = self._load_test_data(config["round"])
        
        self.model.eval()
        
        all_labels = []
        all_preds = []

        test_loss = 0.0
        with torch.no_grad():
            for x, y in dl_test:
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
            
            
        # Calculate metrics

        test_metric_loss = test_loss/len(dl_test)
        test_metric_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        test_metric_acc = accuracy_score(all_labels, all_preds)
        test_metric_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        test_metric_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return test_metric_loss, len(dl_test), {
            "client_id": self.client_id,
            "test_metric_loss": test_metric_loss,
            "test_metric_acc": test_metric_acc,
            "test_metric_recall": test_metric_recall,
            "test_metric_precision": test_metric_precision,
            "test_metric_f1": test_metric_f1
            }
    
    # Returns: loss, acc, precision, recall, f1
    def _validate(self, dl_val):
        self.model.eval()
        
        val_loss = 0.0
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        
        val_metric_loss = val_loss/len(dl_val)
        val_metric_acc = accuracy_score(all_labels, all_preds)
        val_metric_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_metric_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_metric_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return val_metric_loss, val_metric_acc, val_metric_precision, val_metric_recall, val_metric_f1
    
    def _load_data(self, dataset, server_run):
        df = pd.read_parquet(path(self.c.path_dataset, "5-federated", self.distribution_case, f"client-{self.client_id}", f"{dataset}-chunk-{server_run}.parquet"), columns=self.columns)
        X, y = df.drop(columns=self.c.appl), df[self.c.app]
        ds = FQDataset(X, y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle= True if dataset=="train" else False)
        
    
    def _load_train_data(self, server_run):
        
        return self._load_data("train", server_run), self._load_data("validation", server_run)
    
    def _load_test_data(self, server_run):

        return self._load_data("test", server_run)