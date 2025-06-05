import os, json, math, torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from os.path import join as path
from configuration import Configuration
from federated_clients import set_parameters

cluster_A = [
    "client-1",
    "client-5",
    "client-7",
    "client-8",
    "client-9",
    "client-12"]

class MetricsTracker():
    def __init__(self, distribution_case, aggregation_algo):
        self.distribution_case = distribution_case
        self.aggregation_algo = aggregation_algo
        self.c = Configuration()
        self.metrics = {
            "Loss": {  "global": [],"client-1": [], "client-2": [], "client-3": [], "client-4": [], "client-5": [], "client-6": [], "client-7": [], "client-8": [], "client-9": [], "client-10": [], "client-11": [], "client-12": [], "client-13": [], "client-14": []},
            "Accuracy": {  "global": [], "client-1": [], "client-2": [], "client-3": [], "client-4": [], "client-5": [], "client-6": [], "client-7": [], "client-8": [], "client-9": [], "client-10": [], "client-11": [], "client-12": [], "client-13": [], "client-14": []},
            "F1-Score": {  "global": [], "client-1": [], "client-2": [], "client-3": [], "client-4": [], "client-5": [], "client-6": [], "client-7": [], "client-8": [], "client-9": [], "client-10": [], "client-11": [], "client-12": [], "client-13": [], "client-14": []},
            "Recall": {  "global": [], "client-1": [], "client-2": [], "client-3": [], "client-4": [], "client-5": [], "client-6": [], "client-7": [], "client-8": [], "client-9": [], "client-10": [], "client-11": [], "client-12": [], "client-13": [], "client-14": []},
            "Precision": { "global": [], "client-1": [], "client-2": [], "client-3": [], "client-4": [], "client-5": [], "client-6": [], "client-7": [], "client-8": [], "client-9": [], "client-10": [], "client-11": [], "client-12": [], "client-13": [], "client-14": []}
        }
        self.cl = {
            "Loss": 0, "Accuracy": 0, "F1-Score": 0, "Recall": 0, "Precision": 0
        }
        self.best_val_loss_epoch_cl = 0
        self.load_cl()
        
    def update_client(self, client_id, metric_loss, metric_acc, metric_f1, metric_recall, metric_precision, batch_cnt):
        client_id = f"client-{client_id}"
        
        #self.metrics["batch_cnt"][client_id].append(batch_cnt)
        self.metrics["Loss"][client_id].append(metric_loss)
        self.metrics["Accuracy"][client_id].append(metric_acc)
        self.metrics["F1-Score"][client_id].append(metric_f1)
        self.metrics["Recall"][client_id].append(metric_recall)
        self.metrics["Precision"][client_id].append(metric_precision)
            
    def update_global(self, metric_loss, metric_acc, metric_f1, metric_recall, metric_precision):
        client_id = "global"
        
        self.metrics["Loss"][client_id].append(metric_loss)
        self.metrics["Accuracy"][client_id].append(metric_acc)
        self.metrics["F1-Score"][client_id].append(metric_f1)
        self.metrics["Recall"][client_id].append(metric_recall)
        self.metrics["Precision"][client_id].append(metric_precision)
        
        self.save()
        self.figs()
    
    def save_model_global_real(self, model, round):
        path_model = path(self.c.path_results, "global_model", self.distribution_case, self.aggregation_algo)
        os.makedirs(path_model, exist_ok=True)
        torch.save(model.state_dict(), path(path_model, f"model-{round}.torch"))
        
    def load_model_global_real(self, round):
        path_model = path(self.c.path_results, "global_real", self.distribution_case, self.aggregation_algo, f"model-{round}.torch")
        return torch.load(path_model)
    
    def save(self):
        with open(path(self.c.path_results, "scenarios", self.distribution_case, self.aggregation_algo, "metrics.json"), 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
    def load(self):
        with open(path(self.c.path_results, "scenarios", self.distribution_case, self.aggregation_algo, "metrics.json"), 'r') as f:
            self.metrics = json.load(f)
    
    def load_cl(self):
        with open(path(self.c.path_results, "scenarios", "CL", "FC-All", "metrics.json"), 'r') as f:
            json_data = json.load(f)
            self.cl["Loss"] = json_data["best_val_loss"]
            self.cl["Accuracy"] = json_data["test_acc"]
            self.cl["F1-Score"] = json_data["test_f1"]
            self.cl["Recall"] = json_data["test_recall"]
            self.cl["Precision"] = json_data["test_precision"]
    
    def figs_client_group(self, dataset):
        plt.figure(figsize=(14, 8))
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'CMU Serif'
        plt.rcParams['font.size'] = '12'
        
        metric = "F1-Score"
        plt.axhline(y=self.cl[metric], color='r', linestyle='--', label='CL')
        
        weights_a = [0] * 112
        weights_b = [0] * 112
        cluster_a_scores = [0] * 112
        cluster_b_scores = [0] * 112
        for client, scores in self.metrics[metric].items():
            if client == "global" or client == "global-real" or client == "CL":
                continue
                
            if client in cluster_A:
                for round in range(0,112,1):
                    df_test_len = len(pd.read_parquet(path(self.c.path_dataset, "5-federated", dataset, client, f"test-chunk-{round+1}.parquet"), columns=self.c.appl))
                    try:
                        cluster_a_scores[round] += scores[round] * df_test_len
                        weights_a[round] += df_test_len
                    except:
                        pass
                    
            else:
                for round in range(0,112,1):
                    df_test_len = len(pd.read_parquet(path(self.c.path_dataset, "5-federated", dataset, client, f"test-chunk-{round+1}.parquet"), columns=self.c.appl))
                    try:
                        cluster_b_scores[round] += scores[round] * df_test_len
                        weights_b[round] += df_test_len
                    except:
                        pass
                    
        for round in range(0,112,1):
            cluster_a_scores[round] /= weights_a[round]
            cluster_b_scores[round] /= weights_b[round]
            
        x_values = np.arange(1, len(cluster_a_scores)+1)
        
        plt.plot(x_values, cluster_a_scores, marker='.', linestyle='-', color="green", alpha=0.7, label="A csoport")
        plt.plot(x_values, cluster_b_scores, marker='.', linestyle='-', color="red", alpha=0.7, label="B csoport")
        
        x_values = np.arange(1, len(self.metrics[metric]["client-10"])+1)  # Create x-axis values
        plt.plot(x_values, self.metrics[metric]["client-10"], marker='.', linestyle='--', label="Kliens 10")
        
        x_values = np.arange(1, len(self.metrics[metric]["client-14"])+1)  # Create x-axis values
        plt.plot(x_values, self.metrics[metric]["client-14"], marker='.', linestyle='--', label="Kliens 14")
        
        x_values = np.arange(1, len(self.metrics[metric]["global"])+1)  # Create x-axis values
        plt.plot(x_values, self.metrics[metric]["global"], color="blue", marker='.', linestyle='--', label=self.aggregation_algo)

            
        plt.ylim(0, 1)
        plt.xticks(np.arange(4, self.c.rounds+1, 4), rotation=0)
        plt.yticks(np.arange(0.02, 1.0, 0.05))
        
        _add_night_rectangles()
        
        # plt.title(f"{self.distribution_case} - {self.aggregation_algo} - {metric} progression over time.")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.margins(0)
        plt.savefig(path(self.c.path_results, "scenarios", self.distribution_case, self.aggregation_algo, f"thesis-F1.png"), dpi=600)
        plt.clf()
        plt.close()
    
    def figs(self):
        plt.figure(figsize=(14, 8))
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'CMU Serif'
        plt.rcParams['font.size'] = '12'

        
        
        for metric in self.metrics:
            y_min = 1
            plt.axhline(y=self.cl[metric], color='r', linestyle='--', label='CL')
            
            colors = ["red", "blue", "green", "orange", "purple", "pink", "brown", "gray"]
            ci_a, ci_b = 0, 0 # color index
            for client, scores in self.metrics[metric].items():
                x_values = np.arange(1, len(scores)+1)  # Create x-axis values
                if client == "global" or client == "global-real" or client == "CL":
                    pass
                else:
                    if client in cluster_A:
                        plt.plot(x_values, scores, marker='.', linestyle='-', label=client)
                        ci_a += 1
                    else:
                        plt.plot(x_values, scores, marker='.', linestyle='-.', label=client)
                        ci_b += 1
                        
                for score in scores:
                    if score < y_min:
                        y_min = score
            
            x_values = np.arange(1, len(self.metrics[metric]["global"])+1)  # Create x-axis values
            plt.plot(x_values, self.metrics[metric]["global"], color="blue", marker='.', linestyle='--', label=self.aggregation_algo)

            if metric == "Loss":
                plt.ylim(0, 5)
            else:
                y_min = math.floor(y_min*100)/100
                plt.ylim(y_min, 1)
            plt.xticks(np.arange(4, self.c.rounds+1, 4), rotation=0)
            plt.yticks(np.arange(0.02, 1.0, 0.05))
            #plt.xlabel("Run")
            #plt.ylabel(f"{metric}")
            
            _add_night_rectangles()
            
            # plt.title(f"{self.distribution_case} - {self.aggregation_algo} - {metric} progression over time.")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.margins(0)
            plt.savefig(path(self.c.path_results, "scenarios", self.distribution_case, self.aggregation_algo, f"{metric}.png"), dpi=600)
            plt.clf()
        plt.close()
            
def figs_overall_scenario(scenario):
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'CMU Serif'
    plt.rcParams['font.size'] = '14'
    c = Configuration()
    
    # Load Central model metrics
    metrics_cl = {}
    with open(path(c.path_results, "scenarios", "CL", "FC-All", "metrics.json"), 'r') as f:
        json_data = json.load(f)
        metrics_cl["Loss"] = json_data["best_val_loss"]
        metrics_cl["Accuracy"] = json_data["test_acc"]
        metrics_cl["F1-Score"] = json_data["test_f1"]
        metrics_cl["Recall"] = json_data["test_recall"]
        metrics_cl["Precision"] = json_data["test_precision"]
    
    
    
    path_case = path(c.path_results, "scenarios", scenario)
    os.makedirs(path(path_case, "Overall"), exist_ok=True)
    dirs = os.listdir(path_case)
    
    for metric in ["Accuracy", "F1-Score", "Recall", "Precision"]:
        y_min = 1
        plt.axhline(y=metrics_cl[metric], color='r', linestyle='--', label='CL')
        
        
        colors = ["red", "blue", "green", "orange", "purple"]
        ci = 0 # color index
        for aggregation_algo in dirs:
            if aggregation_algo != "Overall":
                metrics_tracker = MetricsTracker(scenario, aggregation_algo)
                metrics_tracker.load()
                scores = metrics_tracker.metrics[metric]["global"]
                x_values = np.arange(1, len(scores)+1)
                plt.plot(x_values, scores, marker='.', linestyle='-', color=colors[ci], label=aggregation_algo)
                for score in scores:
                    if score < y_min:
                        y_min = score
                
                ci+=1
        
        y_min = math.floor(y_min*100)/100
        _add_night_rectangles()
        plt.ylim(y_min, 1)
        plt.xticks(np.arange(4, c.rounds+1, 4), rotation=0)
        plt.yticks(np.arange(y_min, 1.0, 0.05))
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.margins(0)
        plt.savefig(path(c.path_results, "scenarios", scenario, "Overall", f"{metric}.png"), dpi=300)
        plt.clf()
    plt.close()
    
def _add_night_rectangles():
    c = Configuration()
    # Highlight night time 21:00-6:00
    # 00:00 - 3:00  : 1 x  | mod 8 = 1
    # 3:00  - 6:00  : 2 x  | mod 8 = 2
    # 6:00  - 9:00  : 3 x  | mod 8 = 3
    # 9:00  - 12:00 : 4
    # 12:00 - 15:00 : 5
    # 15:00 - 18:00 : 6
    # 18:00 - 21:00 : 7
    # 21:00 - 00:00 : 8 x | mod 8 = 0
    for start, end in [(1,3), (c.rounds, c.rounds)]+[(i, i+3) for i in range(8, c.rounds, 8)]:
        plt.axvspan(start, end, color='darkblue', alpha=0.3)

def on_fit_config_fn(server_round):
    return {"round": server_round}

def on_evaluate_config_fn(server_round):
    return {"round": server_round}

# Runs after every evaluation round (Global model tested on all clients)
def evaluate_fn(model, metric_tracker: MetricsTracker):
    def fn(server_round, parameters, config=None):
        if server_round == 0:
            return None
        
        set_parameters(model, parameters)
        metric_tracker.save_model_global_real(model, server_round)
        
    return fn

# Runs after every fit round (Local models)
def fit_metrics_aggregation_fn(metrics_tracker: MetricsTracker):
    def fn(fit_results):
        cnt = 0
        weighted_average_acc = 0
        weighted_average_loss = 0
        weighted_average_f1 = 0
        weighted_average_recall = 0
        weighted_average_precision = 0
        
        for batch_cnt, metrics in fit_results:
            cnt += batch_cnt
            client_id = metrics["client_id"]
            weighted_average_acc += metrics["val_metric_acc"] * batch_cnt
            weighted_average_loss += metrics["val_metric_loss"] * batch_cnt
            weighted_average_f1 += metrics["val_metric_f1"] * batch_cnt
            weighted_average_recall += metrics["val_metric_recall"] * batch_cnt
            weighted_average_precision += metrics["val_metric_precision"] * batch_cnt
        
        return {
            "weighted_average_acc": weighted_average_acc / cnt,
            "weighted_average_loss": weighted_average_loss / cnt,
            "weighted_average_f1": weighted_average_f1 / cnt,
            "weighted_average_recall": weighted_average_recall / cnt,
            "weighted_average_precision": weighted_average_precision / cnt
        }
    return fn

# Runs after every evaluate round (Global model)
def evaluate_metrics_aggregation_fn(metrics_tracker: MetricsTracker):
    def fn(evaluation_results):
        cnt = 0
        weighted_average_acc = 0
        weighted_average_loss = 0
        weighted_average_f1 = 0
        weighted_average_recall = 0
        weighted_average_precision = 0
        
        for batch_cnt, metrics in evaluation_results:
            cnt += batch_cnt
            client_id = metrics["client_id"]
            weighted_average_acc += metrics["test_metric_acc"] * batch_cnt
            weighted_average_loss += metrics["test_metric_loss"] * batch_cnt
            weighted_average_f1 += metrics["test_metric_f1"] * batch_cnt
            weighted_average_recall += metrics["test_metric_recall"] * batch_cnt
            weighted_average_precision += metrics["test_metric_precision"] * batch_cnt
            
            metrics_tracker.update_client(
                client_id, 
                metrics["test_metric_loss"], 
                metrics["test_metric_acc"], 
                metrics["test_metric_f1"], 
                metrics["test_metric_recall"], 
                metrics["test_metric_precision"], 
                batch_cnt
            )
        
        metrics_tracker.update_global(
            weighted_average_loss / cnt, 
            weighted_average_acc / cnt, 
            weighted_average_f1 / cnt, 
            weighted_average_recall / cnt, 
            weighted_average_precision / cnt
        )
        
        return {
            "weighted_average_acc": weighted_average_acc / cnt,
            "weighted_average_loss": weighted_average_loss / cnt,
            "weighted_average_f1": weighted_average_f1 / cnt,
            "weighted_average_recall": weighted_average_recall / cnt,
            "weighted_average_precision": weighted_average_precision / cnt
        }
    return fn