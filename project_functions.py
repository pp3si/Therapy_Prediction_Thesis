"""
This file is for storing functionality that we'll want to use in multiple different files in the project,
to avoid code duplication
"""

import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math, copy, time
import project_functions
import os
import gc
import tracemalloc
from time import perf_counter
import argparse
from sklearn.model_selection import KFold
import random
from scipy import stats
    
def drop_stubs(OQ_list, min_size=2):
    """
    given one of the lists in OQ_lists.pkl, drop all time series strictly less than min_size
    """
    new_list = []
    for series in OQ_list:
        if len(series) >= min_size:
            new_list.append(series)
    return new_list

def decompose(series):
    """
    Take a series and, at each timestep, split it into all the preceding values and the next one
    """
    truncations = [series[:i] for i in range(1,len(series))]
    nexts = series[1:]
    return truncations, nexts

def make_data_rectangular(OQ_trajectories, df_width=None, impute_val=None,
                          from_back=True, exact_width=False, add_const=False):
    """
        Random forests expect all inputs to have the same number of features, so we transform the given list of timeseries to be so.
        This function also makes a separate training point for each step in each timeseries ( like decompose() ).

        We truncate the long series and fill in the short ones

        If exact_width, we only consider timeseries of length at least df_width+1 (which each have df_width or more training observations).
            So, that variable is a little poorly named, I guess.
        
    """

    #If no impute val provided, use the mean of the train data
    if impute_val is None:
        with open('OQ_lists.pkl', 'rb') as file:
            OQ_lists = pickle.Unpickler(file).load()
        train_OQ_list, _ = OQ_lists
        train_OQ_list = drop_stubs(train_OQ_list)
        overall_mean_OQ_train = np.mean(np.concatenate(train_OQ_list))
        impute_val = overall_mean_OQ_train


    #Make lists to store the training observations and correct next value
    OQ_no_labels = []
    OQ_labels = []
    running_means = []

    min_width = df_width if exact_width else 1

    longest_trajectory = 0
    for trajectory in OQ_trajectories:
        longest_trajectory = max(longest_trajectory, len(trajectory))

        #Add what is known at each timestep, and the appropriate next observation to predict, to the lists
        for i in range(min_width, len(trajectory)):
            OQ_no_labels.append(trajectory[:i])
            OQ_labels.append(trajectory[i])
            running_means.append(np.mean(trajectory[:i]))
    
    if not df_width: df_width = longest_trajectory # Default is full-size, pretty unnecessarily big

    if type(impute_val)==str and impute_val=="personal_mean":
        data = np.vstack([np.ones(df_width)*mean for mean in running_means])
        assert len(data) == len(OQ_no_labels)
    elif type(impute_val)==str:
        raise ValueError("Unknown imputation type given")
    else:
        # makes training data into array; sequences longer than df_width are truncated
        data = np.full((len(OQ_no_labels), df_width), impute_val)

    """If from_back is true (default) then for too-short series,
        fill in the first values with the imputation and make the last ones informative
        Otherwise, make the real values the first ones and postpend the imputation value"""
    if from_back:
        for i, traj in enumerate(OQ_no_labels):
            data[i,-min(len(traj), df_width):] = traj[-df_width:]
    else:
        for i, traj in enumerate(OQ_no_labels):
            data[i, :min(len(traj), df_width)] = traj[-df_width:]
    if add_const:
        data = np.hstack((np.ones(len(data)).reshape((-1,1)), data))
    return data, OQ_labels

class Simple_LSTM(nn.Module):
        def __init__(self, hidden_size, dropout, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden_size=hidden_size, batch_first=True, dropout=dropout, num_layers=num_layers)
            self.linear = nn.Linear(hidden_size,1)
        def forward(self,x):
            x, _ = self.lstm(x)
            x = self.linear(x)
            return x

def run_LSTM(widths, out_folder, hidden_size, num_layers, dropout, batch_size, lr, cv, num_epochs,
             datafile = 'OQ_lists_new.pkl'):
    
    torch.manual_seed(2024)
    random.seed(64)
    np.random.seed(42)

    print("Entering run_LSTM")
    tracemalloc.start()
    #Load data
    print("Loading data")

    '''
    If cross-validating, we won't use the validation set at all, but will instead
    run the model-training script cv times using cv-fold cross-validation. Best models
    will be saved according to their score on their local cross-validation val set,
    and we'll report the final average cv-based MSE at the end. Once we have a winner,
    we can run a training on just that set and use the actual mid-training model with
    lowest val-set MSE (using the actual validation set.)
    '''
    with open(datafile, 'rb') as file:
        OQ_lists = pickle.Unpickler(file).load()
    train_OQ_list, val_OQ_list = OQ_lists

    train_OQ_list = drop_stubs(train_OQ_list)
    val_OQ_list = drop_stubs(val_OQ_list)

    train_sets = []
    val_sets = []
    if cv == 1:
        #The parallel lists of local training and validation sets are just... the training and validation sets
        train_sets.append(train_OQ_list)
        val_sets.append(val_OQ_list)
    else:
        #In this case, we draw everything from the training set
        kfold = KFold(n_splits=cv, shuffle=True, random_state=2024)
        for (train_index, val_index) in kfold.split(train_OQ_list):
            train_sets.append([train_OQ_list[i] for i in range(len(train_OQ_list)) if i in train_index])
            val_sets.append([train_OQ_list[i] for i in range(len(train_OQ_list)) if i in val_index])


    print("Loaded data")


    #Make LSTM
    #A decent amount of syntax help from https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/

    assert torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #Define the main training loop before running

    val_step = 100
    def scope():
        times = []

        #train the model
        for epoch in range(num_epochs):

            #Memory profiling
            times.append(perf_counter())
            gc.collect()
            if epoch==0:
                snap1 = tracemalloc.take_snapshot()
            if (epoch > 0 and (not epoch%5)) or (len(times) > 2 and times[-1]-times[-2] > times[-2]-times[-3]+10):
                snap2 = tracemalloc.take_snapshot()
                s1v2 = snap2.compare_to(snap1, 'lineno')
                with open('./'+local_out_folder+'/memory_leak_analysis.txt', 'w') as f:
                    f.write(f"[ Memory usage increase from snapshot 1 to snapshot 2 at epoch {epoch} ]\n")
                    for stat in s1v2[:15]:
                        f.write(f"{stat}\n")
                with open('./'+local_out_folder+'/memory_snapshot2.txt', 'w') as f:
                    f.write(f"[ Memory usage in snapshot 2 at epoch {epoch} ]\n")
                    top_stats = snap2.statistics("lineno")
                    for stat in top_stats[:15]:
                        f.write(f"{stat}\n")

            
            model.train()
            time_str = '' if not epoch else f", previous epoch took {(times[-1]-times[-2]):.3f}s"
            print(f"Entering training step in epoch {epoch}"+time_str)
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)[:,-1,:].unsqueeze(1)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            print("Finished training all batches")
            
            #Validate
            if (not epoch % val_step) or epoch==(num_epochs-1):
                print(f"Validating in epoch {epoch}")
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_train)[:,-1,:].unsqueeze(1)
                    train_mse = criterion(y_pred, y_train)
                    y_pred = model(X_val)[:,-1,:].unsqueeze(1)
                    val_mse = criterion(y_pred, y_val)

                    train_mse_cpu = train_mse.detach().cpu().item()
                    val_mse_cpu = val_mse.detach().cpu().item()

                    if len(list(val_mse_list.values())) == 0 or val_mse_cpu < np.min(list(val_mse_list.values())):
                        print(f"Found new min of {val_mse_cpu}")
                        filename = local_out_folder+f"/window_{window_size}_epoch_{epoch}_mse_{val_mse_cpu:.3f}"
                        with open(filename, 'wb') as file:
                            torch.save(model, file)
                            print("Dumped model successfully")

                    train_mse_list[epoch] = train_mse_cpu
                    val_mse_list[epoch] = val_mse_cpu

                    if epoch == num_epochs-1:
                        final_val_mses.append(val_mse_cpu)

                    print(f"Epoch {epoch}, train mse: {train_mse_cpu}, val mse: {val_mse_cpu}", flush=True)
        
        print(f"Adding train_mse_dict[{window_size}]")
        train_mse_dict[window_size] = train_mse_list
        val_mse_dict[window_size] = val_mse_list
        print(f"Added train_mse_dict[{window_size}]")

    final_val_mses = []

    #Actually run the training loop, with cross-validation
    for split in range(cv):
        print(f"Entering split {split}")
        local_train_OQ_list = train_sets[split]
        local_val_OQ_list = val_sets[split]
        overall_mean_OQ_train = np.mean(np.concatenate(local_train_OQ_list))
        train_mse_dict = {}
        val_mse_dict = {}
        local_out_folder = out_folder+f'/split_{split}'
        if not os.path.exists('./'+local_out_folder):
            os.makedirs('./'+local_out_folder)

        for window_size in widths:
            print(f"Using window size {window_size}", flush=True)
            train_mse_list = {}
            val_mse_list = {}
            print("Rectangularizing train data")
            train_data, train_labels = make_data_rectangular(local_train_OQ_list, df_width = window_size, impute_val=overall_mean_OQ_train)
            print("Rectangularized train data, rectangularizing val data")
            val_data, val_labels = make_data_rectangular(local_val_OQ_list, df_width = window_size, impute_val=overall_mean_OQ_train)
            print("Rectangularized val data, loading train data into tensors")
            X_train, y_train = torch.tensor(train_data).float().unsqueeze(2).to(device), torch.tensor(train_labels).float().unsqueeze(1).unsqueeze(2).to(device)
            print("Tensor-ed train data, loading val data into tensors")
            X_val, y_val = torch.tensor(val_data).float().unsqueeze(2).to(device), torch.tensor(val_labels).float().unsqueeze(1).unsqueeze(2).to(device)
            print("Tensor-ed val data")

            model = Simple_LSTM(hidden_size, dropout, num_layers)
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)

            print("Calling scope")
            scope()
            print("Finished scope call")

        print("Dumping mse_dicts")
        with open(local_out_folder+'/train_mse_dict.pkl', 'wb') as file:
            pickle.Pickler(file=file).dump(train_mse_dict)
        with open(local_out_folder+'/val_mse_dict.pkl', 'wb') as file:
            pickle.Pickler(file=file).dump(val_mse_dict)
        print("Dumped mse_dicts")

    print(f"Mean final OOS MSE: {np.mean(final_val_mses)}")

    tracemalloc.stop()