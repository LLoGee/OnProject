import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pickle
import pandas as pd

get_class = lambda attr, name: getattr(__import__("{}.{}".format(attr, name), fromlist=["dummy"]), name)


def initialize(config, device):
    parameters = {}
    
    # reader = get_class("utils", config.get("data", "reader_name"))(config)
    formatter = get_class("utils", config.get("data", "formatter_name"))(config)
    batch_size = config.getint("train", "batch_size")
    shuffle = config.getboolean("train", "shuffle")

    collate_fn_decr = lambda mode: (lambda data, mode=mode: formatter.process(data, mode))
    
    # According to the formatter name, select an unused path and then read different data
    if config.get("data", "formatter_name") == 'BilstmFormatter':
        #The imported data at this time should be no crf
        with open('./data/Data/train_data.pkl', 'rb') as file:
            dataset_train = pickle.load(file)
        with open('./data/Data/valid_data.pkl', 'rb') as file:
            dataset_valid = pickle.load(file)
            
    if config.get("data", "formatter_name") == 'CrfFormatter':
        #The data imported at this time should have crf
        with open('./data/Data_Crf/train_data.pkl', 'rb') as file:
            dataset_train = pickle.load(file)
        with open('./data/Data_Crf/valid_data.pkl', 'rb') as file:
            dataset_valid = pickle.load(file)
            
    # weight for dataset
    if config.get("data", "weight") == 'True':
        sample_weights = []
        class_counts = pd.read_csv('./data/Data/train_data_label_distribution.csv')['count'].tolist()
        class_counts = torch.tensor(class_counts, dtype=torch.float64)
        class_weights = [1.0 / (count**0.1) for count in class_counts]
        if config.get('model', 'crf_option') == 'None':
            sample_weights = [class_weights[data['labels']] for data in dataset_train]
        else:
            for data in dataset_train:
                labels = data['labels']
                flags = data['flags']
                results = [labels[i] for i in range(len(labels)) if flags[i] == 1]
                weight = sum([class_weights[i] for i in results])
                sample_weights.append(weight)
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        parameters["dataset_train"] = DataLoader(dataset=dataset_train, batch_size=batch_size, collate_fn=collate_fn_decr("train"), sampler = sampler)
    else:
        parameters["dataset_train"] = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_decr("train"))
    parameters["dataset_valid"] = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_decr("valid"))

    parameters["model"] = get_class("model", config.get("model", "model_name"))(config)
    parameters["model"] = parameters["model"].to(device)
    
    parameters["optimizer"] = get_optim(parameters["model"], config)

    return parameters


def get_optim(model, config):
    hyper_params = {key: value for key, value in config["optimizer"].items() if key != "optimizer_name"}
    optimizer_name = config.get("optimizer", "optimizer_name")
    optimizer = getattr(optim, optimizer_name)
    command = "optim(params, {})".format(", ".join(["{}={}".format(key, value) for key, value in hyper_params.items()]))
    return eval(command, {"optim": optimizer, "params": model.parameters()})