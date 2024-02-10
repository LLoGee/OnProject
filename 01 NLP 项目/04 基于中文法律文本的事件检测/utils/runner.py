import os
import sys
import json
import copy
import torch
import pandas as pd
from utils.global_variables import Global
from utils.evaluation import Evaluation


def run(parameters, config, device):
    trained_epoch = -1
    train_loss_lis = []
    valid_loss_lis = []
    # Obtain the maximum training epoch. The initial default epoch is 15
    # If there is demand for subsequent models, increase it
    max_epoch = config.getint("train", "epoch")
    valid_interval = config.getint("train", "valid_interval")
    saver = {}
    for epoch in range(trained_epoch + 1, max_epoch):
        train_loss, _ = run_one_epoch(parameters, config, device, epoch, "train")
        train_loss_lis.append(train_loss)
    
        if epoch % valid_interval == 0:
            with torch.no_grad():
                valid_loss, valid_metric = run_one_epoch(parameters, config, device, epoch, "valid")
                valid_loss_lis.append(valid_loss)
                print()
                if saver == {} or valid_metric["micro_f1"] > saver["valid"]["micro_f1"]:
                    saver["epoch"] = epoch
                    saver["valid"] = valid_metric
                    if not os.path.exists('./performance/{}'.format(config.get('model', 'model_name'))):
                        os.makedirs('./performance/{}'.format(config.get('model', 'model_name')))
                    with open('./performance/{}/valid-metric.txt'.format(config.get('model', 'model_name')), 'w', encoding='utf-8') as f:
                        json.dump(valid_metric, f, indent=4, ensure_ascii=False)
                        
    print("Best Epoch {}\nValid Metric: {}\n".format(saver["epoch"], saver["valid"]))
    
    # Save the loss as a csv file to facilitate subsequent visualization
    loss_df = pd.DataFrame({'train': train_loss_lis, 'valid': valid_loss_lis})
    # Set index=False to avoid saving index columns
    loss_df.to_csv('./performance/{}/loss.csv'.format(config.get('model', 'model_name')), index=False)  

def run_one_epoch(parameters, config, device, epoch, mode):
    model = parameters["model"]

    if mode == "train":
        model.train()
        optimizer = parameters["optimizer"]
    elif mode == "valid" or mode == "test":
        model.eval()
    else:
        raise NotImplementedError

    dataset = copy.deepcopy(parameters["dataset_{}".format(mode)])

    total_loss = 0
    evaluation = Evaluation(config)

    if mode == 'train':
        optimizer.zero_grad()
    grad_accumulation_steps = 2

    for step, data in enumerate(dataset):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

        # input and output
        if config.get("model", "crf_option") == "Crf":
            results = model(data=data, mode=mode, crf_mode="train")
            loss = results["loss"]
            total_loss += loss.item()
            results = model(data=data, mode=mode, crf_mode="test")
            evaluation.expand(results["prediction"], results["labels"])
                
        else:
            results = model(data=data, mode=mode)
            loss = results["loss"]
            total_loss += loss.item()
            evaluation.expand(results["prediction"], results["labels"])

        # print("\r{}: Epoch {} Step {:0>4d}/{} | Loss = {:.4f}".format(mode, epoch, step + 1, len(dataset), round(total_loss / (step + 1), 4)), end="")
        
        # backward
        if mode == "train":
            loss.backward()
            if (step+1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

    metric = evaluation.get_metric("all")
    del metric['report']
    sys.stdout.write("\r")
    print("\r{}: Epoch {} | Metric: {}".format(mode, epoch, metric))

    return total_loss, metric
