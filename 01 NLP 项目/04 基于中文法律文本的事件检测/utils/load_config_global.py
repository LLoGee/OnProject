import configparser
from utils.global_variables import Global
import pickle
import numpy as np
import torch

def load_config_global(model_config):
  config = configparser.ConfigParser()
  config.read('./config/'+ model_config)
  config_runtime_dict ={}
  config.add_section("runtime")
  # config.get("model", "crf_option"), return str --> 'Crf' or 'None'
  if config.get("model", "crf_option") == 'None':
    with open('./data/Data/config_runtime_dict_crf_0.pkl', 'rb') as file:
      config_runtime_dict = pickle.load(file)
  else:
    with open('./data/Data_Crf/config_runtime_dict_crf_1.pkl', 'rb') as file:
      config_runtime_dict = pickle.load(file)
  for key, value in config_runtime_dict.items():
    config.set("runtime", key, value)

  # load global_variables
  if config.get("model", "crf_option") == 'None':
    with open('./data/Data/Global_dict_crf_0.pkl', 'rb') as file:
      Global_dict = pickle.load(file)
  else:
    with open('./data/Data_Crf/Global_dict_crf_1.pkl', 'rb') as file:
      Global_dict = pickle.load(file)
      Global.type2id = Global_dict["type2id"]

  if Global.word2vec_mat is None:
      Global.word2vec_mat = np.load('./data/word2vec.npy')
  Global.word2id = Global_dict["word2id"]
  Global.id2word = Global_dict["id2word"]
  Global.label2id = Global_dict["label2id"]
  Global.id2label = Global_dict["id2label"]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  Global.device = device
  print("Device:", device)
  
  return config, device