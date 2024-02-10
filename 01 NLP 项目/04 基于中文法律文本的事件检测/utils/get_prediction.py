import jieba
from utils.global_variables import Global
import numpy as np
import torch

def get_prediction(text, config, model, device):
    words = list(jieba.cut(text, cut_all=False))
    tokens = [Global.word2id[x] if x in Global.word2id else Global.word2id[x] ["<UNK>"] for x in words]
    labels = [0] * len(tokens)
    length = len(tokens)
    sequence_length = config.getint("runtime", "sequence_length")
    pad_tokens = np.array(tokens  + [Global.word2id["<PAD>"]] * (sequence_length - length)).reshape(1, -1)
    pad_labels = np.array(labels + [config.getint("data", "pad_label_id")] * (sequence_length - length)).reshape(1, -1)
    flags = [[1] * length + [0] * (sequence_length - length)]
    masks = np.array([1] * length + [0] * (sequence_length - length)).reshape(1, -1)
    lengths = np.array([length])
    
    tlt = lambda t: torch.LongTensor(t)
    tokens, labels, masks, lengths = tlt(pad_tokens), tlt(pad_labels), tlt(masks), tlt(lengths)
    data = {"tokens":tokens, "labels":labels, "lengths":lengths, "flags":flags, 'masks':masks}
    
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    
    results = model(data=data, mode='test', crf_mode="test")
    prediction = results["prediction"]
    
    return words, prediction[0]