import argparse
import json
import os
import time
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import Transformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config/config.json')
    parser.add_argument("--vocab_pkl", type=str, default="data/preprocessed/dictionary.pkl")
    parser.add_argument("--test_path", type=str, default="data/preprocessed/test_")
    parser.add_argument("--output_dir", type=str, default="saved_model")
    parser.add_argument("--device", type=str, default='gpu')
    args = parser.parse_args()

    device = torch.device('cuda') if args.device is 'gpu' else 'cpu'
    print("device: ", device)

    with open(args.config_file) as f:
        config = json.load(f)


    batch_size = config['train']['batch_size']
    step_batch = config['train']['step_batch']
    max_epoch = config['train']['max_epoch']
    eval_interval = config['train']['eval_interval']
    warmup = config['train']['warmup']
    beta1 = config['train']['beta1']
    beta2 = config['train']['beta2']
    smoothing = config['train']['smoothing']
    
    n_layers = config['model']['n_layers']
    max_len = config['model']['n_position']
    d_model = config['model']['d_model']
    d_ff = config['model']['d_ff']
    n_heads = config['model']['n_head']
    dropout_p = config['model']['dropout_p']
    d_k = d_model // n_heads
    

    with open(args.vocab_pkl, 'rb') as fr:
        word_to_id, id_to_word = pickle.load(fr)
    vocab_size = len(word_to_id)
    print("vocab length: ", vocab_size)

    pad_idx = word_to_id['<pad>']
    print("padding index: ", pad_idx)

    with open(args.test_path + "src.pkl", 'rb') as fr:
        src, _ = pickle.load(fr)
    with open(args.test_path + "trg.pkl", 'rb') as fr:
        _, trg_out, _ = pickle.load(fr)

    test = TensorDataset(src, trg_out)
    test_loader = DataLoader(test, batch_size=100, shuffle=False)
    model = Transformer(vocab_size, d_model, n_heads, d_k, d_ff, n_layers, dropout_p, max_len).to(device)

    ckpt = torch.load('saved_model/batch_30/model_18.ckpt')
    model.load_state_dict(ckpt['model_state_dict'])
    


    predictions = []
 
    for src, trg_out in tqdm(test_loader):
        with torch.no_grad():
            src_len = torch.sum(src != 0, dim=-1)
            trg_in = torch.zeros_like(trg_out)
            trg_in[:, 0] = torch.tensor([1]).to(torch.int64)  # <bos> : 1

            src = src.to(device)
            trg_in = trg_in.to(device)
            trg_out = trg_out.to(device)
            
            batch = src.size(0)
            result = torch.empty(batch, max_len)

            for i in range(max_len):
                out = model.forward([src, src_len], trg_in)
                pred = torch.max(out, dim=-1)[1]
                if i != max_len-1:
                    trg_in[:, i+1] = pred[:, i]
                result[:, i] = pred[:, i].to('cpu')

            for batch in range(result.size(0)):
                prediction = []
                temp = ''
                for idx in result[batch]:
                    word = id_to_word[int(idx)]
                    if '@@' in word:
                        temp = word[:-2]
                        continue
                    if temp:
                        if word != word.lower():
                            word = temp + ' ' + word
                        else:
                            word = temp + word
                        temp = ''
                    if word == '<eos>':
                        break
                    prediction.append(word)
                sentence = ' '.join(prediction)
                predictions.append(sentence + '\n')

                
                       
    print('Prediction' + '-'*60)
    print(predictions[0])
    print(predictions[100])
    print(predictions[1000])

    print(f"Length of predicted sentences : {len(predictions)}")       

    with open('saved_model/predcitions.txt', 'w', encoding='utf8') as fw:
        fw.writelines(predictions)