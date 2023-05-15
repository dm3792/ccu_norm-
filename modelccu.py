import argparse
from tqdm import tqdm
from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import copy
import math
from loaders.ldc_data import load_ldc_data
from ldcNormsCombine import generate_input
import os
import sys
import shutil
import pandas as pd
# from scoring import score_cp
import csv 
from matplotlib import pyplot as plt
import random
from functools import partial
from multiprocessing import Pool
from scoring.average_precision import calculate_average_precision
import json
import numpy as np

# TODO:
# - support several different training modes:
# PRIORITY 1
# --- controllable downsampling (what ratio of positive to negative examples do we train on?  should not affect validation)
# LOWER PRIORITY
# --- the number of changepoints predicted in the document so far
# --- in addition to downsampling, support different weights for 0 vs 1 examples in the loss
#       https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
# - it should have early stopping guided by average precision on the validation set

# utterance1<NA1 utterance2<NV2 utterance3<CP,NV2 utterance4 utterance5

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

class ChangepointNormsDataset(Dataset):
    def __init__(self, split, utterances_before, utterances_after,confident_only):
        self.split = split
        self.utterances_before = utterances_before
        self.utterances_after = utterances_after

        self.examples = generate_input(self.split,self.utterances_before,self.utterances_after,confident_only)
        if(self.split=="INTERNAL_TRAIN"):
            print("hereeeeee")
            label = [item['label'] for item in self.examples]
            majority_class_indices = [i for i, label in enumerate(label) if label == 0]
            minority_class_indices = [i for i, label in enumerate(label) if label == 1]
            undersampled_majority_indices = resample(majority_class_indices, replace=False, n_samples=len(minority_class_indices), random_state=42)
            undersampled_indices = undersampled_majority_indices + minority_class_indices
            undersampled_X = [self.examples[i] for i in undersampled_indices]
            labelr = [item['label'] for item in self.examples]
            majority_class_indicesr = [i for i, labelr in enumerate(labelr) if labelr == 0]
            minority_class_indicesr = [i for i, labelr in enumerate(labelr) if labelr == 1]
            print("length of maj")
            print(len(majority_class_indicesr))
            print("length of min")
            print(len(minority_class_indicesr))
            self.examples = undersampled_X


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
       
        return self.examples[idx]


class ChangepointNormsClassifier(nn.Module):
    def __init__(self, encoder,downsample,num_layers):
        super().__init__()
        # config = AutoConfig.from_pretrained(encoder)
        # config.update({'conv_stride': downsample})
        #self.model = AutoModel.from_pretrained(encoder,config)
        self.model = AutoModel.from_pretrained(encoder)

        # TODO: make the complexity of the classifier configurable (eg, more layers, etc)
        # self.classifier = nn.Linear(self.model.config.hidden_size, 1)
        if num_layers == 0:
            self.classifier = nn.Linear(self.model.config.hidden_size, 1)
        else:
            layers = []
            input_size = self.model.config.hidden_size
            for _ in range(num_layers):
                layers.append(nn.Linear(input_size, 256))
                layers.append(nn.ReLU())
                input_size = 256
            layers.append(nn.Linear(input_size, 1))
            self.classifier = nn.Sequential(*layers)
        self.early_stopping = EarlyStopping(patience=3, delta=0.01)


    def forward(self, inputs):
        print("input")
        print(inputs['input_ids'])
        print(inputs['attention_mask'])
        outputs = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)

        return logits

        # pass CLS token representation through classifier
        # logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        # return logits

def tokenize(batch, tokenizer, args):
   
    if args.include_utterance:
        return tokenizer(
            batch['norms'],
            batch['utterance'],
            return_tensors='pt',
            padding=True,
            truncation=True
        )
    else:
        return tokenizer(
            batch['norms'],
            return_tensors='pt',
            padding=True,
            truncation=True
        )
    
def filter_file_system_preds(file_system_preds, text_char_threshold, time_sec_threshold, filtering):
    file_system_preds = list(sorted(
        file_system_preds,
        key=lambda file_pred: float(file_pred['timestamp'])
    ))

    if len(file_system_preds) <= 1:
        return file_system_preds

    if file_system_preds[0]['type'] == 'text':
        distance_threshold = text_char_threshold
    else:
        assert file_system_preds[0]['type'] in {'audio', 'video'}
        distance_threshold = time_sec_threshold

    to_remove = set()
    while True:
        candidates = []
        remaining_idxs = list(sorted(set(range(len(file_system_preds))) - to_remove))

        if len(remaining_idxs) <= 1:
            break

        for i in range(len(remaining_idxs)):
            distance_before, distance_after = -1, -1

            remaining_idx = remaining_idxs[i]
            if i > 0:
                before_idx = remaining_idxs[i - 1]
                distance_before = file_system_preds[remaining_idx]['timestamp'] - file_system_preds[before_idx][
                    'timestamp']
            else:
                before_idx = None

            if i < len(remaining_idxs) - 1:
                after_idx = remaining_idxs[i + 1]
                distance_after = file_system_preds[after_idx]['timestamp'] - file_system_preds[remaining_idx][
                    'timestamp']
            else:
                after_idx = None

            # if the adjacent predictions are too close, we should consider removing it
            if max(distance_before, distance_after) < distance_threshold:
                if filtering == 'highest':
                    sort_key = -1 * float(file_system_preds[remaining_idx]['llr'])
                elif filtering == 'lowest':
                    sort_key = float(file_system_preds[remaining_idx]['llr'])
                elif filtering == 'most_similar':
                    sort_key = -1 * math.inf
                    if before_idx is not None:
                        sort_key = max(
                            sort_key,
                            abs(
                                float(file_system_preds[remaining_idx]['llr']) -
                                float(file_system_preds[before_idx]['llr'])
                            )
                        )

                    if after_idx is not None:
                        sort_key = max(
                            sort_key,
                            abs(
                                float(file_system_preds[remaining_idx]['llr']) -
                                float(file_system_preds[after_idx]['llr'])
                            )
                        )
                else:
                    raise ValueError(f'Unknown filtering type: {filtering}')

                candidates.append((sort_key, remaining_idx))

        if len(candidates) == 0:
            break

        candidates = list(sorted(candidates))
        to_remove.add(candidates[0][1])

    return [
        file_system_preds[i] for i in range(len(file_system_preds)) if i not in to_remove
    ] 

def filter_system_preds(system_preds, text_char_threshold, time_sec_threshold, filtering, n_jobs=1):
    if filtering == 'none':
        return system_preds

    assert filtering in {'highest', 'lowest', 'most_similar'}

    by_file = defaultdict(list)
    for system_pred in system_preds:
        by_file[system_pred['file_id']].append(system_pred)

    if n_jobs == 1:
        filtered_system_preds = []
        for file_id, file_system_preds in tqdm(by_file.items(), desc='filtering system predictions', leave=False):
            filtered_system_preds.extend(
                filter_file_system_preds(file_system_preds, text_char_threshold, time_sec_threshold, filtering)
            )
    else:
        by_file_preds = list(by_file.values())
        random.shuffle(by_file_preds)
        with Pool(n_jobs) as pool:
            filtered_system_preds = list(
                tqdm(
                    pool.imap_unordered(
                        partial(
                            filter_file_system_preds,
                            text_char_threshold=text_char_threshold,
                            time_sec_threshold=time_sec_threshold,
                            filtering=filtering
                        ),
                        by_file_preds,
                        chunksize=50
                    ),
                    total=len(by_file_preds),
                    desc='filtering system predictions',
                    leave=False
                )
            )
        filtered_system_preds = [
            system_pred for file_system_preds in filtered_system_preds
            for system_pred in file_system_preds
        ]

    return filtered_system_preds

# # def calculate_average_precision(
#         refs, hyps,
#         text_char_threshold=100,
#         time_sec_threshold=10,
#         filtering='none',
#         n_jobs=1
# ):
#     hyps = filter_system_preds(
#         hyps, text_char_threshold,
#         time_sec_threshold, filtering, n_jobs=n_jobs
#     )

#     # NIST uses non-zero values of "Class" to indicate annotations / predictions
#     # in LDC's randomly selected annotation regions
#     for ref in refs:
#         ref['Class'] = ref['timestamp']
#         ref['start'] = ref['timestamp']
#         ref['end'] = ref['timestamp']

#     for hyp in hyps:
#         hyp['Class'] = hyp['timestamp']
#         hyp['start'] = hyp['timestamp']
#         hyp['end'] = hyp['timestamp']

#     ref_df = pd.DataFrame.from_records(refs)
#     hyp_df = pd.DataFrame.from_records(hyps)

#     output_dir = 'tmp_scoring_%s' % os.getpid()
#     os.makedirs(output_dir, exist_ok=True)

#     score_cp(
#         ref_df, hyp_df,
#         delta_cp_text_thresholds=[text_char_threshold],
#         delta_cp_time_thresholds=[time_sec_threshold],
#         output_dir=output_dir
#     )

#     APs, score_df = {}, pd.read_csv(
#         os.path.join(output_dir, 'scores_by_class.tab'), delimiter='\t'
#     )
#     for _, row in score_df[score_df['metric'] == 'AP'].iterrows():
#         APs[row['genre']] = float(row['value'])

#     shutil.rmtree(output_dir)

#     return APs


def calculate_llrs(logits):
    assert len(logits.shape) == 1
    probs = torch.sigmoid(logits)
    print(probs)
    return torch.log(probs / (1 - probs))

def get_ldc_changepoints(split):
    assert split in {'INTERNAL_TRAIN', 'INTERNAL_VAL', 'INTERNAL_TEST'}

    changepoints = []
    #change later
    for file_info in load_ldc_data(include_preprocessed_audio_and_video=False, use_cache=True).values():
        if split not in file_info['splits'] :
            continue

        for changepoint in file_info['changepoints']:
            changepoint['file_id'] = file_info['file_id']
            changepoint['type'] = file_info['data_type']
            changepoint.pop('comment')
            changepoint.pop('annotator')
            changepoints.append(changepoint)

    return changepoints

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str)
    parser.add_argument('--utterances-before', type=int, default=1)
    parser.add_argument('--utterances-after', type=int, default=1)
    parser.add_argument('--include-utterance', action='store_true')
    parser.add_argument('--encoder', type=str, default='xlm-roberta-base')
    parser.add_argument('--regularisation', type=str, default='l2')
    parser.add_argument('--downsample', type=int, default=2)
    # TODO: support different learning rates for pretrained encoder and randomly initialized classificatin head (should be higher)
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-5)
    parser.add_argument('--lrscheduler', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--classifierlayers', type=int, default=0)
    parser.add_argument('--confident-only', action='store_true')

    args = parser.parse_args()

    l1_lambda = 0.01
    l2_lambda = 0.01
    dropout_prob = 0.1
    # PRIORITY 1
    # TODO: create output directory for experiment results using variant name, store argument configs there as config.json

    torch.manual_seed(args.seed)


    if args.device == 'cuda':
        assert torch.cuda.is_available()
    device = torch.device(args.device)

    model = ChangepointNormsClassifier(args.encoder,args.downsample,args.classifierlayers).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    # # TODO: support configurable weight decay, higher LR for classification head
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if(args.lrscheduler):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train, valid, test = ChangepointNormsDataset('INTERNAL_TRAIN', args.utterances_before, args.utterances_after,args.confident_only), \
                   ChangepointNormsDataset('INTERNAL_VAL', args.utterances_before, args.utterances_after,args.confident_only),\
                      ChangepointNormsDataset('INTERNAL_TEST', args.utterances_before, args.utterances_after,args.confident_only)
   
    
    train_loader, valid_loader, test_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True), \
        DataLoader(valid, batch_size=2 * args.batch_size),DataLoader(test, batch_size=args.batch_size)

   
    best_loss = math.inf
    metrics_dict=[]
    epochs_tab = []
    train_loss_tab =[]
    val_loss_tab=[]
    for epoch in range(args.epochs):
        model = model.train()
        t_tot_loss, t_preds, t_labels = 0, [], []
        for t_batch in tqdm(train_loader, desc='training epoch %s' % epoch, leave=False):
            optimizer.zero_grad()
            t_tokenized = tokenize(
                t_batch, tokenizer, args
            ).to(device)
            
            t_logits = model(t_tokenized)
            actual = t_batch['label']
            actual = actual.unsqueeze(1)



            t_loss = nn.BCEWithLogitsLoss()(
                t_logits.float(), actual.float().to(device))
            
            if(args.regularisation=='l1'):
                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.norm(param, 1)
                
                t_loss += l1_lambda * regularization_loss

            if(args.regularisation=='l1'):
                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.norm(param, 2)
                
                t_loss += l2_lambda * regularization_loss
            
            if(args.regularisation=='dropout'):
                t_logits = torch.nn.functional.dropout(t_logits, p=dropout_prob)

        

            t_loss.backward()

            optimizer.step()

            t_tot_loss += t_loss.item()
            t_preds.extend(torch.sigmoid(t_logits).detach().cpu().numpy())
            t_labels.extend(t_batch['label'].numpy())
        



        
        model = model.eval()
        tp,tn,fp,fn=0,0,0,0
        v_tot_loss, v_preds, v_labels = 0, defaultdict(list), defaultdict(list)
        v_file_ids, v_data_types, v_timestamps, v_llrs = [], [], [], []
        for v_batch in tqdm(valid_loader, desc='validating epoch %s' % epoch, leave=False):
            v_tokenized = tokenize(
                v_batch, tokenizer, args
            ).to(device)
            
            v_logits = model(v_tokenized)
            print("logits")
            print(v_logits)
            crazy = v_batch['label']
            crazy = crazy.unsqueeze(1)
            v_loss = nn.BCEWithLogitsLoss()(
                v_logits.float(), crazy.float().to(device)
            )

            for i, val in enumerate(torch.sigmoid(v_logits)):
             
                if val<0.5 and crazy[i]==0:
                    tn+=1
                elif val<0.5 and crazy[i]==1:
                    fn+=1
                elif val>0.5 and crazy[i]==0:
                    fp+=1
                else:
                    tp+=1
            


            v_tot_loss += v_loss.item()
            for v_file, v_pred, v_label in zip(
                v_batch['file_id'], torch.sigmoid(v_logits).detach().cpu().numpy(), v_batch['label'].numpy()
            ):
                v_preds[v_file].append(v_pred)
                v_labels[v_file].append(v_label)
                v_file_ids.extend(v_batch['file_id'])
                v_data_types.extend(v_batch['data_type'])
                v_timestamps.extend(v_batch['timestamp'])
                v_llrs.extend(calculate_llrs(v_logits.squeeze()).detach().cpu().numpy().tolist())

            
            
        valid_ldc_predictions = [
            {
                'file_id': file_id,
                'type': data_type,
                'timestamp': timestamp.item(),
                'llr': llr
            } for file_id, data_type, timestamp, llr in zip(
                v_file_ids, v_data_types, v_timestamps, v_llrs
            )
        ]

        valid_ldc_changepoints = get_ldc_changepoints('INTERNAL_VAL')
        
        print('end of epoch:  ' + str(epoch))
        print('train loss: ' + str(t_tot_loss/len(t_labels)))
        print('validation loss: ' +  str(v_tot_loss/len(v_labels)))

        # print(valid_ldc_changepoints[0])
        # print("-------------")
        # print(valid_ldc_predictions[0])

        with open('yourpfile1'+str(args.regularisation)+str(args.learning_rate)+str(args.include_utterance)+
              str(args.downsample)+str(args.lrscheduler)+str(args.classifierlayers)+str(args.confident_only)+'.txt', 'w') as f:
            for line in valid_ldc_changepoints:
                f.write(f"{line}\n")
        with open('yourpfile2'+str(args.regularisation)+str(args.learning_rate)+str(args.include_utterance)+
              str(args.downsample)+str(args.lrscheduler)+str(args.classifierlayers)+str(args.confident_only)+'.txt', 'w') as f:
            for line in valid_ldc_predictions:
                f.write(f"{line}\n")


        average_val_precision = calculate_average_precision(valid_ldc_changepoints,valid_ldc_predictions)
        
        print('average val precision: '+ str(average_val_precision))
        val_precision=0
        val_recall=0
        val_f1=0
        val_accuracy=0
        if(tp+fp!=0):
            val_precision = tp/(tp+fp)
        if(tp+fn!=0):
            val_recall = tp/(tp+fn)
        if(tp+tn+fp+fn!=0):
            val_accuracy= ((tp+tn)/(tp+tn+fp+fn))
        if(val_precision+val_recall!=0):
            val_f1 = val_precision*val_recall/(val_precision+val_recall)
        print('validation precision: ' +  str(val_precision))
        print('validation recall: ' +  str(val_recall))
        print('validation accuracy: ' +  str(val_accuracy))
        print('validation f1 score: ' +  str(val_f1))
       
       

        if v_tot_loss < best_loss:
            best_loss = v_tot_loss
            best_weights = copy.deepcopy(model.state_dict())
            if(args.checkpoint):
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': v_tot_loss/len(v_labels),
                }, "best_weight.pt")
        
        if(args.lrscheduler):
            scheduler.step()
        
        metrics_dict.append(
        {
        'Epoch':epoch,
        'Train loss':t_tot_loss/len(t_labels), 
        'Val loss':v_tot_loss/len(v_labels),
        'Average val precision':average_val_precision,
        'val precision':val_precision,
        'val recall':val_recall,
        'val accuracy':val_accuracy,
        'val f1':val_f1,
        }
        )

        

        epochs_tab.append(epoch)
        train_loss_tab.append(t_tot_loss/len(t_labels))
        val_loss_tab.append(v_tot_loss/len(v_labels))

        # model.early_stopping(average_val_precision)
        # if model.early_stopping.early_stop:
        #     print('Early stopping')
        #     break

      

    model.load_state_dict(best_weights)
    model = model.eval()
    te_tot_loss, te_preds, te_labels = 0, defaultdict(list), defaultdict(list)
    for te_batch in tqdm(test_loader,  leave=False):
        te_tokenized = tokenize(
            te_batch, tokenizer, args
        ).to(device)
        te_logits = model(te_tokenized)
        crazy = te_batch['label']
        crazy = crazy.unsqueeze(1)
        te_loss = nn.BCEWithLogitsLoss()(
            te_logits.float(), crazy.float().to(device)
        )

        te_tot_loss += te_loss.item()
        for te_file, te_pred, te_label in zip(
            te_batch['file_id'], torch.sigmoid(te_logits).detach().cpu().numpy(), te_batch['label'].numpy()
        ):
            te_preds[te_file].append(torch.sigmoid(te_logits))
            te_labels[te_file].append(te_label)


    print('test loss: '+ str(te_tot_loss/len(te_labels)))

    
    

    fields = ['Epoch', 'Train loss', 'Val loss','Average val precision','val precision','val recall','val accuracy','val f1'] 

    with open('model_metrics'+str(args.regularisation)+str(args.learning_rate)+str(args.include_utterance)+
              str(args.downsample)+str(args.lrscheduler)+str(args.classifierlayers)+str(args.confident_only)+'.csv', 'w', newline='') as file: 
        writer = csv.DictWriter(file, fieldnames = fields)
        writer.writeheader() 
        writer.writerows(metrics_dict)
    print('adding test loss')
    with open('model_metrics'+str(args.regularisation)+str(args.learning_rate)+str(args.include_utterance)+
              str(args.downsample)+str(args.lrscheduler)+str(args.classifierlayers)+str(args.confident_only)+'.csv', 'w', newline='') as file: 
        file.write('test_loss: '+ str(te_tot_loss/len(te_labels)))
    
    #fields = ['train loss', 'val loss'] 

    # with open('epoch_losses.csv', 'w', newline='') as file: 
    #     writer = csv.DictWriter(file, fieldnames = fields)
    #     writer.writeheader() 
    #     writer.writerows(metrics_dict)

    

    # plt.plot(epochs_tab, train_loss_tab,label='train')
    # plt.plot(epochs_tab, val_loss_tab,label='val')
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss per sample")
    # plt.legend()
    # plt.savefig('metric.png')


    

      
