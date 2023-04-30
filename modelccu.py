import argparse
from tqdm import tqdm
from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
import copy
import math
from loaders.ldc_data import load_ldc_data
from ldcNormsCombine import generate_input
import os
import sys
import shutil
import pandas as pd
from scoring import score_cp
import csv 
from matplotlib import pyplot as plt



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

class ChangepointNormsDataset(Dataset):
    def __init__(self, split, utterances_before, utterances_after):
        # PRIORITY 1 (once unblocked)
        # TODO: load the LDC dataset using the LDC loader functionality - Amith function use
        # TODO: load the associated UIUC predicted norms for the file_ids in the split
        self.split = split
        self.utterances_before = utterances_before
        self.utterances_after = utterances_after

        self.examples = generate_input(self.split,self.utterances_before,self.utterances_after)
        


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
       
        return self.examples[idx]


class ChangepointNormsClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.model = AutoModel.from_pretrained(encoder)
        # TODO: make the complexity of the classifier configurable (eg, more layers, etc)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        # pass CLS token representation through classifier
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

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
    
def calculate_average_precision(
        refs, hyps,
        text_char_threshold=100,
        time_sec_threshold=10,
        filtering='none'
):
    assert filtering == 'none', 'filtering not yet supported'

    for ref in refs:
        ref['Class'] = str(ref['timestamp'])
        ref['start'] = ref['timestamp']
        ref['end'] = ref['timestamp']

    for hyp in hyps:
        hyp['Class'] = str(hyp['timestamp'])
        hyp['start'] = hyp['timestamp']
        hyp['end'] = hyp['timestamp']

    ref_df = pd.DataFrame.from_records(refs)
    hyp_df = pd.DataFrame.from_records(hyps)

    output_dir = 'tmp_scoring_%s' % os.getpid()
    os.makedirs(output_dir, exist_ok=True)
   
    score_cp(
        ref_df, hyp_df,
        delta_cp_text_thresholds=[text_char_threshold],
        delta_cp_time_thresholds=[time_sec_threshold],
        output_dir=output_dir
    )

    APs, score_df = {}, pd.read_csv(
        os.path.join(output_dir, 'scores_by_class.tab'), delimiter='\t'
    )
    for _, row in score_df[score_df['metric'] == 'AP'].iterrows():
        APs[row['genre']] = float(row['value'])

    shutil.rmtree(output_dir)

    return APs




def get_ldc_changepoints(split):
    assert split in {'INTERNAL_TRAIN', 'INTERNAL_VAL', 'INTERNAL_TEST'}

    changepoints = []
    for file_info in load_ldc_data(include_preprocessed_audio_and_video=True, use_cache=True).values():
        if split not in file_info['splits'] :
            continue

        for changepoint in file_info['changepoints']:
            changepoint['file_id'] = file_info['file_id']
            changepoint['type'] = file_info['data_type']
            changepoints.append(changepoint)

    print(changepoints)
    return changepoints

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str)
    parser.add_argument('--utterances-before', type=int, default=1)
    parser.add_argument('--utterances-after', type=int, default=1)
    parser.add_argument('--downsample', type=float, default=1.0)
    parser.add_argument('--include-utterance', action='store_true')
    parser.add_argument('--encoder', type=str, default='xlm-roberta-base')
    # TODO: support dropout
    # TODO: support different learning rates for pretrained encoder and randomly initialized classificatin head (should be higher)
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-5)
    parser.add_argument('--lrscheduler', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    args = parser.parse_args()

    # PRIORITY 1
    # TODO: create output directory for experiment results using variant name, store argument configs there as config.json

    torch.manual_seed(args.seed)


    if args.device == 'cuda':
        assert torch.cuda.is_available()
    device = torch.device(args.device)

    model = ChangepointNormsClassifier(args.encoder).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    # # TODO: support configurable weight decay, higher LR for classification head
    # # TODO: support learning rate scheduler (linear?)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if(args.lrscheduler):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train, valid, test = ChangepointNormsDataset('INTERNAL_TRAIN', args.utterances_before, args.utterances_after), \
                   ChangepointNormsDataset('INTERNAL_VAL', args.utterances_before, args.utterances_after),\
                      ChangepointNormsDataset('INTERNAL_TEST', args.utterances_before, args.utterances_after)
   
    
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
            
            t_loss.backward()

            optimizer.step()

            t_tot_loss += t_loss.item()
            t_preds.extend(t_logits.detach().cpu().numpy())
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
            crazy = v_batch['label']
            crazy = crazy.unsqueeze(1)
            v_loss = nn.BCEWithLogitsLoss()(
                v_logits.float(), crazy.float().to(device)
            )

            for i, val in enumerate(v_logits):
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
                v_batch['file_id'], v_logits.detach().cpu().numpy(), v_batch['label'].numpy()
            ):
                v_preds[v_file].append(v_pred)
                v_labels[v_file].append(v_label)
                v_file_ids.extend(v_batch['file_id'])
                v_data_types.extend(v_batch['data_type'])
                v_timestamps.extend(v_batch['timestamp'].numpy())
                v_llrs.extend(nn.LogSigmoid()(v_logits).detach().cpu().numpy().tolist())
        valid_ldc_predictions = [
            {
                'file_id': file_id,
                'type': data_type,
                'timestamp': timestamp,
                'llr': llr[0]
            } for file_id, data_type, timestamp, llr in zip(
                v_file_ids, v_data_types, v_timestamps, v_llrs
            )
        ]

        valid_ldc_changepoints = get_ldc_changepoints('INTERNAL_VAL')
        
        print('end of epoch:  ' + str(epoch))
        print('train loss: ' + str(t_tot_loss/len(t_labels)))
        print('validation loss: ' +  str(v_tot_loss/len(v_labels)))

        average_val_precision = calculate_average_precision(valid_ldc_changepoints,valid_ldc_predictions)
        print('average val precision: '+ str(average_val_precision))
        val_precision=0
        val_recall=0
        val_f1=0
        val_accuracy=0
        if(tp+fp!=0 and tp+fn!=0):
            val_precision = tp/(tp+fp)
            val_recall = tp/(tp+fn)
            val_f1 = val_precision*val_recall/(val_precision+val_recall)
            val_accuracy= ((tp+tn)/(tp+tn+fp+fn))
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
        'val f1':val_f1
        }
        )

        epochs_tab.append(epoch)
        train_loss_tab.append(t_tot_loss/len(t_labels))
        val_loss_tab.append(v_tot_loss/len(v_labels))

      

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
            te_batch['file_id'], te_logits.detach().cpu().numpy(), te_batch['label'].numpy()
        ):
            te_preds[te_file].append(te_pred)
            te_labels[te_file].append(te_label)

    print('test loss: '+ str(te_tot_loss/len(te_labels)))

    

    fields = ['Epoch', 'Train loss', 'Val loss','Average val precision','val precision','val recall','val accuracy','val f1'] 

    with open('model_metrics.csv', 'w', newline='') as file: 
        writer = csv.DictWriter(file, fieldnames = fields)
        writer.writeheader() 
        writer.writerows(metrics_dict)

    

    plt.plot(epochs_tab, train_loss_tab,label='train')
    plt.plot(epochs_tab, val_loss_tab,label='val')
    plt.xlabel("Epochs")
    plt.ylabel("Loss per sample")
    plt.legend()
    plt.savefig('metric.png')


    

      
