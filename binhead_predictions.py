#tmux code for getting predictions from multiple models (via binary classification softmax)

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, BertForNextSentencePrediction
import torch
from tqdm import tqdm #.progress_apply()
from tqdm.notebook import tqdm_notebook
import torch
import torch.nn.functional as nnf


target_berts = ['trained_20S', 'trained_10S','trained_10E', 'trained_20mix', 'trained_20E']
model_path = "wsd_data/BERTS/"
tokenizer_path = "wsd_data/BERTS/cse_bert"
MAXLEN = 180

test_df = pd.read_csv("wsd_data/combo_dfs/combo_ready/combo_test_fin.csv")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, truncation = True, padding = "max_length", max_length = MAXLEN)

def model_init(target_bert): #needs full path
    return BertForSequenceClassification.from_pretrained(target_bert)

#dobiti predictione
def apply_classi(s1, s2):
    encoded_input = tokenizer(s1, s2, padding=True, truncation=True, max_length=180, return_tensors='pt')
    labels = torch.LongTensor([0])
    predict = model(**encoded_input, labels=labels) #
    probabilities = nnf.softmax(predict.logits, dim=1)
    pos_prob = float(probabilities[0][1])
    classification = int(torch.argmax(predict.logits))
    return [pos_prob, classification]

result_holder = pd.DataFrame([])

for model_name in target_berts:
    tqdm.pandas(desc = 'Working on {}'.format(model_name)) 
    temp_df = pd.DataFrame()
    model = model_init(model_path + model_name)
    encoded = test_df.progress_apply(lambda x: apply_classi(x.sent1, x.sent2), axis = 1, result_type="expand")
    encoded.columns = ["{}_prob".format(model_name).strip("trained_"),"{}_label".format(model_name).strip("trained_")]
    result_holder = pd.concat([result_holder, encoded], axis = 1)
    
result_holder.to_csv("wsd_data/testing_dump/all_reslts_plz.csv")
