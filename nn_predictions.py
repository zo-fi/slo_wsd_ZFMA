from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch
import pandas as pd
import os
import numpy as np
tqdm.pandas()
#goes over multiple BERT models, saves all predictions (nn) in a single .csv file

src_path = "wsd_data/slo_sense_1.5_splits/"
src1 = pd.read_csv(src_path + "train_nopair.csv")
src2 = pd.read_csv(src_path + "val_nopair.csv")
train_df = pd.concat([src1, src2])
test_df = pd.read_csv(src_path + 'test_nopair_fix.csv')

model_path = "wsd_data/BERTS/"
tokenizer_path = model_path + "cse_bert"
target_berts = [i for i in os.listdir("wsd_data/BERTS/") if "." not in i]


def model_init(target_bert):
    return SentenceTransformer(target_bert)

def query2label(sent, sense_emb_storage, senseID_storage):
    query_embedding = model.encode(sent)
    cos_scores = util.cos_sim(query_embedding, sense_emb_storage)[0]
    target_ind = cos_scores.argmax()
    return senseID_storage[int(target_ind)]

def generate_preds(model):
    #does the preds for a single model
    senseID_storage = []
    sense_embed_storage = []
    #np.zeros(shape=(4633, 768))
    for senseID, group in train_df.groupby("senseID"):
        senseID_storage.append(senseID)
        sent_list = group.sent.to_list()
        embeddings = model.encode(sent_list)
        sense_embed = np.mean(embeddings, axis = 0)
        sense_embed_storage.append(sense_embed)
    nn_preds = test_df.sent.progress_apply(lambda x: query2label(x, sense_embed_storage, senseID_storage))
    return nn_preds

result_holder = pd.DataFrame([])
for model_name in target_berts:
    tqdm.pandas(desc = 'Working on {}'.format(model_name)) 
    temp_df = pd.DataFrame()
    model = model_init(model_path + model_name)
    encoded = generate_preds(model)
    encoded.columns = ["{}_nn_pred".format(model_name)]
    result_holder = pd.concat([result_holder, encoded], axis = 1)
    
result_holder.to_csv("wsd_data/testing_dump/nn_testing.csv")