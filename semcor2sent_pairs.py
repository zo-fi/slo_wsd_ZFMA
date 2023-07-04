import pandas as pd
from tqdm import tqdm
from itertools import combinations
from itertools import product
import math

sc = pd.read_csv("wsd_data/eng_wing/semcor_filt_nopair.csv")
combo_majors = sc.groupby("senseID").filter(lambda x: x.sent.count() >= 20)
combo_maj_downscale = combo_majors.groupby("senseID").sample(20)
combo_minors = sc.drop(combo_majors.index)
combo_src = pd.concat([combo_minors, combo_maj_downscale])
#filtered semcor into a dataset of sentence combinations


def sent_combos(target_df):
  #exhaustive combinations of examples in the dataset
  pair_df = pd.DataFrame(columns=['text_pair','sense1','sense2', "sent1_ind", "sent2_ind", "lemma",'label'])
  sense_groups = target_df.groupby("lemma")
  for lemma, body in tqdm(sense_groups):
    tupl = [(sent, sense, sent_ind) for sent, sense, sent_ind in zip(body.sent, body.senseID, body.sentID)]
    combos = [i for i in combinations(tupl, 2)]
    for pair in combos:
      pack1, pack2 = pair
      t1, sense1, sent_ind1 = pack1
      t2, sense2, sent_ind2 = pack2
      text_pair = t1 + " [SEP] " + t2
      label = 1 if (sense1 == sense2) else 0
      pair_df.loc[len(pair_df.index)] = [text_pair, sense1, sense2, sent_ind1, sent_ind2, lemma, label] 
  return pair_df

combo_df = sent_combos(combo_src)

def balance_plz(df):
    #Downsamples negative class examples
    pos = df[df.label == 1].copy()
    neg = df[df.label == 0].copy()
    req_prop = round(len(pos)/len(neg), 3)
    neg_down = neg.groupby("senseID2x", group_keys = False).apply(lambda x: x.sample(frac = req_prop, random_state=101))
    return pd.concat([pos, neg_down])

print(combo_df.label.value_counts())
combo_df_balanced = balance_plz(combo_df)
print(combo_df_balanced.label.value_counts())

combo_df_eng.to_csv("wsd_data/eng_wing/combo_df_eng.csv")
combo_df_eng_bal.to_csv("wsd_data/eng_wing/combo_df_eng_bal.csv")