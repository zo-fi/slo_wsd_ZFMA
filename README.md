# Slovene WSD [MA thesis]

This repo contains code and datasets related to a MA thesis on Eng/Slo WSD NLP. A detailed description of the project can be found at [thesis_link].

1. Use Ideas
   
The repo will not work out of the box, since the trained models and datasets are larger than GitHub's preferences [1]. Some training files were reused multiple times. The folder structure is different than in its development environment (remote server). However, some code may be of use for similar NLP projects such as BERT training with layer freezing, MFS prediction or model comparison frameworks. Includes heavily filtered slovene WSD collection and a complementary out-of-vocabulary collection. The two most successful models can have been uploaded to Huggingface Hub and can be retrieved with the command below:
```
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("zo-fi/sloWSD_100slo")
model = AutoModel.from_pretrained("zo-fi/sloWSD_100slo")
#or
tokenizer = AutoTokenizer.from_pretrained("zo-fi/sloWSD_20mix")
model = AutoModel.from_pretrained("zo-fi/sloWSD_20mix")
```
See hugginface_hub documentation for additional information [3]. Loading the models does not require a Huggingface Hub account.


2. Context
   
We trained a WSD model for Slovene with limited coverage (4633 senses for 1597 lemmas). The training was done by combining existing sentences into a combo_df with (non)matching sentence pairs (no sense definitions were used). A mixed SLO-ENG train dataset was also used effectively. A complementary out-of-vocabulary DF was used and we found a (non-significant) drop in OOV performance with larger training sets. Training was done using a 16 GB GPU, taking around 4 hours per model (depending on the number of included sentence combinations; enabled by layer freezing, using limited tokenizer size and gradient accumulation steps). Using tmux is highly recommended to avoid interruptions during training.

3. Repo overview
   
_hyperparam_srch.py_ --> hyperparameter optimization with Optuna on a smaller train_df

_model_train.py _--> basic training loop with layer freezing

_binhead_predictions.py_ --> sentence pair to sense match predictions (binary) and softmax (probability)

_nn_predictions.py_ --> nearest neighbor predictions between test_df sentences and (simple) sense embeddings from train & val_df sentences

_preprocessing.ipynb_ --> notebook with procedures relating to data cleaning, filtering and generating combo_dfs

_semcor2sent_pairs.py_ --> helper script for generating sentence pairs from SemCor [2] 

_testing.ipynb_ --> notebook with testing procedures. Uses .csv files of predictions from multiple models rather than evaluating each model individually

_wsd_data_ --> Contains .csv files used to create Slovene dfs with sentence combinations. Also a complementary OOV df.


**Further links**:

[1] 
To load the basic CSE BERT used in training, follow the instructions at [https://huggingface.co/EMBEDDIA/crosloengual-bert].

[2]
Not included in this repo. See http://lcl.uniroma1.it/wsdeval/training-data or similar. I did encounter issues with some XML files for this collection (missing head tags that needed to be added manually so Python would parse the XML).
[3]
https://huggingface.co/docs/huggingface_hub/quick-start


