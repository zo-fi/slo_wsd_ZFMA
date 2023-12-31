# Slovene WSD [MA thesis]

This repo contains code and datasets related to a MA thesis on Eng/Slo WSD NLP. A detailed description of the project can be found at [thesis_link] (Slovene).

1. Use Ideas
   
The repo will not work out of the box, since the trained models and datasets are larger than GitHub's preferences. Some support scripts were reused multiple times. The folder structure is different than in its development environment (remote server). However, some code may be of use for similar NLP projects, such as BERT training with layer freezing, model comparison frameworks or WSD-specific code. The repo includes a heavily filtered and cleaned slovene WSD collection and a complementary out-of-vocabulary collection. The two most successful models can have been uploaded to Huggingface Hub and can be retrieved with the commands below:
```
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("zo-fi/sloWSD_100slo")
model = AutoModel.from_pretrained("zo-fi/sloWSD_100slo")

#or

tokenizer = AutoTokenizer.from_pretrained("zo-fi/sloWSD_20mix")
model = AutoModel.from_pretrained("zo-fi/sloWSD_20mix")

#or BertForSequenceClassification.from_pretrained() if used for sense discrimination in sentence pairs with a common lemma.
```
See Hugging Face Hub  documentation for additional options [1]. Loading the models does not require a Hugging Face account. To load the basic CSE BERT used in training, follow the instructions at [https://huggingface.co/EMBEDDIA/crosloengual-bert].

2. Context
   
We trained a WSD model for Slovene with limited coverage (4,633 senses for 1,597 lemmas). The training was done by combining existing sentences into a dataset of (non)matching sentence pairs (no sense definitions were used). A mixed SLO-ENG train dataset was also used effectively. A complementary out-of-vocabulary dataset was used and we found a (non-significant) drop in OOV performance with larger training sets. Training was done using a 16 GB GPU, taking around 4 hours per model (depending on the number of included sentence combinations; enabled by layer freezing, using limited tokenizer size and gradient accumulation steps). Using _tmux_ was invaluable to avoid interruptions during training.

3. Repo overview
   
_hyperparam_srch.py_ --> hyperparameter optimization with Optuna using a smaller train dataset

_model_train.py_--> basic training loop with layer freezing

_binhead_predictions.py_ --> sentence pair to sense match predictions (binary) and softmax (probability)

_nn_predictions.py_ --> nearest neighbor predictions between test dataset sentences and (simple) sense embeddings from trainining & validation dataset sentences

_preprocessing.ipynb_ --> notebook with procedures relating to data cleaning, filtering and generating datasets of sentence combinations

_semcor2sent_pairs.py_ --> helper script for generating sentence pairs from SemCor [2] 

_testing.ipynb_ --> notebook with testing procedures. Uses .csv files of predictions from multiple models rather than evaluating each model individually

_wsd_data_ --> Contains .csv files used to create Slovene datasets of sentence combinations and a complementary OOV dataset.


**Further links**:

[1]
https://huggingface.co/docs/huggingface_hub/quick-start

[2]
Not included in this repo. See [http://lcl.uniroma1.it/wsdeval/training-data] or similar. I did encounter issues with some XML files for in some versions of SemCor (missing head tags that needed to be replaced manually before parsing in Python works).

