# Code used to train models
# 1000 eval steps for largest df, 200 for 20% ones. Early stopping at 3 evaluations without improvement
# uses layer freezing and early stopping
# source data and savepath for new model need to be switched out accordingly.


from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback, IntervalStrategy
import evaluate
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import pandas as pd

train_src = "wsd_data/eng_wing/mix_20.csv"#"wsd_data/combo_dfs/combo_train_20.csv"
slobert = "wsd_data/BERTS/cse_bert"
data_path = "wsd_data/combo_dfs/combo_ready/"

#512: crash
#256: crash
#180: no crash
#128: no crash
#80:  no crash
tokenizer = AutoTokenizer.from_pretrained(slobert)

train_df = pd.read_csv(train_src).rename(columns = {"text_pair":"example"})
val_df = pd.read_csv(data_path + 'combo_val_fin.csv').rename(columns = {"text_pair":"example"})

#removing unneeded columns, shuffle
train_df = train_df[["sent1", "sent2", "label"]].sample(frac=1) #shuffle je dobra ideja ajges
val_df = val_df[["sent1", "sent2", "label"]].sample(frac=1)


train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
SWSD_dataset_dict = DatasetDict({"train" : train_dataset, "validation" : val_dataset})

def encode(examples):
  out = tokenizer(examples["sent1"], examples["sent2"], truncation = True, padding = "max_length", max_length = MAX_LEN)
  return out

encoded_dataset = SWSD_dataset_dict.map(encode, batched=True)
encoded_dataset.set_format(columns=['attention_mask', 'input_ids', 'token_type_ids', 'label'])

model = AutoModelForSequenceClassification.from_pretrained(slobert, return_dict=True, num_labels=2)
freeze_list = ["bert.encoder.layer.{}".format(str(i)) for i in range(0, 8)]
for name, param in model.named_parameters():
    if any([name.startswith(ban_param) for ban_param in freeze_list]):
        param.requires_grad = False

        
training_args = TrainingArguments(
    "test",
    #Training hyperparams
    learning_rate = 3e-5,
    num_train_epochs = 2,
    gradient_accumulation_steps = 16,
    per_device_train_batch_size = 48,
    #Eval approach strategy 
    evaluation_strategy = "steps", 
    save_total_limit = 4, 
    eval_steps = 200,
    load_best_model_at_end = True,
    save_strategy = "steps",
    save_steps = 200,
    logging_steps = 200,
    logging_strategy = "steps"
    #evaluation_strategy = "epoch", 
    #save_total_limit = 4, 
    #logging_strategy = "epoch", 
    #save_strategy = "epoch"
)

trainer = Trainer(
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=encoded_dataset["train"], 
    eval_dataset=encoded_dataset["validation"],
    model=model,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

#model.save_pretrained("wsd_data/big_early_ssbert")
model.save_pretrained("wsd_data/BERTS/trained_20mix")

