from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import evaluate
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import optuna
import pandas as pd

slobert = "wsd_data/BERTS/cse_bert"
data_path = "wsd_data/combo_dfs/combo_ready/"
MAX_LEN = 80 #512 #! reduce if taking forever.
tokenizer = AutoTokenizer.from_pretrained(slobert)
#metric = evaluate.load("matthews_correlation")

train_df = pd.read_csv(data_path + 'combo_train_mini10.csv').rename(columns = {"text":"example"})
val_df = pd.read_csv(data_path + 'combo_val_fin.csv').rename(columns = {"text_pair":"example"})
train_df = train_df[["sent1", "sent2", "label"]].sample(frac=1)
val_df = val_df[["sent1", "sent2", "label"]].sample(frac=1)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
SWSD_dataset_dict = DatasetDict({"train" : train_dataset, "validation" : val_dataset})

def encode(examples):
  encoded = tokenizer(examples["sent1"], examples["sent2"], truncation = True, padding = "max_length", max_length = MAX_LEN)
  return encoded

encoded_dataset = SWSD_dataset_dict.map(encode, batched=True)
encoded_dataset.set_format(columns=['attention_mask', 'input_ids', 'token_type_ids', 'label'])

def model_init():
  model = AutoModelForSequenceClassification.from_pretrained(slobert, return_dict=True, num_labels=2)
  freeze_list = ["bert.encoder.layer.{}".format(str(i)) for i in range(0, 8)]
  for name, param in model.named_parameters():
    if any([name.startswith(ban_param) for ban_param in freeze_list]):
      param.requires_grad = False
  return model

#def compute_metrics(eval_pred):
#  predictions, labels = eval_pred
#  predictions = predictions.argmax(axis = -1)
#  return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test", evaluation_strategy = "epoch", save_total_limit = 4, logging_strategy = "epoch", save_strategy = "epoch")
trainer = Trainer(
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=encoded_dataset["train"], 
    eval_dataset=encoded_dataset["validation"],
    model_init=model_init
)

def optuna_hp_space(trial): #4x3x4x6
    #Suggest_categorical should be avoided when possible, unless it's truly categorical (adam vs ada)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 5e-5, step=1e-5),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 6, step=2),
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 2, 16, step = 2),
        "per_device_train_batch_size": trial.suggest_int("per_device_train_batch_size", 8, 64, step = 8)
    }

best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=30
)

print(best_trial)

#Kaj se potem zgodi z best_trial?
                                                         
