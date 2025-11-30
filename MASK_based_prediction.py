from datasets import load_dataset

from datasets import ClassLabel,Dataset,DatasetDict
import random
import pandas as pd
import math
import torch
from transformers import DataCollatorForLanguageModeling
import torch
from transformers import AutoTokenizer
import numpy as np,array
def split_sequence(sequence, n_steps=13,Lag=6):
        try:
            sequence = sequence.values
            X, y = list(), list()
            for i in range(len(sequence)):
            # find the end of this pattern
                end_ix = i + n_steps
            # check if we are beyond the sequence
                if end_ix > len(sequence)-1*Lag:
                    break
            # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix+Lag-1]
                seq_x = seq_x.astype('int64')
                seq_y = seq_y.astype('int64')
                X_Str = str(seq_x).replace('[',"").replace(']',"")
                X.append(X_Str + "  " + str(seq_y))
                y.append(str(seq_y))       
                 
        except Exception as E:
            print(E)
        return X,y

model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
MASK_TOKEN_ID = tokenizer.mask_token_id
MASK_TOKEN = tokenizer.convert_ids_to_tokens(tokenizer.mask_token_id)
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
datasets["train"][10]
infy = pd.read_csv("INFY.csv")
train_x,train_y = split_sequence(infy['close'])

for x in train_x:
    tokenizer.add_tokens(x)
for x in train_y:
    tokenizer.add_tokens(x)
St  = pd.DataFrame(train_x)
St['label'] = train_y
St.columns =['text','label']
tdf = St[0:int(len(St) * .80)]
vdf = St[int(len(St) * .80):]
tds = Dataset.from_pandas(tdf)
vds = Dataset.from_pandas(vdf)
datasets = DatasetDict()
datasets['train'] = tds
datasets['validation'] = vds

def tokenize_function_mask(examples):

    usage_arr = [ examples['text'][i].replace(examples['label'][i], MASK_TOKEN)  for i in range(len(examples['label']))]
    tokenized_data = tokenizer(usage_arr, padding="max_length", truncation=True)
    label_arr_list = []
    for i in range(len(usage_arr)):

        label_arr = [-100] * len(tokenized_data.input_ids[i])
        if MASK_TOKEN_ID in tokenized_data.input_ids[i]:
            label_arr[tokenized_data.input_ids[i].index(MASK_TOKEN_ID)] = tokenizer.convert_tokens_to_ids(examples['label'][i])
        label_arr_list.append(label_arr)
    tokenized_data['labels'] = label_arr_list
    return tokenized_data

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
  
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    print(df.head(10))
show_random_elements(datasets["train"])

def tokenize_function(examples):
    return tokenizer(examples["text"])
tokenized_datasets = datasets.map(tokenize_function_mask, batched=True, num_proc=4, remove_columns=["text", "label"])
block_size = 128
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=30,
    num_proc=4,
)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
model.resize_token_embeddings(len(tokenizer))

from transformers import Trainer, TrainingArguments


model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-infy2",
    eval_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
trainer.save_model(output_dir="/model/Stocks")
model.load_state_dict(torch.load("path/to/your/local/model.pth"))
