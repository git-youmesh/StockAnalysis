from datasets import load_dataset
from datasets import ClassLabel,Dataset,DatasetDict
import random
import pandas as pd
import math
import torch
from transformers import DataCollatorForLanguageModeling
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
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
                X.append(X_Str + "  " + "[MASK]")
                y.append(str(seq_y))       
                 
        except Exception as E:
            print(E)
        return X,y
def tokenize_function(examples):
    return tokenizer(examples["text"])


model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
MASK_TOKEN_ID = tokenizer.mask_token_id
MASK_TOKEN = tokenizer.convert_ids_to_tokens(tokenizer.mask_token_id)
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
datasets["train"][10]
infy = pd.read_csv("INFY.csv")
train_x,train_y = split_sequence(infy['close'])
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
model = AutoModelForCausalLM.from_pretrained("./model")
#tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text", "label"])
text = St.tail(1)['text'].values[0] 
lable = St.tail(1)['label'].values[0]  
print(text,lable)
t = tokenizer.encode(text, return_tensors="pt")
res = model.generate(t)
print(tokenizer.decode(res[0], skip_special_tokens=False))



inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")











