from transformers import AutoTokenizer, BertForSequenceClassification
from datasets import load_dataset
import pandas as pd 
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pandas as pd
import numpy as np,array
def split_sequence(sequence, n_steps=13,Lag=1):
      try:
            sequence = sequence.values
             
            X, y = list(), list()
            for i in range(len(sequence)-n_steps ):
            # gather input and output parts of the pattern
                seq_x= sequence[i:i+n_steps] 
                X.append(seq_x)                 
      except Exception as E:
            print(E)
      return X
df = pd.read_csv("INFY.csv")
 
examples = split_sequence(df['close'] , 14,6)

data_values =[]
for X in examples:
      data_values.append(str(X).replace('[','').replace(']',''))
examples= data_values

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(examples)
print(embeddings.shape)
def semantic_search(query,embeddings, examples):
  query_embedding = model.encode(query) #A
  scores =[]
  for j in range(len(embeddings)):
    scores.append(scipy.spatial.distance.cosine(query_embedding, embeddings[j]))
  print("Most similar example:", examples[scores.index(min(scores))])

semantic_search("1348", embeddings, examples)




