 
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from sklearn.metrics import classification_report
 
# Load our data


 
def evaluate_performance(y_true, y_pred):
  """Precision measures how many of the items found are relevant,
which indicates the accuracy of the relevant results.
Recall refers to how many relevant classes were found, which
indicates its ability to find all relevant results.
Accuracy refers to how many correct predictions the model makes
out of all predictions, which indicates the overall correctness of the
model.
The F1 score balances both precision and recall to create a model’s
overall performance"""
 
  performance = classification_report(
  y_true, y_pred,
  target_names=["Negative Review", "Positive Review"]
  )
  print(performance)

data = load_dataset("rotten_tomatoes")

# Load model and tokenizer

mytoken =" "
login(token=mytoken)
from transformers import pipeline
# Path to our HF model
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Load model into pipeline
pipe = pipeline(
 model=model_path,
 tokenizer=model_path,
 return_all_scores=True,
 device="cuda:0"
)

y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "text")),total=len(data["test"])):
 negative_score = output[0]["score"]
 positive_score = output[2]["score"]
 assignment = np.argmax([negative_score, positive_score])
 y_pred.append(assignment)

 
evaluate_performance(data["test"]["label"], y_pred)

