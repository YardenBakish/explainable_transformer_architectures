import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_CACHE'] = '/home/ai_center/ai_users/yardenbakish/'
os.environ['HF_HOME'] = '/home/ai_center/ai_users/yardenbakish/'


from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch import nn
import copy
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import torch
from datasets import load_dataset

import pyarrow.parquet as pq


def load_squad():
    
    df = load_dataset("rajpurkar/squad_v2")

  
    df= df['validation'].to_pandas()
    return df



class Squad_Dataset(Dataset):

  def __init__(self, context, questions, answers, tokenizer, max_len):
    self.questions = questions
    self.context = context
    self.answers = answers

    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.questions)
  
  def __getitem__(self, item):
    question = str(self.question[item])
    answer = self.answers[item]["text"]
    context = self.context[item]


    prompt = f"Use the context to answer the question.\nUse few words.\nContext:{context}\nQuestion:{question}\nAnswer:"
    print(prompt)
    exit(1)
    encoding = self.tokenizer(
      prompt,
      add_special_tokens=True,
      #max_length=self.max_len,
      return_token_type_ids=True,
      #pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      #truncation = True
    )

    return {
      'prompt_text': prompt,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'token_type_ids': encoding['token_type_ids'].flatten(),
      'targets': answer
    }

def create_data_loader(df, tokenizer, max_len, batch_size,debug=False):
  ds = Squad_Dataset(
    context =df.context.to_numpy(),
    questions =df.question.to_numpy(),
    answers =df.answers.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  if debug:
    ds = Subset(ds, range(1500))

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0
  )
