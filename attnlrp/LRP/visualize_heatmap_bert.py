import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config

#from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from lxt.models.bert_PE import BertForSequenceClassification, attnlrp

from helper_scripts.helper_functions import update_json
from datasets.imdb import load_imdb, MovieReviewDataset, create_data_loader
from tqdm import tqdm
from lxt.utils import pdf_heatmap, clean_tokens


import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import copy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent numerical overflow
)


model = BertForSequenceClassification.from_pretrained("lannelin/bert-imdb-1hidden", device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage = True)
#model = LlamaForCausalLM.from_pretrained(path, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("lannelin/bert-imdb-1hidden")

# optional gradient checkpointing to save memory (2x forward pass)
model.gradient_checkpointing_enable()

# apply AttnLRP rules
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model.to(device)


#attnlrp.register(model)

MAX_LEN = 512
BATCH_SIZE = 1
RANDOM_SEED = 42
df = load_imdb()
df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# optional gradient checkpointing to save memory (2x forward pass)
model.gradient_checkpointing_enable()


for name, param in model.named_parameters(): 
    if  param.requires_grad == True:
        print(name)  
#exit(1)

# apply AttnLRP rules
attnlrp.register(model)

count = 0
for d in tqdm(test_data_loader):
    if count !=11:
        count+=1
        continue
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    
    input_embeds = model.get_input_embeddings()(input_ids)
    
    #print("A")
    #print(input_embeds)
    #print("\n\n")
    #print(tokenizer.convert_ids_to_tokens(input_ids[0]))
    outputs = model(
        inputs_embeds = input_embeds.requires_grad_(),
        #input_ids=input_ids,
        #use_cache=False,
        #attention_mask=attention_mask
      )['logits']
  
    #input_embeds = model.get_input_embeddings()(input_ids)
   
    #print("B")
    #print(input_embeds)
    #print("\n\n")
  
    max_logits, max_indices = torch.max(outputs, dim=1)
    print(max_indices)
    print(targets)

    
    max_logits.backward(max_logits)
    #print(input_embeds.grad.float())
    #exit(1)
    relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision

    relevance = relevance / relevance.abs().max()
    

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)



    pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')


       