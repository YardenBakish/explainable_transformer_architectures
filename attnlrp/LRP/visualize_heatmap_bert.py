import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/ai_center/ai_users/yardenbakish/'
os.environ['HF_HOME'] = '/home/ai_center/ai_users/yardenbakish/'
#from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from lxt.models.bert_PE import BertForSequenceClassification, attnlrp

from helper_scripts.helper_functions import update_json
from attDatasets.imdb import load_imdb, MovieReviewDataset, create_data_loader
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



import argparse
parser = argparse.ArgumentParser(description='Train a segmentation')
parser.add_argument('--model-size', type=str,
                        choices=['llama_2_7b', 'llama_tiny'],
                        default = 'llama_tiny',
                       # required = True,
                        help='')
parser.add_argument('--variant', type=str,
                       default="baseline")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--pe', action='store_true')
parser.add_argument('--reform', action='store_true')

parser.add_argument('--quant', action='store_true')

parser.add_argument('--trained_model', type=str,)
parser.add_argument('--clamp', action='store_true')


parser.add_argument('--sequence-length', type=int,
                       )
parser.add_argument('--pe_only', action='store_true')
parser.add_argument('--sep_heads', action='store_true')



parser.add_argument('--no-padding', action='store_true')
parser.add_argument('--without-abs', action='store_true')
parser.add_argument('--single-norm', action='store_true')





args = parser.parse_args()





quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent numerical overflow
)


model = BertForSequenceClassification.from_pretrained("lvwerra/bert-imdb", device_map="cuda", torch_dtype=torch.bfloat16, low_cpu_mem_usage = True)
#model = LlamaForCausalLM.from_pretrained(path, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("lvwerra/bert-imdb")

# optional gradient checkpointing to save memory (2x forward pass)
model.gradient_checkpointing_enable()

# apply AttnLRP rules
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model.to(device)


#attnlrp.register(model)

MAX_LEN = 512
BATCH_SIZE = 1
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
df = load_imdb()
df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE, no_padding = True)

# optional gradient checkpointing to save memory (2x forward pass)
model.gradient_checkpointing_enable()


#for name, param in model.named_parameters(): 
#    if  param.requires_grad == True:
#        print(name)  
#exit(1)

# apply AttnLRP rules



attnlrp.register(model)

count = 0
for d in tqdm(test_data_loader):
    if count !=55:
        count+=1
      
        continue
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    
    input_embeds = model.get_input_embeddings()(input_ids)
    
    if args.pe:
        position_ids = torch.arange(input_ids.shape[-1], device="cuda:0").view(1, -1)

     

        #print(model.config.max_position_embeddings)
        #print(model.config.hidden_size)
        
        position_embeddings = model.bert.embeddings.position_embeddings(position_ids).requires_grad_()
   
        outputs = model(
            inputs_embeds = input_embeds.requires_grad_(),
            position_embeddings = position_embeddings,
         
          )['logits']
    
    else:
    
        outputs = model(
            inputs_embeds = input_embeds.requires_grad_(),

          )['logits']
  

  
    max_logits, max_indices = torch.max(outputs, dim=1)


    
    max_logits.backward(max_logits)
    #print(input_embeds.grad.float())
    #exit(1)
    relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision

    #
    if args.pe:
        PE_relevance = position_embeddings.grad.float().sum(-1).cpu()[0]
        
        relevance+=PE_relevance
   

    relevance = relevance / relevance.abs().max()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)



    pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')
    print("made it")
    exit(1)
       