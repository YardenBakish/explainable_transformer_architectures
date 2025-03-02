import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config
import matplotlib.pyplot as plt
from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from helper_scripts.helper_functions import update_json
from datasets.imdb import load_imdb, MovieReviewDataset, create_data_loader
from tqdm import tqdm
from lxt.utils import pdf_heatmap, clean_tokens
from utils import flip, get_latest_checkpoint
from sklearn.metrics import auc

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
import os


import argparse
parser = argparse.ArgumentParser(description='Train a segmentation')
parser.add_argument('--model-size', type=str,
                        choices=['llama_2_7b', 'llama_tiny'],
                       # required = True,
                        help='')
parser.add_argument('--variant', type=str,
                       default="baseline")
parser.add_argument('--fract', type=float,
                        default=0.3,
                        help='')
parser.add_argument('--trained_model', type=str,)

args = parser.parse_args()
args.dataset = 'imdb'

config.get_config(args, pert=True)
save_dir = f'{args.pretrained_model_path}/pert_results'
os.makedirs(save_dir, exist_ok=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

PATH = args.original_models
MAX_LEN = 512
BATCH_SIZE = 1

#model_checkpoint = "finetuned_models/imdb/llama_tiny/vanilla/dir/pytorch_model.bin"

tokenizer = AutoTokenizer.from_pretrained(PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

#original
#llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb_config, attn_implementation="eager")
llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map="cuda",  attn_implementation="eager")



conf = llamaModel.config
conf.num_labels = 2
conf.pad_token_id = tokenizer.pad_token_id

sequence_length = args.sequence_length
#current
kwargs = {"attn_layer": args.attn_layer, "sequence_length": sequence_length }
last_checkpoint_dir = get_latest_checkpoint(args.pretrained_model_path)
last_checkpoint = f'{last_checkpoint_dir}/pytorch_model.bin' 


model = LlamaForSequenceClassification.from_pretrained(last_checkpoint, config = conf,  torch_dtype=torch.bfloat16, device_map="cuda" ,**kwargs)

model.to(device)




df = load_imdb()
df_train, df_test = train_test_split(df, test_size=args.fract, random_state=RANDOM_SEED)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# optional gradient checkpointing to save memory (2x forward pass)
model.gradient_checkpointing_enable()

# apply AttnLRP rules
attnlrp.register(model)


UNK_token = tokenizer.unk_token_id
fracs = np.linspace(0.,1.,11)

for flip_case in ['generate', 'pruning']:
    all_flips = {}
    all_flips_str = {}

    count = 0

    M,E, EVOLUTION = [],[], []
    M_str,E_str, EVOLUTION_str = [],[], []

    for d in tqdm(test_data_loader):
    

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)


        input_embeds = model.get_input_embeddings()(input_ids)

        outputs = model(
            inputs_embeds = input_embeds.requires_grad_(),
            use_cache=False,
            attention_mask=attention_mask
          )['logits']
    

        max_logits, max_indices = torch.max(outputs, dim=1)
   
        max_logits.backward(max_logits)
        relevance = input_embeds.grad.float().sum(-1).cpu()[0] 

        relevance = relevance / relevance.abs().max()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        '''
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = clean_tokens(tokens)

        pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')

        '''



        

        m, e, evolution = flip(model,
                              x=relevance, 
                             token_ids=input_ids, 
                             tokens=tokens,
                             y_true=targets, 
                             attention_mask = attention_mask,
                             fracs=fracs, 
                             flip_case=flip_case,
                             tokenizer=tokenizer,
                             device=device)

       
        M.append(m)
        E.append(e)
        EVOLUTION.append(evolution)


        M_str.append([str(elem) for elem in m])
        E_str.append([str(elem) for elem in e])
        EVOLUTION_str.append(evolution)
           
        all_flips["res"]= {'E':E, 'M':M, 'Evolution':EVOLUTION} 
        all_flips_str["res"]= {'E':E_str, 'M':M_str} 



    f, axs = plt.subplots(1, 2, figsize=(14, 8))
    for k, v in all_flips.items():
        print(len(v['M']))
        axs[0].plot(np.nanmean(v['M'], axis=0), label=k)
        axs[0].set_title('($y_0$-$y_p$)$^2$')
        axs[1].plot(np.nanmean(v['E'], axis=0), label=k)
        axs[1].set_title('logits$_k$')    
    plt.legend()

  

    f.savefig(f'{save_dir}/imdb_{flip_case}.png' , dpi=300)
    #update_json(f'imdb_{flip_case}.json', all_flips_str)
    update_json(f'{save_dir}/imdb_pert_res.json', {f'{flip_case}_AU_MSE': auc(fracs, np.nanmean(v['M'], axis=0)),
                                        f'{flip_case}_AU_AC': auc(fracs, np.nanmean(v['E'], axis=0))})

    #pickle.dump(all_flips, open(os.path.join(save_dir, 'all_flips_{}_imdb.p'.format(flip_case)), 'wb'))