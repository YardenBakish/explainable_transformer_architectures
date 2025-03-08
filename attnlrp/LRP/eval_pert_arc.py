import os
os.environ['TRANSFORMERS_CACHE'] = '/home/ai_center/ai_users/yardenbakish/'
os.environ['HF_HOME'] = '/home/ai_center/ai_users/yardenbakish/'

ANSWERS  = {'A': 0, 'B': 1, 'C': 2,'D': 3}


import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config
import matplotlib.pyplot as plt
from lxt.models.llama_PE import LlamaForCausalLM, attnlrp
from helper_scripts.helper_functions import update_json
from attDatasets.ai2_arc import load_ai2_arc, AI2_ARC_Dataset, create_data_loader
from tqdm import tqdm
from lxt.utils import pdf_heatmap, clean_tokens
from utils import flip_arc, get_latest_checkpoint
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
                       required = True,
                        help='')

parser.add_argument('--fract', type=float,
                        default=0.3,
                        help='')
parser.add_argument('--pe', action='store_true')
parser.add_argument('--reform', action='store_true')
parser.add_argument('--should-keep', action='store_true')


parser.add_argument('--pe_only', action='store_true')
parser.add_argument('--quant', action='store_true')



args = parser.parse_args()


if (args.pe_only and not args.pe) or (args.reform and not args.pe):
    print("no")
    exit(1)


path = "meta-llama/Llama-3.1-8B-Instruct"

if args.model_size == 'llama_tiny':
    path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
if args.model_size == 'llama_2_7b':
    path = f"original_models/{args.model_size}/vanilla" 


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent numerical overflow
)

model_dir = f"{args.model_size}" if args.quant == False else f"{args.model_size}_QUANT"
model_dir = f"{model_dir}_KEEP" if args.should_keep == True else model_dir

save_dir = f'finetuned_models/{model_dir}/pert_results_arc'
os.makedirs(save_dir, exist_ok=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


MAX_LEN = 512
BATCH_SIZE = 1
#print(path)
#exit(1)
if args.quant:
    model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,quantization_config=quantization_config, device_map="cuda", attn_implementation="eager", low_cpu_mem_usage = True)
#model = LlamaForCausalLM.from_pretrained(path, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map="cuda")
else:
    model = LlamaForCausalLM.from_pretrained(path, device_map="cuda", torch_dtype=torch.bfloat16,   attn_implementation="eager")



tokenizer = AutoTokenizer.from_pretrained(path)


#model.to(device)


model.gradient_checkpointing_enable()
attnlrp.register(model)

df = load_ai2_arc()
test_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

UNK_token = tokenizer.unk_token_id
fracs = np.linspace(0.,1.,11)
for flip_case in ['generate', 'pruning']:
    samples_num = 0

    all_flips = {}
    all_flips_str = {}

    count = 0

    M, E, E_LOGIT, INSERT_LEAST_IMPORTANT, INSERT_LEAST_IMPORTANT_LOGIT,   EVOLUTION = [],[], [], [],[], []
    M_str,E_str, EVOLUTION_str = [],[], []

    for i, d in enumerate(tqdm(test_data_loader)):
    

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)


        input_embeds = model.get_input_embeddings()(input_ids)

        if args.pe:
            position_ids = torch.arange(
                    0.0, input_ids.shape[1], device=input_embeds.device,  requires_grad=True,
                   dtype=torch.float32
                ).reshape(1, -1)

            position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids) for i in range(model.config.num_hidden_layers)]
            position_embeddings = [(x[0].requires_grad_(),x[1].requires_grad_()) for x in  position_embeddings ]

            output_logits = model(
                inputs_embeds = input_embeds.requires_grad_(),
                position_embeddings = position_embeddings,
                #input_ids=input_ids,
                use_cache=False,
                #attention_mask=attention_mask
              ).logits
        else:
            output_logits = model(
                inputs_embeds = input_embeds.requires_grad_(),
                use_cache=False,
                #attention_mask=attention_mask
              ).logits
    
        #print(output_logits[0, -1, :].shape)
        #exit(1)
        max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)
       
        next_token_id = max_indices.item() 
        next_token = tokenizer.convert_ids_to_tokens(next_token_id)
        if next_token.replace("▁", "") not in ANSWERS:
            continue
        if targets.item() !=  ANSWERS[next_token.replace("▁", "")]:
            #print(f"{i}, {flip_case}, {samples_num}, {targets.item()}, {next_token}" )
            continue
        samples_num+=1
        #if (samples_num >=20):
        #    break
        max_logits.backward(max_logits)
        relevance = input_embeds.grad.float().sum(-1).cpu()[0] 
        relevance = relevance / relevance.abs().max()

        if args.pe:
            acc_relevancy = 0.
            for pos_embed in position_embeddings:
                for i in range(2):
                    if args.reform:
                        curr_relevancy = torch.matmul(pos_embed[i].grad.abs(), pos_embed[i].transpose(-1, -2).abs()).detach()
                        curr_relevancy = curr_relevancy.float().sum(-1).cpu()[0]
                    else:
                        curr_relevancy = pos_embed[i].grad.float().sum(-1).cpu()[0]
                    curr_relevancy = curr_relevancy / curr_relevancy.abs().max()
                    acc_relevancy += curr_relevancy.abs()

            acc_relevancy = (acc_relevancy - acc_relevancy.min()) / (acc_relevancy.max() - acc_relevancy.min())
            if args.pe_only:
                relevance = acc_relevancy.detach()
            else:
                relevance+=acc_relevancy
                relevance = relevance / relevance.abs().max()




        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        #print(tokens)
        #exit(1)
        '''
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = clean_tokens(tokens)

        pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')

        '''



        

        m, e, e_logit, evolution, less_relevantArr, less_relevantArr_logit = flip_arc(model,
                              x=relevance, 
                              highest_idx = next_token_id,
                              logit0 = max_logits.detach().item(),
                             token_ids=input_ids, 
                             tokens=tokens,
                             y_true=targets, 
                             should_keep = args.should_keep,
                             attention_mask = attention_mask,
                             fracs=fracs, 
                             flip_case=flip_case,
                             tokenizer=tokenizer,
                             device=device,
                             )

       
        M.append(m)
        E.append(e)
        E_LOGIT.append(e_logit)
        if less_relevantArr != []:
            INSERT_LEAST_IMPORTANT.append(less_relevantArr)
            INSERT_LEAST_IMPORTANT_LOGIT.append(less_relevantArr_logit)


        EVOLUTION.append(evolution)


        M_str.append([str(elem) for elem in m])
        E_str.append([str(elem) for elem in e])
        EVOLUTION_str.append(evolution)
           
        all_flips["res"]= {'E':E, 'M':M, 'E_LOGIT': E_LOGIT, 'INSERT_LEAST_IMPORTANT': INSERT_LEAST_IMPORTANT, 'INSERT_LEAST_IMPORTANT_LOGIT': INSERT_LEAST_IMPORTANT_LOGIT,    'Evolution':EVOLUTION} 
        all_flips_str["res"]= {'E':E_str, 'M':M_str} 



    f, axs = plt.subplots(1, 2, figsize=(14, 8))
    for k, v in all_flips.items():
        print(len(v['M']))
        axs[0].plot(np.nanmean(v['M'], axis=0), label=k)
        axs[0].set_title('($y_0$-$y_p$)$^2$')
        axs[1].plot(np.nanmean(v['E'], axis=0), label=k)
        axs[1].set_title('logits$_k$')    
    plt.legend()

  
    pe = "_pe" if args.pe else ""
    if args.pe_only:
        pe = "_peOnly"
    if args.reform:
        pe = "_peReform"
    f.savefig(f'{save_dir}/vis_{flip_case}_{pe}.png' , dpi=300)


    if flip_case == "generate":
        f, axs = plt.subplots(1, 2, figsize=(14, 8))
        for k, v in all_flips.items():
            axs[0].plot(np.nanmean(v['E_LOGIT'], axis=0), label=k)
            axs[0].set_title('MERF_INSERTION')
            axs[1].plot(np.nanmean(v['INSERT_LEAST_IMPORTANT_LOGIT'], axis=0), label=k)
            axs[1].set_title('LERF_INSERTION')    
        plt.legend()

  
        pe = "_pe" if args.pe else ""
        if args.pe_only:
            pe = "_peOnly"
        if args.reform:
            pe = "_peReform"
        f.savefig(f'{save_dir}/logits_{flip_case}_{pe}.png' , dpi=300)

        update_json(f'{save_dir}/imdb_pert_res_{pe}.json', {f'{flip_case}_AU_MSE': auc(fracs, np.nanmean(v['M'], axis=0)), 'samples_num': samples_num, 'pe': args.pe, 'pe_only': args.pe_only, 'reform': args.reform, 'quant': args.quant,
                                        f'{flip_case}_AU_AC': auc(fracs, np.nanmean(v['E'], axis=0)), f'MERF': auc(fracs, np.nanmean(v['E_LOGIT'], axis=0)), f'LERF': auc(fracs, np.nanmean(v['INSERT_LEAST_IMPORTANT_LOGIT'], axis=0))})
        continue

    #update_json(f'imdb_{flip_case}.json', all_flips_str)
    update_json(f'{save_dir}/imdb_pert_res_{pe}.json', {f'{flip_case}_AU_MSE': auc(fracs, np.nanmean(v['M'], axis=0)), 'samples_num': samples_num, 'pe': args.pe, 'pe_only': args.pe_only, 'reform': args.reform, 'quant': args.quant,
                                        f'{flip_case}_AU_AC': auc(fracs, np.nanmean(v['E'], axis=0))})

    #pickle.dump(all_flips, open(os.path.join(save_dir, 'all_flips_{}_imdb.p'.format(flip_case)), 'wb'))