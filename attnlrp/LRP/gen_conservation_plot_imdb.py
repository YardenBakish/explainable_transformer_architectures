import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager
#from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from lxt.models.llama_PE import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
import os
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--model-size', type=str,
                            choices=['llama_2_7b', 'llama_tiny'],
                            default = 'llama_tiny',
                           # required = True,
                            help='')
    parser.add_argument('--variant', type=str,
                           default="baseline")
    parser.add_argument('--exp-name', type=str,
                           default="baseline")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pe', action='store_true')
    parser.add_argument('--reform', action='store_true')

    parser.add_argument('--quant', action='store_true')
    parser.add_argument('--mode', type=str,
                            choices=['generate', 'analyze'],
                            required = True,
                            help='')
    

    parser.add_argument('--without-abs', action='store_true')
    parser.add_argument('--single-norm', action='store_true')
    parser.add_argument('--ignore_start', action='store_true')

    parser.add_argument('--trained_model', type=str,)

    parser.add_argument('--sequence-length', type=int,
                           )
    args = parser.parse_args()
    args.dataset = 'imdb'
 

    #rgs.model_size = 'llama_tiny'
    config.get_config(args, pert = True)
    args.work_dir = f'visualizations/conservation_scatter_imdb/{args.ext}'
    if args.exp_name:
        args.work_dir = f'{args.work_dir}/{args.exp_name}'


    os.makedirs(args.work_dir, exist_ok=True)
    return args



def analyze(args):
    res_file =  f'{args.work_dir}/data.json'
    save_file = f'{args.work_dir}/conservation_plot.png'


    with open(res_file, 'r') as file:
        data = json.load(file)
    res = data
    a_values = res['logits']
    b_values = res['relevance']  
    c_values = res['relevance_plus_pos']   
    # Create a figure and axis  
    plt.figure(figsize=(10, 8)) 
    ax = plt.gca()
    plt.scatter(a_values, b_values,  s=5, label='LRP')
    plt.scatter(a_values, c_values,  s=5, label='LRP+PE')

    a_min, a_max = min(a_values), max(a_values)
    margin = (a_max - a_min) * 0.1  # 10% margin
    plt.xlim(0.0, a_max + margin)
    plt.ylim(0.0, a_max + margin)

    tick_count = 4
    tick_step = (a_max - -2) / (tick_count - 1)
    tick_positions = np.arange(0.0, a_max + tick_step, tick_step)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    ax.tick_params(axis='x', labelsize=12)  # Very large font size for x-axis values
    ax.tick_params(axis='y', labelsize=12)  # Normal font size for y-axis values

    #plt.ylim(min(min(b_values), min(c_values)) - margin, max(max(b_values), max(c_values)) + margin)
    plt.xlabel(r'output $f$', fontsize=20)
    plt.ylabel(r'$\Sigma_{i} R(x_i)$', fontsize=20)


    line_range = np.linspace(0.0, a_max, 100)

    # Add the y=x line
    plt.plot(line_range, line_range, color='black', linestyle='-', linewidth=1)

    #plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(save_file)


def generate(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    PATH = args.original_models

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



    if args.quant:
        llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, local_files_only = True, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb_config, attn_implementation="eager")
    else:
        llamaModel = LlamaForSequenceClassification.from_pretrained(PATH,  device_map="cuda",   attn_implementation="eager")


    conf = llamaModel.config
    conf.num_labels = 2
    conf.pad_token_id = tokenizer.pad_token_id


    model = LlamaForSequenceClassification.from_pretrained(args.model_checkpoint, config = conf,  torch_dtype=torch.bfloat16, device_map="cuda")
    model.to(device)


    MAX_LEN = 512
    BATCH_SIZE = 1

    df = load_imdb()
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # optional gradient checkpointing to save memory (2x forward pass)
    model.gradient_checkpointing_enable()

    # apply AttnLRP rules
    attnlrp.register(model)
    res = []
    count = 0
    for i, d in enumerate(tqdm(test_data_loader)):

        if count > 1000:
            break
        
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)


        input_embeds = model.get_input_embeddings()(input_ids)
        if args.pe:
            position_ids = torch.arange(
                        0.0, input_ids.shape[1], device=input_embeds.device,  requires_grad=True,
                       dtype=torch.float32
                    ).reshape(1, -1)

            position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids) for t in range(model.config.num_hidden_layers)]
            position_embeddings = [(x[0].requires_grad_(),x[1].requires_grad_()) for x in  position_embeddings ]

            outputs = model(
                    inputs_embeds = input_embeds.requires_grad_(),
                    position_embeddings = position_embeddings,
                    #input_ids=input_ids,
                    use_cache=False,
                    attention_mask=attention_mask
                  )['logits']
        else:
            outputs = model(
                    inputs_embeds = input_embeds.requires_grad_(),
                    use_cache=False,
                    attention_mask=attention_mask
                  )['logits']


        max_logits, max_indices = torch.max(outputs, dim=1)
        max_logits.backward(max_logits)

        next_token_id = max_indices.item() 
        relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision

        if next_token_id != targets.item():
            continue
        count+=1
        relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision
        #relevance = relevance / relevance.abs().max()
        relevance = relevance.sum()
        if args.pe:
            pos_relevancy = 0.
            for pos_embed in position_embeddings:
                if args.reform:
                    if args.without_abs:

                        x1 = torch.matmul(pos_embed[0].grad, pos_embed[0].transpose(-1, -2)).sum(-1).cpu()[0]  / (2*pos_embed[0].shape[-1])
                        x2 = torch.matmul(pos_embed[1].grad, pos_embed[1].transpose(-1, -2)).sum(-1).cpu()[0]  / (2*pos_embed[0].shape[-1])
                    else:
                        x1 = torch.matmul(pos_embed[0].grad.abs(), pos_embed[0].transpose(-1, -2).abs()).sum(-1).cpu()[0]  / (2*pos_embed[0].shape[-1])
                        x2 = torch.matmul(pos_embed[1].grad.abs(), pos_embed[1].transpose(-1, -2).abs()).sum(-1).cpu()[0]  / (2*pos_embed[0].shape[-1])
                    
                    pe_per_layer = x1+x2
                    if args.ignore_start:
                        pe_per_layer = pe_per_layer[1:]
                    pos_relevancy += (pe_per_layer).sum()
                    #print(x1.shape)
                    #exit(1)
                else:
                    x1= pos_embed[0].grad.float().sum(-1).cpu()[0]
                    x2= pos_embed[1].grad.float().sum(-1).cpu()[0]
                    pe_per_layer = x1+x2

                    if args.ignore_start:
                        pe_per_layer = pe_per_layer[1:]
                    if args.without_abs:
                        pos_relevancy += (pe_per_layer.sum())
                    else:
                        pos_relevancy += (pe_per_layer.sum()).clamp(min=0) #(x1+x2).sum()
            pos_relevancy.detach_().cpu()
            for pos_embed in position_embeddings:
                pos_embed[0].detach_()
                pos_embed[1].detach_()
        max_logits.detach_().cpu() 
        relevance.detach_().cpu()
        max_indices.detach_().cpu()
        model.zero_grad()

        if args.pe:
            res.append([max_logits,relevance+pos_relevancy ])

        else:
            res.append([max_logits,relevance])
    
    save_file = f'{args.work_dir}/data.json'
    logits = [item[0].item() for item in res]
    relevance = [item[1].item() for item in res]
    if args.pe:
        update_json(save_file, {'logits':logits, 'relevance_plus_pos': relevance})
    else:
        update_json(save_file, {'logits':logits,'relevance':relevance})




    #exit(1)    
    #tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    #tokens = clean_tokens(tokens)

    #print(tokens)
    #exit(1)
    #tokens = []
    #pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')




if __name__ == "__main__":
    args          = parse_args()
    if args.mode == "generate":
        print(args.mode)
        generate(args)
    else:
        analyze(args)
       