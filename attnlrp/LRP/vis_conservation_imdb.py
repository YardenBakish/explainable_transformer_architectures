import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config

#from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from lxt.models.llama_PE_vis import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
import lxt.functional as lf

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

import matplotlib.pyplot as plt
import numpy as np
import json
MAPPER = {"llama_tiny_baseline":"Tiny-Llama", "llama_2_7b_baseline": "Llama-2-7b", "llama_2_7b_baseline2": "llama-2-7b-Quantized"}


def save_heatmap(values, tokens, figsize, title, save_path):
    fig, ax = plt.subplots(figsize=figsize)

    abs_max = abs(values).max()
    im = ax.imshow(values, cmap='bwr', vmin=-abs_max, vmax=abs_max)

    layers = np.arange(values.shape[-1])

    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(tokens)))

    ax.set_xticklabels(layers)
    ax.set_yticklabels(tokens)

    plt.title(title)
    plt.xlabel('Layers')
    plt.ylabel('Tokens')
    plt.colorbar(im)

    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def hidden_relevance_hook(module, input, output):
    if isinstance(output, tuple):
        output = output[0]
    module.hidden_relevance = output.detach().cpu()


import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--model-size', type=str,
                            choices=['llama_2_7b', 'llama_tiny'],
                           required = True,
                            help='')
    parser.add_argument('--variant', type=str,
                           default="baseline")
    parser.add_argument('--fract', type=float,
                            default=0.3,
                            help='')
    parser.add_argument('--pe', action='store_true')
    parser.add_argument('--reform', action='store_true')
    parser.add_argument('--mode', type=str,
                                choices=['generate', 'analyze','generate_analyze', 'analyze_all'],
                                required = True,
                                help='')
    parser.add_argument('--pe_only', action='store_true')
    parser.add_argument('--rule_matmul', action='store_true')

    parser.add_argument('--quant', action='store_true')
    parser.add_argument('--experimental', action='store_true')
    parser.add_argument('--sequence-length', type=int,
                           )
    parser.add_argument('--trained_model', type=str,)



    parser.add_argument('--no-padding', action='store_true')
    parser.add_argument('--ignore_start', action='store_true')


    parser.add_argument('--without-abs', action='store_true')
    parser.add_argument('--single-norm', action='store_true')


    args = parser.parse_args()
    args.dataset = 'imdb'
    #rgs.model_size = 'llama_tiny'
    
    if (args.pe_only and not args.pe) or (args.reform and not args.pe) or (args.rule_matmul and not args.pe):
        print("no")
        exit(1)

    config.get_config(args, pert = True)
    args.work_dir = f'visualizations/conservation_bars_imdb'
    ext_json_file = "_pe" if args.pe else ""
    if args.rule_matmul:
        ext_json_file = "_matmul_rule"
    if args.pe_only:
        ext_json_file = "_peOnly"
    if args.reform:
        ext_json_file = "_peReform"
    if args.experimental:
        ext_json_file = f"_experimental"
    if args.ignore_start:
        ext_json_file = f"{ext_json_file}_ignore_start"

    args.json_file = f'{args.work_dir}/{ext_json_file}.json'  

  
    os.makedirs(args.work_dir, exist_ok=True)
    return args



def analyze(args):
    with open(f"{args.json_file}", 'r') as file:
        data = json.load(file)
    res = data[f'{args.ext}']

    fig, ax = plt.subplots(figsize=(10, 6))
    sem = res["SEMANTIC"]
    pos = np.array(res["PE"])
    #pos[pos<0] = 0
    print(len(pos))
    print(len(sem))

    ax.bar(range(len(sem)), sem, label='AttnLRP',zorder=2)
    if args.pe:
        #pos_sums = [tensor.sum().item() for tensor in pos_relevance_trace]
        for i in range(len(pos)):
            if pos[i] > 0:
                ax.bar(i, pos[i], bottom=sem[i],color='#ff7f0e',  zorder=2)  # Stacked on top when positive
            else:
                ax.bar(i, pos[i], bottom=0,color='red',  zorder=2)
        #ax.bar(range(len(pos)), pos, bottom=sem, label='PE' ,zorder=2)
    #plt.xlabel('Layers')
    #plt.ylabel('T(R)')
    #ax.set_title('Conservation of Relevancy out of 100%\nMean across the IMDB dataset')
    ax.set_xticks([])
    ax.yaxis.grid(True, linestyle='-', alpha=0.5,zorder=0)

    ax.bar(0, 0, color='#ff7f0e', label='PE (Positive)', zorder=2)  # Placeholder for positive
    ax.bar(0, 0, color='red', label='PE (Negative)', zorder=2) 

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    plt.savefig(f"{args.work_dir}/{args.ext}.png", dpi=300, bbox_inches='tight')




def analyze_all(args):
    
    with open(f"{args.json_file}", 'r') as file:
        data = json.load(file)

    group_labels = []  

    grid_above = [[0 for i in range(3)] for j in range(3)]
    grif_below = [[0 for i in range(3)] for j in range(3)]


    i=0
    for k in data:
        print(k)
        group_labels.append(MAPPER[k])
        grid_above[i] = [elem for elem  in data[k]["PE"]]
        grif_below[i] = [elem for elem  in data[k]["SEMANTIC"]]
        i+=1


    bottom_lst = []
    upper_lst  = []

    for i in range(3):
        for j in range(3):
            upper_lst.append(grid_above[i][j])
            bottom_lst.append(grif_below[i][j])


    positions = []
    group_positions = [] 
    for i in range(len(bottom_lst)):
        group_idx = i // 3
        within_idx = i % 3
        # Each group starts at position group_idx * 4
        # Within a group, bars are at positions 0, 1, 2
        # This creates a gap of 1 unit between groups
        positions.append(group_idx * 4 + within_idx)
        if within_idx == 1:
            group_positions.append(positions[-1])

    group_labels2 = ["First", "Intermediate", "Penultimate"]
    group_labels2 = group_labels
    fig, ax = plt.subplots(figsize=(28, 22))
    group_labels = ["First", "Intermediate", "Penultimate"]
    bar_labels = group_labels * 3
    ax.set_xticks(positions, bar_labels, rotation=45, ha='right', fontsize=48)

    max_height = max([(b+u) for b, u in zip(bottom_lst, upper_lst)])
    label_height = max_height * 1.05  # Position labels slightly above the tallest bar

    for i, pos in enumerate(group_positions):
        ax.text(pos, label_height, group_labels2[i], ha='center', fontsize=51, fontweight='bold')

    #colors = [(0,0,0.9), (0,0,0.95) ,(0,0,1.0)]  # Colors to repeat
    #color_list = [colors[i % len(colors)] for i in range(len(bottom_lst))] 

    ax.bar(positions, bottom_lst, label='AttnLRP',zorder=2)

        #pos_sums = [tensor.sum().item() for tensor in pos_relevance_trace]
    ax.bar(positions,  upper_lst, bottom=bottom_lst,  label='PE Only' ,zorder=2)
    #plt.xlabel('Layers')
    #plt.ylabel('T(R)')
    #ax.set_title('Conservation of Relevancy out of 100%\nMean across the IMDB dataset')
    #ax.set_xticks([])
    ax.yaxis.grid(True, linestyle='-', alpha=0.5,zorder=0)
    ax.tick_params(axis='y', labelsize=58)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, markerscale=1, fontsize=45, frameon=False)

    plt.savefig(f"conservation_firstVSlastLast_tiny.png", dpi=300, bbox_inches='tight')


def generate(args):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


    #if args.model_size == 'llama_tiny': 
    #    model_checkpoint = "finetuned_models/imdb/llama_tiny/baseline/checkpoint_0/pytorch_model.bin"
    #if args.model_size == 'llama_2_7b':
    #    model_checkpoint = "finetuned_models/imdb/llama_2_7b/baseline/best_checkpoint/pytorch_model.bin"
    #    if args.variant == "baseline2":
    #        model_checkpoint = "finetuned_models/imdb/llama_2_7b/baseline2/best_checkpoint/pytorch_model.bin"


    PATH = args.original_models

    #print(PATH)
    #exit(1)

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

    #llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb_config, attn_implementation="eager")

    conf = llamaModel.config
    conf.num_labels = 2
    conf.pad_token_id = tokenizer.pad_token_id


    model = LlamaForSequenceClassification.from_pretrained(args.model_checkpoint, config = conf,  torch_dtype=torch.bfloat16, device_map="cuda")
    model.to(device)


    MAX_LEN = 512
    BATCH_SIZE = 1

    df = load_imdb()
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE,args.no_padding)

    # optional gradient checkpointing to save memory (2x forward pass)
    model.gradient_checkpointing_enable()

    # apply AttnLRP rules
    attnlrp.register(model)
    for layer in model.model.layers:
        layer.register_full_backward_hook(hidden_relevance_hook)

    count = 0
    total_sem_relevancy_per_layer = []
    total_pos_relevancy_per_layer = []
    for d in tqdm(test_data_loader):

        if count >=1000:
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
            identity_matrices   = None
            if args.rule_matmul:
                identity_matrices = model.identity_matrices
                identity_matrices = [x.requires_grad_() for x in identity_matrices] 
               

            outputs = model(
                    inputs_embeds = input_embeds.requires_grad_(),
                    position_embeddings = position_embeddings,
                    identity_matrices = identity_matrices,
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
            print("mismatch")
            continue
        print("MATTCH")
        
        count+=1

        relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision
        #relevance = relevance / relevance.abs().max()

        relevance_trace = [relevance]
        pos_relevance_trace = []
        
        for i, layer in enumerate(model.model.layers):
        
            relevance = layer.hidden_relevance[0].sum(-1)
            if args.pe:
                    if args.reform:
                        #print(position_embeddings[i][0].shape)
                        #print(position_embeddings[i][0].grad.shape)
                        #exit(1)
                        
                        if args.without_abs:

                            x1 = torch.matmul(position_embeddings[i][0].grad, position_embeddings[i][0].transpose(-1, -2)).sum(-1).cpu()[0] / (2*position_embeddings[i][0].shape[-1])
                            x2 = torch.matmul(position_embeddings[i][1].grad, position_embeddings[i][1].transpose(-1, -2)).sum(-1).cpu()[0] / (2*position_embeddings[i][0].shape[-1])
                  
                        else:
                        
                            x1 = torch.matmul(position_embeddings[i][0].grad.abs(), position_embeddings[i][0].transpose(-1, -2).abs()).sum(-1).cpu()[0] / (2*position_embeddings[i][0].shape[-1])
                            x2 = torch.matmul(position_embeddings[i][1].grad.abs(), position_embeddings[i][1].transpose(-1, -2).abs()).sum(-1).cpu()[0] / (2*position_embeddings[i][0].shape[-1])
                    
                    elif args.rule_matmul:
                        #print(identity_matrices[i].grad.float().shape)
                        x1= identity_matrices[i].grad.float().diagonal().cpu()
                        #print(x1.shape)
                        #print(model.config.num_hidden_layers)
                        x1 /= model.config.num_hidden_layers
                     
                        x2= 0.
                       
                    
                    else:
                        x1= position_embeddings[i][0].grad.float().sum(-1).cpu()[0]
                        x2= position_embeddings[i][1].grad.float().sum(-1).cpu()[0]

                    
                    curr_relevancy = x1 +x2
                    if args.ignore_start:
                        curr_relevancy = curr_relevancy[1:]
                
                    #print(curr_relevancy.shape)  
                    #exit(1)
                    #if args.without_abs:
                    #    pass
                    #else:
                    #    curr_relevancy = curr_relevancy.abs()
                    
                    pos_relevance_trace.append(curr_relevancy)

                    #print(curr_relevancy.shape)




            relevance_trace.append(relevance)

        pos_relevance_trace.append(torch.tensor(0.0))
        #sums = np.array([tensor.sum().item() for tensor in relevance_trace])
        #sums = (sums - sums.min()) / (sums.max() - sums.min())


        sums = [tensor.sum().item() for tensor in relevance_trace]
        pos_sums = [tensor.sum().item() for tensor in pos_relevance_trace]

        #pos_sums[0] = sum(pos_sums)

        total_sem_relevancy_per_layer.append(sums)
        total_pos_relevancy_per_layer.append(pos_sums)


 


    #TODO consider normalizing each sample before avareging
    total_sem_relevancy_per_layer =  [sum(x) / len(x) for x in zip(*total_sem_relevancy_per_layer)]
  
    if args.pe:
        total_pos_relevancy_per_layer =  [sum(x) / len(x) for x in zip(*total_pos_relevancy_per_layer)]


    total_sem_relevancy_per_layer = np.array(total_sem_relevancy_per_layer)
    #TODO: we need to change normalizeion to [0,1] (not what we are doing)
    #total_sem_relevancy_per_layer /= total_sem_relevancy_per_layer[-1]
    total_pos_relevancy_per_layer = np.array(total_pos_relevancy_per_layer)
    #TODO: technically not sure why we did one row below
    
    #total_pos_relevancy_per_layer /= model.config.num_hidden_layers
    if args.without_abs:
        pass
    else:
        total_pos_relevancy_per_layer[total_pos_relevancy_per_layer<0] = 0 
    total_pos_relevancy_per_layer = np.cumsum(total_pos_relevancy_per_layer[::-1])[::-1]

    total_pos_relevancy_per_layer = total_pos_relevancy_per_layer.tolist() 
    total_sem_relevancy_per_layer     = total_sem_relevancy_per_layer.tolist() 
    filename = args.json_file
    #filename = f'{args.work_dir}/res.json'
    update_json(f'{filename}',{f"{args.ext}": {f"PE":total_pos_relevancy_per_layer, "SEMANTIC": total_sem_relevancy_per_layer }} )




if __name__ == "__main__":
    args          = parse_args()
    if args.mode == "generate":
        generate(args)
    elif args.mode == "analyze_all":
        analyze_all(args)
    elif args.mode == "generate_analyze":
        generate(args)
        analyze(args)
    else:
        analyze(args)

       
'''

fig, ax = plt.subplots(figsize=(10, 6))

print(len(total_sem_relevancy_per_layer))
print(len(total_pos_relevancy_per_layer))
ax.bar(range(len(total_sem_relevancy_per_layer)), total_sem_relevancy_per_layer, label='Conventional relevancy',zorder=2)
if args.pe:
    #pos_sums = [tensor.sum().item() for tensor in pos_relevance_trace]
    ax.bar(range(len(total_sem_relevancy_per_layer)), total_pos_relevancy_per_layer, bottom=total_sem_relevancy_per_layer, label='PE relevancy' ,zorder=2)
#plt.xlabel('Layers')
#plt.ylabel('T(R)')
#ax.set_title('Conservation of Relevancy out of 100%\nMean across the IMDB dataset')
ax.set_xticks([])
ax.yaxis.grid(True, linestyle='-', alpha=0.5,zorder=0)

for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

plt.savefig(f"conservation_firstVSlast_{args.model_size}_{args.variant}.png", dpi=300, bbox_inches='tight')'''
#plt.savefig(f"conservation_firstVSlast_{args.model_size}_{args.variant}.png", dpi=300, bbox_inches='tight')

#relevance_trace = torch.stack(relevance_trace)
 

'''
categories = ['Top layer', 'First layer']
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(categories, total_sem_relevancy_per_layer, label='Conventional relevancy', width = 0.3, zorder=2)

ax.bar(categories, total_pos_relevancy_per_layer, label='PE relevancy',bottom=total_sem_relevancy_per_layer, width = 0.3, zorder=2)


ax.set_title('Conservation of relevancy out of 100%\nMean across the IMDB dataset', fontsize=14)
ax.set_ylabel('Percentage', fontsize=12)
#ax.set_ylim(0, 120) 
ax.yaxis.grid(True, linestyle='-', alpha=0.5,zorder=0)
ax.set_xlim(-0.5, len(categories) - 0.5)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.legend(loc='lower center',  frameon=False,  # Remove the legend border
          handlelength=1,  # Shorter handle length for squares
          handleheight=1, bbox_to_anchor=(0.5, -0.15), ncol=2)
#ax.set_yticks(np.arange(0, 121, 20))
plt.tight_layout()
plt.savefig(f"conservation_firstVSlast_{args.model_size}_{args.variant}.png", dpi=300, bbox_inches='tight')
'''