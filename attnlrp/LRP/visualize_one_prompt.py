import torch
from transformers import AutoTokenizer
from lxt.models.llama_PE import LlamaForCausalLM, attnlrp
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup

os.environ['TRANSFORMERS_CACHE'] = '/home/ai_center/ai_users/yardenbakish/'
os.environ['HF_HOME'] = '/home/ai_center/ai_users/yardenbakish/'

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




def save_all_heatmaps(values_list, tokens, figsize, title, save_path, titles):
    num_plots = len(values_list)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    for idx, values in enumerate(values_list):
        ax = axes[idx]
        abs_max = abs(values).max()
        im = ax.imshow(values, cmap='bwr', vmin=-abs_max, vmax=abs_max)
        layers = np.arange(values.shape[-1])
        ax.set_xticks(np.arange(len(layers)))
        ax.set_yticks(np.arange(len(tokens)))

        ax.set_xticklabels(layers,fontsize=30)
        ax.set_yticklabels(tokens,fontsize=30)

        ax.set_title(f'{titles[idx]}',fontsize=30)
        #ax.set_xlabel('Layers')
        #ax.set_ylabel('Tokens')
        fig.colorbar(im, ax=ax).ax.tick_params(labelsize=30)
        #plt.title(title)
        #plt.xlabel('Layers')
        #plt.ylabel('Tokens')
        #plt.colorbar(im).ax.tick_params(labelsize=15) 

    plt.tight_layout()
    #plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def hidden_relevance_hook(module, input, output):
    if isinstance(output, tuple):
        output = output[0]
    module.hidden_relevance = output.detach().cpu()

parser = argparse.ArgumentParser(description='Train a segmentation')

parser.add_argument('--pe', action='store_true')
parser.add_argument('--norm', action='store_true')




args = parser.parse_args()
save_dir = 'visualizations/single_sample'
os.makedirs(save_dir, exist_ok=True)

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent numerical overflow
)


# load model & apply AttnLRP
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",quantization_config=quantization_config,  torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model.eval()
attnlrp.register(model)

# apply hooks
for layer in model.model.layers:
    layer.register_full_backward_hook(hidden_relevance_hook)

# forward & backard pass
#prompt_response = """\
#    I am faster than my brother but not as much as my syster. I am slower than my"""

prompt_response = """\
    I am faster than my brother but not as much as my syster. I am slower than my"""


#prompt_response = """\
#    Although it's raining, we will go to the"""

#prompt_response = """\
#The following is multiple choice question (with answers).
#
#A student in an empty classroom shouts, "Hello!" Which best explains what the student hears after the shout?
#A. an increased loudness of sound
#B. a reflection of sound
#C. an increased frequency of sound
#D. a refraction of sound
#
#Please make sure to answer (A,B,C, or D)
#Answer is:"""
input_ids = tokenizer(prompt_response, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)

if args.pe:
    position_ids = torch.arange(
            0.0, input_ids.shape[1], device=input_embeds.device,  requires_grad=True,
           dtype=torch.float32
        ).reshape(1, -1)
    position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids) for i in range(model.config.num_hidden_layers)]
    position_embeddings = [(x[0].requires_grad_(),x[1].requires_grad_()) for x in  position_embeddings ]
    output_logits = model(inputs_embeds=input_embeds.requires_grad_(),position_embeddings = position_embeddings, use_cache=False).logits

else:
    output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits

max_logits, max_indices = torch.max(output_logits[:, -1, :], dim=-1)
max_logits.backward(max_logits)

print("Prediction:", tokenizer.convert_ids_to_tokens(max_indices))
relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision
if args.norm:
    relevance = relevance / relevance.abs().max()
# trace relevance through layers
relevance_trace = [relevance]
pos1_trace = []
pos2_trace = []


for i, layer in enumerate(model.model.layers):
    relevance = layer.hidden_relevance[0].sum(-1)
    if args.pe:

        pos1= position_embeddings[i][0].grad.float().sum(-1).cpu()[0]
        pos2= position_embeddings[i][1].grad.float().sum(-1).cpu()[0]
        if args.norm:
            pos1 = pos1 / pos1.abs().max()
            pos2 = pos2 / pos2.abs().max()

        
        pos1_trace.append(pos1)
        pos2_trace.append(pos2)

    # normalize relevance at each layer between -1 and 1
    if args.norm:
        relevance = relevance / relevance.abs().max()
    relevance_trace.append(relevance)

relevance_trace = torch.stack(relevance_trace)
if args.pe:
    pos1_trace = torch.stack(pos1_trace)
    pos2_trace = torch.stack(pos2_trace)
  
    relevance_trace = [relevance_trace,pos1_trace,pos2_trace]


tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
name_path = "trace.png" if args.pe == False else "trace_pe.png"


name_path = f"norm_{name_path}" if args.norm else name_path


#remove here
name_path = f"2_{name_path}"
#for i in range(len(relevance_trace)):
#    smallest_20 = torch.topk(relevance_trace[i], 7, dim=1, largest=False).values  # Smallest 20
#    largest_20 = torch.topk(relevance_trace[i], 7, dim=1, largest=True).values  # Largest 20
#    relevance_trace[i] = torch.cat((smallest_20, largest_20), dim=1)



if args.pe:
    save_all_heatmaps([x.float().numpy().T for x in relevance_trace], tokens, (60, 20), f"Latent Relevance Trace (Normalized)", f'{save_dir}/{name_path}', ["AttnLRP","PE1","PE2"])
else:
    save_heatmap(relevance_trace.float().numpy().T, tokens, (20, 10), f"Latent Relevance Trace (Normalized)", f'{save_dir}/{name_path}')